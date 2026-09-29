"""EarthBased Vendor Copilot: intent routing, structured + vector retrieval, grounded answers.

Every answer is built from retrieved evidence. A deterministic composer always
works; if ANTHROPIC_API_KEY is set, an LLM rewrites the answer from the same
evidence, and its output is rejected if it fails the grounding check.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from .config import ALL_SCORE_KEYS, FACTOR_KEYS, FACTOR_LABELS, FACTOR_SHORT, STRATEGIC_KEY
from .recommendation import explain_vendor_factor, ordinal, prioritise_vendors, semantic_match

VENDOR_ID_RE = re.compile(r"\bV\d{3}\b", re.IGNORECASE)
CAVEAT = ("Caveat: scores are relative rankings within this synthetic dataset and indicate where to look, "
          "not what to decide. Validate with vendor conversations and operational data before acting.")
CERTAINTY_PATTERNS = [r"\bis guaranteed\b", r"\bare guaranteed\b", r"\bguaranteed to\b", r"\bdefinitely\b",
                      r"\bcertainly\b", r"\bwithout (a )?doubt\b", r"\b100% (certain|sure)\b", r"\bundoubtedly\b"]
FACTOR_KEYWORDS = {
    "operational_reliability": ["operational reliability", "reliability", "reliable", "operational", "fulfilment",
                                "fulfillment", "delivery"],
    "commercial_potential": ["commercial potential", "commercial", "demand", "sales", "gmv", "traction"],
    "catalogue_market_fit": ["catalogue fit", "catalog fit", "catalogue", "catalog", "assortment"],
    "customer_product_fit": ["customer fit", "product fit", "customer satisfaction", "ratings", "rating", "loyalty", "repeat"],
}
POSITIVE_WORDS = ["high", "strong", "good", "great", "excellent", "reliable", "above"]
NEGATIVE_WORDS = ["weak", "low", "poor", "issues", "issue", "problems", "risk", "below", "limited", "struggling"]
CATEGORY_ALIASES = [
    ("packaging", "Sustainable Packaging"), ("cutlery", "Sustainable Packaging"), ("container", "Sustainable Packaging"),
    ("personal care", "Personal Care"), ("skincare", "Personal Care"), ("soap", "Personal Care"),
    ("shampoo", "Personal Care"), ("toothbrush", "Personal Care"), ("kitchen", "Home & Kitchen"),
    ("cleaning", "Home & Kitchen"), ("pantry", "Organic Food & Pantry"), ("organic food", "Organic Food & Pantry"),
    ("grocery", "Organic Food & Pantry"), ("snack", "Organic Food & Pantry"), ("apparel", "Apparel & Accessories"),
    ("clothing", "Apparel & Accessories"), ("decor", "Home Decor"), ("planter", "Home Decor"), ("baby", "Baby & Kids"),
    ("kids", "Baby & Kids"), ("toy", "Baby & Kids"), ("stationery", "Stationery & Office"),
    ("notebook", "Stationery & Office"), ("office supplies", "Stationery & Office"),
]


@dataclass
class Evidence:
    table: pd.DataFrame
    notes: list[str]
    sufficient: bool
    vendor_ids: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        body = self.table.to_string(index=False) if not self.table.empty else "(no records)"
        return body + "\n" + "\n".join(self.notes)


@dataclass
class CopilotResponse:
    question: str
    intent: str
    answer: str
    evidence: pd.DataFrame
    notes: list[str]
    sufficient: bool
    mode: str
    grounded: bool
    grounding_issues: list[str]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    llm_error: str | None = None


def parse_factor_conditions(question: str) -> dict[str, str]:
    """Extract {factor: 'high'|'low'} conditions such as 'high catalogue fit but weak reliability'."""
    conds: dict[str, str] = {}
    for chunk in re.split(r"\bbut\b|\band\b|\bwith\b|\byet\b|,|;|\bwhile\b", question.lower()):
        for key, kws in FACTOR_KEYWORDS.items():
            if any(k in chunk for k in kws):
                if any(re.search(rf"\b{w}\b", chunk) for w in NEGATIVE_WORDS):
                    conds[key] = "low"
                elif any(re.search(rf"\b{w}\b", chunk) for w in POSITIVE_WORDS):
                    conds[key] = "high"
                break
    return conds


def detect_category(question: str, categories: list[str]) -> str | None:
    ql = question.lower()
    for cat in categories:
        if cat.lower() in ql:
            return cat
    for alias, cat in CATEGORY_ALIASES:
        if alias in ql and cat in categories:
            return cat
    return None


def detect_intent(question: str) -> str:
    """Rule-based intent router (transparent and testable)."""
    ql = question.lower()
    ids = VENDOR_ID_RE.findall(question)
    if ids and re.search(r"similar|comparable|alternative|like v\d", ql):
        return "similar_vendors"
    if ids:
        return "vendor_explanation"
    if re.search(r"revenue|profit|valuation|actual earthbased|guarantee|succeed|salary|share price", ql):
        return "out_of_scope"
    if re.search(r"categor", ql) and re.search(r"coverage|gap|limited|more vendor|under-?served|opportunit|supply", ql):
        return "category_gaps"
    if re.search(r"premium|high[- ]value customer|affluent|luxury", ql):
        return "premium_segment"
    if parse_factor_conditions(ql):
        return "factor_contrast"
    if re.search(r"prioriti[sz]e|shortlist|top vendors|best vendors|investigate|focus on|onboard more|should we", ql):
        return "prioritisation"
    return "semantic_search"


def _numbers(text: str) -> list[float]:
    out = []
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _close(x: float, y: float) -> bool:
    tol = max(0.06, 0.006 * abs(x))
    return abs(x - y) <= tol or abs(x - 100 * y) <= tol or abs(x / 100 - y) <= tol / 100 + 1e-9


def check_grounding(answer: str, evidence: Evidence, question: str = "") -> list[str]:
    """Flag vendor IDs, numbers or certainty language in an answer that the evidence does not support."""
    issues: list[str] = []
    allowed = {v.upper() for v in evidence.vendor_ids} | {v.upper() for v in VENDOR_ID_RE.findall(question)}
    mentioned = {v.upper() for v in VENDOR_ID_RE.findall(answer)}
    for vid in sorted(mentioned - allowed):
        issues.append(f"Vendor {vid} is not in the retrieved evidence.")
    low = answer.lower()
    for pat in CERTAINTY_PATTERNS:
        if re.search(pat, low):
            issues.append(f"Certainty language matched '{pat}'.")
    ctx = [abs(n) for n in _numbers(evidence.as_text())]
    for x in _numbers(VENDOR_ID_RE.sub(" ", answer)):
        if x <= 10 and float(x).is_integer():
            continue
        if 1900 <= x <= 2100 and float(x).is_integer():
            continue
        if not any(_close(x, y) for y in ctx):
            issues.append(f"Number {x:g} does not appear in the evidence.")
    return issues


class VendorCopilot:
    """Grounded question answering over the marketplace system."""

    SYSTEM_PROMPT = (
        "You are the EarthBased Vendor Copilot, an analytics assistant for a marketplace team. Answer ONLY from the "
        "CONTEXT provided. Rules: (1) never invent vendors, metrics or numbers; copy numbers exactly as they appear "
        "in the context; (2) if the context is insufficient, say so; (3) mention vendor IDs only if they appear in "
        "the context; (4) never present a recommendation as certain; (5) keep the answer under 180 words and end "
        "with a one-sentence caveat. The data is synthetic.")

    def __init__(self, system, use_llm: bool | None = None, model: str | None = None):
        self.system = system
        self.P = system.profile.set_index("vendor_id")
        self.categories = sorted(self.P["category"].unique())
        has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        self.use_llm = has_key if use_llm is None else (use_llm and has_key)
        self.model = model or os.getenv("EB_LLM_MODEL", "claude-sonnet-4-5")

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> CopilotResponse:
        t0 = time.perf_counter()
        question = (question or "").strip()
        intent = detect_intent(question) if question else "out_of_scope"
        evidence, draft = getattr(self, f"_handle_{intent}")(question)
        answer, mode, tin, tout, cost, err = draft, "deterministic", 0, 0, 0.0, None
        if self.use_llm and evidence.sufficient:
            try:
                text, tin, tout = self._call_llm(question, evidence, draft)
                cost = tin / 1e6 * float(os.getenv("EB_LLM_INPUT_COST_PER_MTOK", "3.0")) + \
                    tout / 1e6 * float(os.getenv("EB_LLM_OUTPUT_COST_PER_MTOK", "15.0"))
                llm_issues = check_grounding(text, evidence, question)
                if llm_issues:
                    err = "LLM answer rejected by grounding check: " + "; ".join(llm_issues[:3])
                else:
                    answer, mode = text, "llm"
            except Exception as exc:  # noqa: BLE001 - network/API failures fall back
                err = f"{type(exc).__name__}: {exc}"[:300]
        issues = check_grounding(answer, evidence, question)
        return CopilotResponse(question, intent, answer, evidence.table, evidence.notes, evidence.sufficient, mode,
                               not issues, issues, (time.perf_counter() - t0) * 1000, tin, tout, cost, err)

    # ------------------------------------------------------------------ helpers
    def _call_llm(self, question: str, evidence: Evidence, draft: str) -> tuple[str, int, int]:
        user = (f"QUESTION:\n{question}\n\nCONTEXT (retrieved records):\n{evidence.as_text()}\n\n"
                f"DETERMINISTIC DRAFT (derived from the same context):\n{draft}\n\n"
                "Write the final answer for a marketplace operations team.")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=30, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 600, "system": self.SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        usage = data.get("usage", {})
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    def _vendor_table(self, ids: list[str], extra: dict[str, str] | None = None) -> pd.DataFrame:
        cols = ["vendor_id", "vendor_name", "category", "city", "segment"] + [f"{k}_pct" for k in ALL_SCORE_KEYS]
        extra = extra or {}
        t = self.P.loc[ids].reset_index()[cols + list(extra)]
        rename = {f"{k}_pct": f"{FACTOR_SHORT[k]} pct" for k in ALL_SCORE_KEYS}
        rename.update(extra)
        return t.rename(columns=rename)

    @staticmethod
    def _insufficient(message: str, notes: list[str] | None = None) -> tuple[Evidence, str]:
        return Evidence(pd.DataFrame(), notes or [], False, []), f"Insufficient evidence. {message}"

    def _line(self, vid: str, keys: list[str]) -> str:
        r = self.P.loc[vid]
        parts = ", ".join(f"{FACTOR_LABELS[k]} {ordinal(r[f'{k}_pct'])} pct" for k in keys)
        return f"- **{vid} {r.vendor_name}** ({r.category}, {r.segment}): {parts}"

    # ------------------------------------------------------------------ handlers
    def _handle_vendor_explanation(self, q: str):
        vid = VENDOR_ID_RE.findall(q)[0].upper()
        if vid not in self.P.index:
            return self._insufficient(f"Vendor {vid} does not exist in the marketplace dataset, so there are no "
                                      "records to assess. Check the vendor ID.", [f"{vid} not found."])
        ql, r = q.lower(), self.P.loc[vid]
        focus = None
        for key, pat in [(STRATEGIC_KEY, r"strateg"), ("commercial_potential", r"commerc|sales|gmv|demand"),
                         ("operational_reliability", r"reliab|operational|fulfil"),
                         ("catalogue_market_fit", r"catalog"), ("customer_product_fit", r"customer|rating|loyal")]:
            if re.search(pat, ql):
                focus = key
                break
        if focus is None:
            focus = max(FACTOR_KEYS, key=lambda k: r[f"{k}_pct"])
        drivers = explain_vendor_factor(self.system, vid, focus)
        facts = [{"vendor_id": vid, "item": "Segment", "value": r.segment, "reference": r.segment_description}]
        for k in ALL_SCORE_KEYS:
            facts.append({"vendor_id": vid, "item": f"{FACTOR_LABELS[k]} percentile", "value": str(int(r[f'{k}_pct'])), "reference": ""})
        for d in drivers.itertuples(index=False):
            facts.append({"vendor_id": vid, "item": f"Driver: {d.indicator}", "value": d.vendor_value,
                          "reference": f"platform median {d.platform_median}; weight {d.weight_or_loading}; {d.effect}"})
        pct = int(r[f"{focus}_pct"])
        lines = [f"**{vid} {r.vendor_name}** ({r.category}, {r.city}) sits in {r.segment}: {r.segment_description}.", "",
                 f"Its {FACTOR_LABELS[focus]} score is at the {ordinal(pct)} percentile of vendors."]
        if focus == STRATEGIC_KEY:
            lines.append("Strategic Value is a data-weighted composite (first principal component) of the four factor "
                         "scores, category importance and retention contribution.")
        if pct < 60 and re.search(r"promising|important|strong|why", ql):
            lines.append(f"That is not a top-tier position, so the data does not strongly support describing it as "
                         f"leading on {FACTOR_LABELS[focus].lower()}.")
        lines.append("The indicators contributing most to that score:")
        for d in drivers.itertuples(index=False):
            lines.append(f"- {d.indicator}: {d.vendor_value} vs platform median {d.platform_median} "
                         f"(weight {d.weight_or_loading}, {d.effect})")
        others = [k for k in ALL_SCORE_KEYS if k != focus]
        lines.append("Other scores: " + ", ".join(f"{FACTOR_LABELS[k]} {ordinal(r[f'{k}_pct'])}" for k in others) + ".")
        lines += ["", CAVEAT]
        return Evidence(pd.DataFrame(facts), [], True, [vid]), "\n".join(lines)

    def _handle_similar_vendors(self, q: str):
        vid = VENDOR_ID_RE.findall(q)[0].upper()
        if vid not in self.P.index:
            return self._insufficient(f"Vendor {vid} does not exist in the dataset.", [f"{vid} not found."])
        F = self.P[FACTOR_KEYS].to_numpy()
        target = self.P.loc[vid, FACTOR_KEYS].to_numpy(dtype=float)
        fsim = F @ target / (np.linalg.norm(F, axis=1) * np.linalg.norm(target) + 1e-9)
        V = self.system.index.vendor_index.vectors
        order = [self.system.index.vendor_ids.index(v) for v in self.P.index]
        csim = V[order] @ self.system.index.vendor_vector(vid)
        sims = pd.DataFrame({"factor_profile_similarity": fsim, "catalogue_similarity": csim}, index=self.P.index)
        sims["combined_similarity"] = sims.mean(axis=1)
        sims = sims.drop(index=vid).sort_values("combined_similarity", ascending=False).head(5)
        table = self._vendor_table(sims.index.tolist())
        for c in sims.columns:
            table[c] = sims[c].round(3).to_numpy()
        lines = [f"Vendors most similar to **{vid} {self.P.loc[vid, 'vendor_name']}**, ranked by the equal-weight mean "
                 "of factor-profile similarity (cosine over the four factor scores) and catalogue-text similarity:"]
        for t in table.itertuples(index=False):
            lines.append(f"- **{t.vendor_id} {t.vendor_name}** ({t.category}): factor profile "
                         f"{t.factor_profile_similarity:.2f}, catalogue {t.catalogue_similarity:.2f}")
        lines += ["", CAVEAT]
        return Evidence(table, ["Combined similarity = mean of factor-profile and catalogue similarity."], True,
                        [vid] + table["vendor_id"].tolist()), "\n".join(lines)

    def _handle_
