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

        def _handle_factor_contrast(self, q: str):
        conds = parse_factor_conditions(q)
        cat = detect_category(q, self.categories)
        base = pd.Series(True, index=self.P.index)
        if cat:
            base &= self.P["category"] == cat

        def apply(hi: int, lo: int) -> pd.Series:
            mask = base.copy()
            for k, d in conds.items():
                mask &= (self.P[f"{k}_pct"] >= hi) if d == "high" else (self.P[f"{k}_pct"] <= lo)
            return mask

        hi, lo, relaxed = 60, 40, False
        mask = apply(hi, lo)
        if not mask.any():
            hi, lo, relaxed = 55, 45, True
            mask = apply(hi, lo)
        cond_text = " and ".join(f"{d} {FACTOR_LABELS[k]}" for k, d in conds.items())
        notes = [f"Thresholds: high >= {hi}th percentile, low <= {lo}th percentile.",
                 f"{int(mask.sum())} vendors meet the conditions."]
        if relaxed:
            notes.append("Thresholds were relaxed because no vendor met the 60/40 thresholds.")
        if not mask.any():
            return self._insufficient(f"No vendor shows {cond_text}, even with relaxed thresholds.", notes)
        sel = self.P[mask]
        score = sum((sel[f"{k}_pct"] if d == "high" else 100 - sel[f"{k}_pct"]) for k, d in conds.items())
        ids = score.sort_values(ascending=False).index[:8].tolist()
        table = self._vendor_table(ids)
        lines = [f"{int(mask.sum())} vendor(s) show {cond_text} (high = at or above the {ordinal(hi)} percentile, "
                 f"low = at or below the {ordinal(lo)}){' within ' + cat if cat else ''}. Closest matches:"]
        lines += [self._line(v, list(conds)) for v in ids[:5]]
        lines += ["", "Mixed profiles like these are candidates for targeted support rather than exclusion, for "
                      "example fulfilment help for vendors with good catalogue fit but operational-risk signals.", CAVEAT]
        return Evidence(table, notes, True, ids), "\n".join(lines)

    def _handle_category_gaps(self, q: str):
        t = self.system.category_table.copy()
        t["median_reliability_pct"] = t["median_reliability_pct"].round().astype(int)
        table = t[["category", "orders", "vendors", "order_share", "vendor_share", "demand_to_supply_ratio",
                   "orders_per_vendor", "median_reliability_pct"]].round(3)
        gaps = table[table["demand_to_supply_ratio"] >= 1.15]
        sub = self.system.subcategory_table.head(3)
        sub_notes = [f"Subcategory {r.subcategory} ({r.category}): {int(r.orders)} orders across "
                     f"{int(r.vendors_covering)} covering vendors ({r.orders_per_covering_vendor:.1f} orders per vendor)."
                     for r in sub.itertuples(index=False)]
        notes = sub_notes + ["Gap threshold: demand-to-supply ratio of 1.15 or more."]
        lines = ["Demand-to-supply ratio is a category's share of orders divided by its share of vendors; values above "
                 "1 mean demand is concentrated on relatively few vendors."]
        if gaps.empty:
            lines.append("No category exceeds the 1.15 threshold, so vendor coverage is broadly proportional to demand.")
        for r in gaps.itertuples(index=False):
            lines.append(f"- **{r.category}**: {r.order_share:.1%} of orders vs {r.vendor_share:.1%} of vendors "
                         f"(ratio {r.demand_to_supply_ratio:.2f}; {r.orders_per_vendor:.1f} orders per vendor; median "
                         f"reliability {ordinal(r.median_reliability_pct)} pct)")
        lines.append("Most concentrated subcategories:")
        lines += [f"- {n}" for n in sub_notes]
        lines += ["", CAVEAT]
        return Evidence(table, notes, True, []), "\n".join(lines)

    def _handle_premium_segment(self, q: str):
        cat = detect_category(q, self.categories)
        P = self.P if not cat else self.P[self.P["category"] == cat]
        prem_pct = (self.P["premium_customer_share"].rank(pct=True) * 100).reindex(P.index)
        mask = (prem_pct >= 60) & (P["customer_product_fit_pct"] >= 50)
        rule = "premium-customer share at or above the 60th percentile and customer fit at or above the 50th percentile"
        if not mask.any():
            mask = prem_pct >= 60
            rule = "premium-customer share at or above the 60th percentile"
        if not mask.any():
            return self._insufficient("No vendor shows a concentration of Premium-segment customers.")
        sel = P[mask].sort_values("premium_customer_share", ascending=False).head(8)
        table = self._vendor_table(sel.index.tolist(), {"premium_customer_share": "premium_customer_share",
                                                        "price_index": "price_index"})
        notes = ["Premium segment = top third of customers by the average price index of vendors they buy from.",
                 f"Selection rule: {rule}.", f"{int(mask.sum())} vendors selected."]
        lines = [f"Vendors whose orders skew toward Premium-segment customers{' in ' + cat if cat else ''} ({rule}):"]
        for r in table.head(5).itertuples(index=False):
            lines.append(f"- **{r.vendor_id} {r.vendor_name}** ({r.category}): {r.premium_customer_share:.1%} of orders "
                         f"from Premium customers, price index {r.price_index:.2f}, Customer fit "
                         f"{ordinal(getattr(r, '_10'))} pct" if False else
                         f"- **{r.vendor_id} {r.vendor_name}** ({r.category}): {r.premium_customer_share:.1%} of orders "
                         f"from Premium customers, price index {r.price_index:.2f}, Customer/Product Fit "
                         f"{ordinal(self.P.at[r.vendor_id, 'customer_product_fit_pct'])} pct, Operational Reliability "
                         f"{ordinal(self.P.at[r.vendor_id, 'operational_reliability_pct'])} pct")
        lines += ["", CAVEAT]
        return Evidence(table, notes, True, table["vendor_id"].tolist()), "\n".join(lines)

    def _handle_prioritisation(self, q: str):
        cat = detect_category(q, self.categories)
        res, spec = prioritise_vendors(self.system.profile, category=cat,
                                       min_percentiles={"operational_reliability": 30}, top_n=5)
        if res.empty:
            return self._insufficient("No vendors pass the reliability floor for this request.")
        table = res[["vendor_id", "vendor_name", "category", "segment", "priority_index"]
                    + [f"{k}_pct" for k in FACTOR_KEYS]].copy()
        table["priority_index"] = table["priority_index"].round(1)
        notes = ["Priority index = equal-weight mean of the four factor percentiles.",
                 f"Filters: {', '.join(spec['filters'])}.", f"{spec['n_candidates']} vendors passed the filters."]
        lines = [f"Vendors ranked by an equal-weight mean of factor percentiles{' within ' + cat if cat else ''}, "
                 "after excluding vendors below the 30th reliability percentile:"]
        for r, rationale in zip(table.itertuples(index=False), res["rationale"]):
            lines.append(f"- **{r.vendor_id} {r.vendor_name}** ({r.category}): priority index {r.priority_index:.1f}. {rationale}")
        lines += ["", "Adjust the weights on the Vendor Intelligence or Matching pages if some factors matter more "
                      "for this decision.", CAVEAT]
        return Evidence(table, notes, True, table["vendor_id"].tolist()), "\n".join(lines)

    def _handle_semantic_search(self, q: str):
        res = semantic_match(self.system, q, top_k=5)
        thr = float(self.system.index.encoder.weak_threshold)
        top = float(res["semantic_similarity"].max()) if not res.empty else 0.0
        if res.empty or top < thr:
            return self._insufficient(
                f"The closest catalogue matches are weak (top similarity {top:.2f}, below the {thr:.2f} threshold), "
                "so I can't identify vendors that supply this.", [f"Top similarity {top:.2f}; weak-match threshold {thr:.2f}."])
        res = res[res["semantic_similarity"] >= thr]
        table = res[["vendor_id", "vendor_name", "category", "top_product", "semantic_similarity",
                     "catalogue_similarity"] + [f"{k}_pct" for k in FACTOR_KEYS]].round(3)
        lines = ["Closest catalogue matches, ranked by semantic similarity of product listings to the request:"]
        for r in table.itertuples(index=False):
            lines.append(f"- **{r.vendor_id} {r.vendor_name}** ({r.category}): '{r.top_product}' (similarity "
                         f"{r.semantic_similarity:.2f}); Reliability {ordinal(r.operational_reliability_pct)} pct, "
                         f"Catalogue fit {ordinal(r.catalogue_market_fit_pct)} pct, Commercial "
                         f"{ordinal(r.commercial_potential_pct)} pct")
        lines += ["", CAVEAT]
        return Evidence(table, [f"Weak-match threshold {thr:.2f}."], True, table["vendor_id"].tolist()), "\n".join(lines)

    def _handle_out_of_scope(self, q: str):
        return self._insufficient(
            "This question asks for something the marketplace dataset does not contain. The repository uses "
            "synthetic, anonymised vendor, product, order and customer records; it holds no company financials, and "
            "no score can establish that a vendor will succeed. I can compare vendors, explain factor scores, "
            "analyse category coverage or match products instead.")
