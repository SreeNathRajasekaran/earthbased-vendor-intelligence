"""Structural synthetic marketplace generator.

Each vendor receives four hidden traits (reliability, commercial pull,
catalogue-market fit, customer fit) drawn around one of five hidden archetypes.
All observable tables are *simulated* from those traits: customers place orders
over time, cancellations/returns/delivery times are drawn per order, ratings
depend on fit and delivery, and repeat purchases depend on satisfaction.
Vendor KPIs are then aggregated from the simulated orders.

The hidden traits are saved to ``data/_ground_truth/`` for validation only.
No downstream model or dashboard metric reads them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, GROUND_TRUTH_DIR_NAME, SEED, SLA_DAYS, WINDOW_START
from .data_dictionary import write_data_dictionary
from .feature_engineering import build_vendor_metrics, compute_vendor_kpis

logger = logging.getLogger(__name__)

# subcategory -> (search-demand index, items, materials, use cases)
CATALOGUE: dict[str, dict] = {
    "Sustainable Packaging": {
        "demand": 1.40, "n_vendors": 12, "price": (150, 1500), "bulk": True,
        "subcats": {
            "Food Containers": (1.6, ["clamshell food containers", "meal boxes", "soup bowls with lids", "compartment trays"],
                                ["sugarcane bagasse", "areca palm leaf", "kraft paperboard", "cornstarch PLA"],
                                ["restaurant takeaway", "cloud kitchens", "catering and bulk food service", "home meal delivery"]),
            "Compostable Cutlery": (1.2, ["spoon and fork sets", "wooden cutlery kits", "stirrers"],
                                    ["birch wood", "bamboo", "CPLA"], ["restaurants", "events and catering", "cafes"]),
            "Paper Bags & Wraps": (1.0, ["carry bags", "food wrapping paper", "bread bags"],
                                   ["recycled kraft paper", "greaseproof paper", "jute-blend paper"],
                                   ["grocery stores", "bakeries", "retail checkout"]),
            "Shipping & Mailers": (0.9, ["mailer boxes", "honeycomb wrap", "paper void fill"],
                                   ["corrugated recycled board", "honeycomb kraft", "recycled paper"],
                                   ["e-commerce shipping", "D2C brands", "gift packaging"]),
            "Cups & Lids": (1.3, ["hot beverage cups", "cold cups", "paper lids"],
                            ["PLA-lined paperboard", "bagasse", "aqueous-coated paper"],
                            ["cafes", "quick service restaurants", "office pantries"]),
        },
    },
    "Home & Kitchen": {
        "demand": 1.10, "n_vendors": 18, "price": (250, 2500), "bulk": True,
        "subcats": {
            "Kitchenware": (1.3, ["cooking spatula set", "chopping board", "serving bowls"],
                            ["bamboo", "neem wood", "coconut shell"], ["everyday home cooking", "gifting", "serving guests"]),
            "Cleaning Supplies": (1.4, ["dish scrub pads", "plant-based dishwash liquid", "reusable cleaning cloths"],
                                  ["loofah", "coconut coir", "plant surfactants"],
                                  ["daily kitchen cleaning", "homes and offices", "housekeeping teams"]),
            "Storage": (1.0, ["glass storage jars", "beeswax food wraps", "cotton produce bags"],
                        ["borosilicate glass", "beeswax-coated cotton", "organic cotton mesh"],
                        ["pantry organisation", "zero-waste grocery shopping", "meal prep"]),
            "Drinkware": (0.9, ["insulated steel bottles", "copper water bottles", "glass tumblers"],
                          ["food-grade stainless steel", "pure copper", "recycled glass"],
                          ["office and commute", "gym and travel", "daily hydration"]),
        },
    },
    "Personal Care": {
        "demand": 1.20, "n_vendors": 20, "price": (150, 900), "bulk": False,
        "subcats": {
            "Bath & Body": (1.3, ["handmade soap bars", "body scrubs", "bath salts"],
                            ["cold-processed oils", "coffee grounds", "Himalayan salt"], ["daily skincare", "sensitive skin", "gifting"]),
            "Hair Care": (1.2, ["shampoo bars", "conditioner bars", "herbal hair oil"],
                          ["reetha and shikakai", "coconut oil", "amla extract"],
                          ["plastic-free hair care", "travel", "dry and frizzy hair"]),
            "Oral Care": (1.1, ["bamboo toothbrushes", "tooth powder", "copper tongue cleaners"],
                          ["bamboo with castor bristles", "herbal clove powder", "pure copper"],
                          ["family oral hygiene", "travel kits", "plastic-free bathrooms"]),
            "Skincare": (1.0, ["face serums", "natural lip balms", "sunscreen sticks"],
                         ["cold-pressed botanical oils", "beeswax", "zinc oxide"],
                         ["daily skincare routine", "dry skin", "outdoor use"]),
        },
    },
    "Organic Food & Pantry": {
        "demand": 1.30, "n_vendors": 18, "price": (120, 900), "bulk": True,
        "subcats": {
            "Grains & Pulses": (1.3, ["millet flour", "unpolished dal", "heirloom rice"],
                                ["organically farmed millets", "pesticide-free pulses", "indigenous rice varieties"],
                                ["everyday home cooking", "restaurants and cloud kitchens", "bulk pantry stocking"]),
            "Spices": (1.2, ["turmeric powder", "whole spice mix", "cold-ground chilli"],
                       ["single-origin turmeric", "stone-ground spices", "sun-dried chillies"],
                       ["home kitchens", "restaurant bulk supply", "gifting hampers"]),
            "Snacks": (1.1, ["roasted makhana", "millet cookies", "trail mix"],
                       ["fox nuts", "jaggery-sweetened millets", "dry fruits"],
                       ["healthy snacking", "office pantries", "kids lunchboxes"]),
            "Beverages": (1.0, ["green tea", "herbal infusions", "cold-pressed juices"],
                          ["Himalayan tea leaves", "tulsi and ginger", "fresh seasonal fruit"],
                          ["daily wellness", "cafes", "corporate gifting"]),
        },
    },
    "Apparel & Accessories": {
        "demand": 0.95, "n_vendors": 16, "price": (400, 3500), "bulk": False,
        "subcats": {
            "Clothing": (1.1, ["t-shirts", "kurtas", "loungewear"], ["organic cotton", "khadi", "hemp blend"],
                         ["everyday wear", "summer wear", "workwear"]),
            "Bags": (1.2, ["tote bags", "laptop sleeves", "backpacks"], ["jute", "upcycled canvas", "cork fabric"],
                     ["daily commute", "grocery shopping", "corporate merchandise"]),
            "Footwear": (0.8, ["sandals", "sneakers", "slippers"], ["natural rubber", "recycled PET fabric", "cork"],
                         ["casual wear", "travel", "outdoor walking"]),
            "Jewellery": (0.7, ["earrings", "bracelets", "necklaces"], ["upcycled brass", "seed beads", "terracotta"],
                          ["festive wear", "gifting", "everyday styling"]),
        },
    },
    "Home Decor": {
        "demand": 0.80, "n_vendors": 18, "price": (300, 4000), "bulk": False,
        "subcats": {
            "Planters": (1.1, ["ceramic planters", "coir pots", "hanging planters"],
                         ["terracotta", "coconut coir", "macrame cotton"], ["balcony gardens", "indoor plants", "office desks"]),
            "Handicrafts": (0.9, ["wall hangings", "decorative trays", "table centrepieces"],
                            ["handwoven sabai grass", "reclaimed wood", "block-printed cotton"],
                            ["living room decor", "festive decoration", "gifting"]),
            "Lighting": (0.8, ["table lamps", "string lights", "candle holders"], ["bamboo", "upcycled glass", "soy wax"],
                         ["ambient lighting", "festive decor", "cafe interiors"]),
            "Textiles": (1.0, ["cushion covers", "table runners", "throws"], ["handloom cotton", "jute", "organic linen"],
                         ["home refresh", "hospitality and boutique hotels", "gifting"]),
        },
    },
    "Baby & Kids": {
        "demand": 1.15, "n_vendors": 8, "price": (250, 2000), "bulk": False,
        "subcats": {
            "Baby Care": (1.4, ["cloth diapers", "baby wipes", "baby massage oil"],
                          ["organic cotton", "bamboo fibre", "cold-pressed coconut oil"],
                          ["newborn care", "sensitive baby skin", "travel"]),
            "Toys": (1.1, ["wooden stacking toys", "puzzles", "activity boards"],
                     ["neem wood with natural dyes", "recycled cardboard", "non-toxic lacquered wood"],
                     ["early learning", "montessori play", "gifting"]),
            "Feeding": (1.0, ["bamboo feeding sets", "steel sippers", "silicone bibs"],
                        ["bamboo fibre", "food-grade steel", "food-grade silicone"], ["toddler meals", "daycare", "travel"]),
        },
    },
    "Stationery & Office": {
        "demand": 0.85, "n_vendors": 10, "price": (80, 800), "bulk": True,
        "subcats": {
            "Notebooks": (1.2, ["notebooks", "journals", "planners"], ["recycled paper", "cotton rag paper", "seed paper"],
                          ["students", "corporate gifting", "office use"]),
            "Writing": (1.0, ["seed pencils", "recycled paper pens", "refillable fountain pens"],
                        ["plantable seed paper", "recycled newspaper", "brass"], ["schools", "corporate events", "daily writing"]),
            "Office Supplies": (1.0, ["desk organisers", "file folders", "sticky notes"],
                                ["bamboo", "recycled board", "recycled paper"],
                                ["office desks", "bulk corporate procurement", "home office"]),
        },
    },
}

CATEGORY_FEATURES = {
    "Sustainable Packaging": ["Leak-resistant and suitable for hot and oily food.", "Microwave-safe and freezer-safe.",
                              "Home-compostable and plastic-free.", "Sturdy stackable design for storage."],
    "Home & Kitchen": ["Free from plastic and toxic coatings.", "Durable for daily use and easy to clean.",
                       "Plastic-free alternative to conventional products."],
    "Personal Care": ["Free from sulphates, parabens and synthetic fragrance.", "Plastic-free, compostable packaging.",
                      "Small-batch handmade formulation."],
    "Organic Food & Pantry": ["Sourced directly from farmer collectives.", "No preservatives or artificial additives.",
                              "Lab-tested for pesticide residue."],
    "Apparel & Accessories": ["Naturally dyed and skin-friendly.", "Fair-trade artisan made.", "Breathable and long-lasting fabric."],
    "Home Decor": ["Handmade by artisan clusters.", "Each piece is unique.", "Low-impact natural finishes."],
    "Baby & Kids": ["Non-toxic and safe for infants.", "Gentle on sensitive skin.", "Free from BPA and harsh chemicals."],
    "Stationery & Office": ["Made with recycled fibre.", "Acid-free pages.", "Plastic-free and plantable options."],
}

CATEGORY_CERTS = {
    "Sustainable Packaging": ["Compostable (IS/ISO 17088)", "FSC", "Plastic-free", "FSSAI food-contact"],
    "Home & Kitchen": ["Plastic-free", "FSC", "EcoMark"],
    "Personal Care": ["Cruelty-free", "Plastic-free", "AYUSH licensed"],
    "Organic Food & Pantry": ["FSSAI", "India Organic", "NPOP"],
    "Apparel & Accessories": ["GOTS", "Fair Trade", "Handloom Mark"],
    "Home Decor": ["Handloom Mark", "Fair Trade", "EcoMark"],
    "Baby & Kids": ["BIS certified", "Cruelty-free", "GOTS"],
    "Stationery & Office": ["FSC", "EcoMark", "Plastic-free"],
}

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Pune", "Jaipur", "Ahmedabad", "Chennai", "Hyderabad", "Kolkata", "Dehradun"]
CITY_WEIGHTS = [0.16, 0.15, 0.15, 0.10, 0.08, 0.08, 0.08, 0.08, 0.07, 0.05]

# Hidden archetypes: mean of (reliability, commercial, catalogue fit, customer fit), onboarding window (days from 2022-04-01)
ARCHETYPES = {
    "established_performer": {"p": 0.22, "mu": (0.9, 1.0, 0.5, 0.5), "onboard": (0, 600)},
    "broad_catalogue_low_traction": {"p": 0.22, "mu": (0.0, -0.8, 0.9, -0.3), "onboard": (0, 821)},
    "niche_loyal": {"p": 0.20, "mu": (0.4, -0.3, -0.6, 1.1), "onboard": (0, 821)},
    "operational_risk_moderate_demand": {"p": 0.18, "mu": (-1.2, 0.4, 0.0, -0.4), "onboard": (0, 821)},
    "early_stage": {"p": 0.18, "mu": (-0.4, -0.6, -0.7, -0.6), "onboard": (550, 821)},
}

PREFIXES = ["Green", "Terra", "Leaf", "Earth", "Bamboo", "Kora", "Prakriti", "Sattva", "Neem", "Vana", "Bhoomi", "Aranya",
            "Mitti", "Tula", "Surya", "Ritu", "Nila", "Sage", "Moss", "Fern", "Jute", "Khadi", "Sal", "Kesar"]
MIDS = ["", "leaf", "root", "craft", "nest", "way", "field", "grove"]
SUFFIXES = {
    "Sustainable Packaging": ["Packaging", "Pack Co.", "Wraps"],
    "Home & Kitchen": ["Home", "Kitchenworks", "Living"],
    "Personal Care": ["Naturals", "Botanicals", "Care"],
    "Organic Food & Pantry": ["Organics", "Farms", "Pantry"],
    "Apparel & Accessories": ["Threads", "Studio", "Weaves"],
    "Home Decor": ["Crafts", "Decor", "Artisans"],
    "Baby & Kids": ["Little Ones", "Kids", "Baby"],
    "Stationery & Office": ["Paperworks", "Stationery", "Supplies"],
}


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _vendor_name(rng: np.random.Generator, category: str, used: set[str]) -> str:
    for _ in range(500):
        name = f"{rng.choice(PREFIXES)}{rng.choice(MIDS)} {rng.choice(SUFFIXES[category])}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("Could not generate a unique vendor name.")


def _generate_vendors(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.Timestamp("2022-04-01")
    arch_names = list(ARCHETYPES)
    arch_p = np.array([ARCHETYPES[a]["p"] for a in arch_names], dtype=float)
    arch_p /= arch_p.sum()
    used: set[str] = set()
    rows, truth = [], []
    idx = 1
    for category, spec in CATALOGUE.items():
        for _ in range(spec["n_vendors"]):
            arch = arch_names[rng.choice(len(arch_names), p=arch_p)]
            a = ARCHETYPES[arch]
            R, C, F, U = np.array(a["mu"]) + rng.normal(0, 0.5, 4)
            lo, hi = a["onboard"]
            onboard = base + pd.Timedelta(days=int(rng.integers(lo, hi)))
            vid = f"V{idx:03d}"
            pool = CATEGORY_CERTS[category]
            certs = rng.choice(pool, size=int(rng.integers(1, min(3, len(pool)) + 1)), replace=False)
            bulk_p = _sigmoid(0.4 + 0.4 * C + 0.3 * R) if spec["bulk"] else 0.12
            rows.append({
                "vendor_id": vid,
                "vendor_name": _vendor_name(rng, category, used),
                "category": category,
                "city": CITIES[rng.choice(len(CITIES), p=CITY_WEIGHTS)],
                "onboarding_date": onboard.date().isoformat(),
                "price_index": round(float(np.clip(np.exp(0.08 * U + rng.normal(0, 0.11)), 0.75, 1.45)), 3),
                "stock_availability": round(float(np.clip(0.86 + 0.06 * R + rng.normal(0, 0.025), 0.55, 0.995)), 3),
                "response_time_hours": round(float(np.clip(np.exp(2.4 - 0.45 * R + rng.normal(0, 0.25)), 1.0, 96.0)), 1),
                "recommendation_acceptance_rate": round(float(np.clip(_sigmoid(-2.6 + 0.45 * U + 0.3 * F + rng.normal(0, 0.15)), 0.01, 0.6)), 4),
                "bulk_order_capable": bool(rng.random() < bulk_p),
                "certifications": "; ".join(sorted(str(c) for c in certs)),
            })
            truth.append({
                "vendor_id": vid, "archetype": arch,
                "true_reliability": R, "true_commercial": C, "true_catalogue_fit": F, "true_customer_fit": U,
                # simulation parameters (internal only)
                "p_cancel": float(_sigmoid(-3.1 - 0.75 * R + rng.normal(0, 0.2))),
                "p_return": float(_sigmoid(-2.9 - 0.45 * R - 0.45 * U + rng.normal(0, 0.2))),
                "delivery_mean": float(max(1.5, 3.6 - 0.7 * R + rng.normal(0, 0.35))),
                "delivery_sd": float(np.exp(-0.3 - 0.35 * R + rng.normal(0, 0.15))),
                "p_repeat": float(_sigmoid(-0.3 + 0.9 * U + 0.45 * R)),
                "conv_rate": float(_sigmoid(-3.3 + 0.45 * C + 0.3 * U + rng.normal(0, 0.15))),
                "atc_rate": float(_sigmoid(-2.2 + 0.5 * U + 0.1 * C + rng.normal(0, 0.15))),
                "order_weight": float(np.exp(0.95 * C + 0.25 * F + 0.2 * U)),
            })
            idx += 1
    return pd.DataFrame(rows), pd.DataFrame(truth)


def _describe(rng, item, material, use, feature, category, bulk, certs, premium) -> str:
    parts = [f"{item[0].upper() + item[1:]} made from {material}.", f"Suitable for {use}.", feature]
    if bulk and CATALOGUE[category]["bulk"]:
        parts.append(str(rng.choice(["Available in bulk cartons for business and restaurant orders.",
                                     "Bulk pricing available for B2B and wholesale orders."])))
    if premium:
        parts.append(str(rng.choice(["Premium finish with gift-ready packaging.", "Premium-grade materials and finishing."])))
    parts.append(f"Certifications: {certs}.")
    return " ".join(parts)


def _generate_products(rng, vendors: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = truth.set_index("vendor_id")
    rows, hidden = [], []
    pid = 1
    for v in vendors.itertuples(index=False):
        F, C, U = t.at[v.vendor_id, "true_catalogue_fit"], t.at[v.vendor_id, "true_commercial"], t.at[v.vendor_id, "true_customer_fit"]
        spec = CATALOGUE[v.category]
        subs = list(spec["subcats"])
        demand = np.array([spec["subcats"][s][0] for s in subs])
        n_p = int(np.clip(rng.poisson(np.exp(2.05 + 0.3 * F + 0.1 * C)), 3, 25))
        k_sub = int(np.clip(round(2.2 + 0.9 * F + rng.normal(0, 0.6)), 1, len(subs)))
        k_sub = min(k_sub, n_p)
        pref = demand ** (1.0 + 1.2 * F)
        pref = pref / pref.sum()
        chosen = rng.choice(len(subs), size=k_sub, replace=False, p=pref)
        cp = pref[chosen] / pref[chosen].sum()
        lo, hi = spec["price"]
        premium = v.price_index >= 1.12
        for j in range(n_p):
            s_idx = int(chosen[j]) if j < k_sub else int(rng.choice(chosen, p=cp))
            sub = subs[s_idx]
            d, items, materials, uses = spec["subcats"][sub]
            item, material, use = str(rng.choice(items)), str(rng.choice(materials)), str(rng.choice(uses))
            name = f"{material[0].upper() + material[1:]} {item.title()}"
            if v.category == "Sustainable Packaging":
                name += f" (Pack of {int(rng.choice([25, 50, 100]))})"
            price = float(round(np.exp(rng.uniform(np.log(lo), np.log(hi))) * v.price_index, 0))
            quality = float(rng.normal(0, 0.5) + 0.3 * U)
            appeal = float(np.exp(0.35 * U + 0.3 * quality + 0.2 * F + rng.normal(0, 0.35)) * d)
            rows.append({
                "product_id": f"P{pid:04d}", "vendor_id": v.vendor_id, "product_name": name,
                "category": v.category, "subcategory": sub, "price": price,
                "stock_availability": round(float(np.clip(v.stock_availability + rng.normal(0, 0.04), 0.3, 1.0)), 3),
                "subcategory_demand_index": d,
                "search_relevance_score": round(float(np.clip(_sigmoid(0.3 + 0.9 * F + rng.normal(0, 0.5)), 0.02, 0.99)), 3),
                "description": _describe(rng, item, material, use, str(rng.choice(CATEGORY_FEATURES[v.category])),
                                         v.category, v.bulk_order_capable, v.certifications, premium),
            })
            hidden.append({"product_id": f"P{pid:04d}", "quality": quality, "appeal": appeal})
            pid += 1
    return pd.DataFrame(rows), pd.DataFrame(hidden)


def _generate_customers(rng, n: int) -> pd.DataFrame:
    cats = list(CATALOGUE)
    cat_p = np.array([CATALOGUE[c]["demand"] for c in cats])
    cat_p = cat_p / cat_p.sum()
    engagement = rng.normal(0, 1, n)
    mean_extra = 3.0 * np.exp(0.3 * engagement)
    n_orders = 1 + np.minimum(rng.negative_binomial(1.5, 1.5 / (1.5 + mean_extra)), 24)
    return pd.DataFrame({
        "customer_id": [f"C{i + 1:04d}" for i in range(n)],
        "location": [CITIES[i] for i in rng.choice(len(CITIES), size=n, p=CITY_WEIGHTS)],
        "category_preference": [cats[i] for i in rng.choice(len(cats), size=n, p=cat_p)],
        "premium_affinity": rng.normal(0, 1, n),
        "engagement_latent": engagement,
        "n_orders": n_orders.astype(int),
    })


def _simulate_orders(rng, vendors, truth, products, hidden, customers) -> pd.DataFrame:
    start = pd.Timestamp(WINDOW_START)
    t = truth.set_index("vendor_id").loc[vendors["vendor_id"]]
    v_ids = vendors["vendor_id"].to_numpy()
    v_pos = {v: i for i, v in enumerate(v_ids)}
    onboard_day = (pd.to_datetime(vendors["onboarding_date"]) - start).dt.days.to_numpy()
    p_cancel, p_return = t["p_cancel"].to_numpy(), t["p_return"].to_numpy()
    del_mean, del_sd = t["delivery_mean"].to_numpy(), t["delivery_sd"].to_numpy()
    p_repeat, weight = t["p_repeat"].to_numpy(), t["order_weight"].to_numpy()
    U = t["true_customer_fit"].to_numpy()
    bulk = vendors["bulk_order_capable"].to_numpy(dtype=bool)
    price_index = vendors["price_index"].to_numpy(dtype=float)

    prod_ids = products["product_id"].to_numpy()
    pv = products["vendor_id"].map(v_pos).to_numpy()
    price = products["price"].to_numpy(dtype=float)
    appeal = hidden["appeal"].to_numpy()
    quality = hidden["quality"].to_numpy()
    prod_cat = products["category"].to_numpy()
    cats = list(CATALOGUE)
    cat_p = np.array([CATALOGUE[c]["demand"] for c in cats])
    cat_p = cat_p / cat_p.sum()
    cat_index = {c: np.flatnonzero(prod_cat == c) for c in cats}

    records = []
    for cust in customers.itertuples(index=False):
        days = np.sort(rng.integers(0, 366, size=int(cust.n_orders)))
        last: dict[str, tuple[int, bool]] = {}
        for d in days:
            cat = cust.category_preference if rng.random() < 0.65 else cats[rng.choice(len(cats), p=cat_p)]
            pidx = cat_index[cat]
            active = onboard_day[pv[pidx]] <= d
            if not active.any():
                continue
            chosen = None
            prev = last.get(cat)
            if prev is not None and prev[1] and rng.random() < p_repeat[prev[0]]:
                mask = active & (pv[pidx] == prev[0])
                if mask.any():
                    w = appeal[pidx] * mask
                    chosen = pidx[rng.choice(len(pidx), p=w / w.sum())]
            if chosen is None:
                w = weight[pv[pidx]] * appeal[pidx] * np.exp(3.0 * cust.premium_affinity * (price_index[pv[pidx]] - 1.0)) * active
                chosen = pidx[rng.choice(len(pidx), p=w / w.sum())]
            vi = pv[chosen]
            if bulk[vi] and CATALOGUE[cat]["bulk"] and rng.random() < 0.3:
                qty = int(rng.integers(5, 31))
            else:
                qty = int(rng.choice([1, 2, 3], p=[0.7, 0.22, 0.08]))
            value = round(price[chosen] * qty * (0.93 if qty >= 10 else 1.0), 2)
            cancelled = bool(rng.random() < p_cancel[vi])
            if cancelled:
                delivery, returned, rating, satisfied = np.nan, False, np.nan, False
            else:
                delivery = float(max(1, int(round(rng.normal(del_mean[vi], del_sd[vi])))))
                returned = bool(rng.random() < p_return[vi])
                late = delivery > SLA_DAYS
                rating = np.nan
                if rng.random() < 0.7:
                    r = 3.85 + 0.45 * U[vi] + 0.3 * quality[chosen] - 0.4 * late - 1.4 * returned + rng.normal(0, 0.55)
                    rating = float(np.clip(np.round(r), 1, 5))
                satisfied = (not returned) and (np.isnan(rating) or rating >= 4) and delivery <= SLA_DAYS + 1
            records.append((cust.customer_id, v_ids[vi], prod_ids[chosen], int(d), qty, value, delivery, returned, cancelled, rating))
            last[cat] = (vi, satisfied)

    orders = pd.DataFrame(records, columns=["customer_id", "vendor_id", "product_id", "day", "quantity", "order_value",
                                            "delivery_days", "returned", "cancelled", "rating"])
    orders["order_date"] = start + pd.to_timedelta(orders["day"], unit="D")
    orders = orders.sort_values(["order_date", "customer_id"]).reset_index(drop=True)
    orders["order_id"] = [f"O{i + 1:05d}" for i in range(len(orders))]
    return orders[["order_id", "customer_id", "vendor_id", "product_id", "order_date", "quantity", "order_value",
                   "delivery_days", "returned", "cancelled", "rating"]]


def _finalise_products(rng, products, truth, vendors, orders) -> pd.DataFrame:
    agg = orders.groupby("product_id").agg(orders=("order_id", "size"), returns=("returned", "sum"),
                                           rating=("rating", "mean"), review_count=("rating", "count"))
    p = products.merge(agg, on="product_id", how="left")
    p[["orders", "returns", "review_count"]] = p[["orders", "returns", "review_count"]].fillna(0).astype(int)
    t = truth.set_index("vendor_id")
    conv = p["vendor_id"].map(t["conv_rate"]).to_numpy() * np.exp(rng.normal(0, 0.25, len(p)))
    F = p["vendor_id"].map(t["true_catalogue_fit"]).to_numpy()
    base_traffic = rng.poisson(25 * np.exp(0.5 * F))
    views = np.maximum(1, np.round(p["orders"].to_numpy() / conv + base_traffic)).astype(int)
    atc_rate = np.clip(p["vendor_id"].map(t["atc_rate"]).to_numpy(), 0.01, 0.9)
    atc = np.maximum(rng.binomial(views, atc_rate), p["orders"].to_numpy())
    p["views"] = views
    p["add_to_cart"] = atc
    p["conversion_rate"] = (p["orders"] / p["views"]).round(4)
    p["rating"] = p["rating"].round(2)
    cols = ["product_id", "vendor_id", "product_name", "category", "subcategory", "price", "rating", "review_count",
            "orders", "returns", "views", "add_to_cart", "conversion_rate", "stock_availability",
            "subcategory_demand_index", "search_relevance_score", "description"]
    return p[cols]


def _finalise_customers(rng, base, orders, vendors) -> pd.DataFrame:
    o = orders.merge(vendors[["vendor_id", "price_index"]], on="vendor_id", how="left").sort_values("order_date")
    o["prior_vendor"] = o.groupby(["customer_id", "vendor_id"]).cumcount() > 0
    agg = o.groupby("customer_id").agg(orders_count=("order_id", "size"), repeat_rate=("prior_vendor", "mean"),
                                       mean_price_index=("price_index", "mean"))
    aov = o[~o["cancelled"]].groupby("customer_id")["order_value"].mean().rename("avg_order_value")
    c = base.merge(agg, on="customer_id", how="left").merge(aov, on="customer_id", how="left")
    c["orders_count"] = c["orders_count"].fillna(0).astype(int)
    c["repeat_rate"] = c["repeat_rate"].fillna(0).round(3)
    c["avg_order_value"] = c["avg_order_value"].fillna(0).round(2)
    c["engagement_score"] = np.clip(np.round(35 + 9 * np.log1p(c["orders_count"]) + 9 * c["engagement_latent"]
                                             + rng.normal(0, 5, len(c))), 0, 100).astype(int)
    seg = pd.qcut(c["mean_price_index"].rank(method="first"), 3, labels=["Value", "Core", "Premium"])
    c["customer_segment"] = seg.astype(str).where(c["orders_count"] > 0, "Core")
    return c[["customer_id", "location", "orders_count", "avg_order_value", "repeat_rate", "category_preference",
              "engagement_score", "customer_segment"]]


def _catalogue_descriptions(vendors: pd.DataFrame, products: pd.DataFrame) -> pd.Series:
    v = vendors.set_index("vendor_id")
    out = {}
    for vid, grp in products.groupby("vendor_id"):
        row = v.loc[vid]
        subs = grp["subcategory"].value_counts().index.tolist()
        names = grp["product_name"].drop_duplicates().head(4).tolist()
        text = (f"{row.vendor_name} is a {row.city}-based {row.category.lower()} vendor with {len(grp)} listed products "
                f"across {', '.join(s.lower() for s in subs)}. Representative products: {'; '.join(names)}.")
        if row.bulk_order_capable:
            text += " Supports bulk and business (B2B) orders."
        if row.price_index >= 1.12:
            text += " Premium price positioning."
        text += f" Certifications: {row.certifications}."
        out[vid] = text
    return pd.Series(out, name="catalogue_description")


def generate_marketplace(seed: int = SEED, n_customers: int = 3000, out_dir: Path | None = None,
                         save: bool = True) -> dict[str, pd.DataFrame]:
    """Generate all synthetic tables; optionally write them to ``out_dir``."""
    out_dir = Path(out_dir or DATA_DIR)
    rng = np.random.default_rng(seed)
    vendors, truth = _generate_vendors(rng)
    products, hidden = _generate_products(rng, vendors, truth)
    base_customers = _generate_customers(rng, n_customers)
    orders = _simulate_orders(rng, vendors, truth, products, hidden, base_customers)
    products = _finalise_products(rng, products, truth, vendors, orders)
    customers = _finalise_customers(rng, base_customers, orders, vendors)

    kpis = compute_vendor_kpis(vendors, products, orders)
    vendors = vendors.merge(kpis, on="vendor_id", how="left")
    vendors["catalogue_description"] = vendors["vendor_id"].map(_catalogue_descriptions(vendors, products))
    rate_cols = ["fulfilment_rate", "cancellation_rate", "return_rate", "repeat_customer_rate", "conversion_rate",
                 "customer_engagement", "category_coverage"]
    vendors[rate_cols] = vendors[rate_cols].round(4)
    vendors[["avg_delivery_days", "average_rating"]] = vendors[["avg_delivery_days", "average_rating"]].round(2)
    vendors[["gmv", "average_order_value"]] = vendors[["gmv", "average_order_value"]].round(2)
    vendor_cols = ["vendor_id", "vendor_name", "category", "city", "onboarding_date", "product_count", "average_rating",
                   "price_index", "stock_availability", "avg_delivery_days", "fulfilment_rate", "cancellation_rate",
                   "return_rate", "response_time_hours", "order_count", "gmv", "average_order_value",
                   "repeat_customer_rate", "conversion_rate", "customer_engagement", "category_coverage",
                   "recommendation_acceptance_rate", "bulk_order_capable", "certifications", "catalogue_description"]
    vendors = vendors[vendor_cols]
    metrics = build_vendor_metrics(vendors, products, orders, customers)

    tables = {"vendors": vendors, "products": products, "orders": orders, "customers": customers,
              "vendor_metrics": metrics.round(5)}
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(out_dir / f"{name}.csv", index=False)
        gt_dir = out_dir / GROUND_TRUTH_DIR_NAME
        gt_dir.mkdir(exist_ok=True)
        truth[["vendor_id", "archetype", "true_reliability", "true_commercial", "true_catalogue_fit",
               "true_customer_fit"]].round(4).to_csv(gt_dir / "vendor_latent_truth.csv", index=False)
        (gt_dir / "README.md").write_text(
            "Hidden generator traits. Used ONLY to validate factor recovery and clustering. "
            "Never read by the dashboard, recommendation engine or copilot.\n", encoding="utf-8")
        write_data_dictionary(out_dir)
        logger.info("Wrote %s vendors, %s products, %s orders, %s customers to %s",
                    len(vendors), len(products), len(orders), len(customers), out_dir)
    return tables
