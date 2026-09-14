#!/usr/bin/env python3
"""Numeric encoding for the 43-tag taxonomy, so tags can join with the I/D/C/R
metric outputs in one CSV/JSON for plotting.

Two numbering systems (kept in sync with taxonomy.py automatically):
  * per-dimension codes  -- within each dimension, tags are 1..N in taxonomy
    order; 0 = none/unknown (dimension has no tag). e.g. Lighting & Weather:
    Daylight=1, Nighttime=2, Dusk / Dawn=3, ... Icy Road=10.
  * global tag id 1..43  -- stable flat index used for the multi-hot vector.

Because dimensions are MULTI-LABEL, per-dimension codes are stored as a LIST
(plus a single 'primary' = first tag in taxonomy order for quick categorical
plots), and a 43-wide multi_hot vector preserves everything.

Run directly to (re)write tag_codes.json:  python3 tag_codes.py
"""
import json
import os

from taxonomy import TAXONOMY

NONE_CODE = 0
DIM_ORDER = list(TAXONOMY)

# per-dimension: {dim: {tag: 1..N}}
DIM_CODES = {dim: {tag: i for i, tag in enumerate(tags, start=1)}
             for dim, tags in TAXONOMY.items()}

# global flat id 1..43 in taxonomy order (multi-hot column order)
ALL_TAGS_ORDERED = [t for tags in TAXONOMY.values() for t in tags]
GLOBAL_ID = {t: i for i, t in enumerate(ALL_TAGS_ORDERED, start=1)}

# short machine-friendly dimension keys (for CSV column names)
DIM_KEY = {
    "Operational Scenario": "op_scenario",
    "Roadway Context & Facility Type": "context",
    "Roadway Surface Type": "surface",
    "Pavement Marking Type & Configuration": "marking",
    "Observed Marking Visibility": "visibility",
    "Lighting & Weather": "lighting_weather",
}


def encode_labels(labels):
    """Flat list of tag names -> encoded dict:
       {per_dim_codes:{dim:[codes]}, per_dim_primary:{dim:code}, multi_hot:[43]}."""
    labset = set(labels or [])
    per_dim_codes, per_dim_primary = {}, {}
    for dim, tags in TAXONOMY.items():
        present = [t for t in tags if t in labset]          # taxonomy order
        per_dim_codes[dim] = [DIM_CODES[dim][t] for t in present]
        per_dim_primary[dim] = DIM_CODES[dim][present[0]] if present else NONE_CODE
    multi_hot = [1 if t in labset else 0 for t in ALL_TAGS_ORDERED]
    return {"per_dim_codes": per_dim_codes,
            "per_dim_primary": per_dim_primary,
            "multi_hot": multi_hot}


def as_dict():
    return {
        "none_code": NONE_CODE,
        "dimension_order": DIM_ORDER,
        "dimension_keys": DIM_KEY,
        "per_dimension_codes": DIM_CODES,     # {dim: {tag: code}}
        "global_tag_id": GLOBAL_ID,           # {tag: 1..43}  (multi_hot order)
        "multi_hot_order": ALL_TAGS_ORDERED,  # 43 tags in column order
    }


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "tag_codes.json")
    json.dump(as_dict(), open(out, "w"), indent=2)
    print(f"wrote {out}: {len(ALL_TAGS_ORDERED)} tags across {len(DIM_ORDER)} dimensions")
