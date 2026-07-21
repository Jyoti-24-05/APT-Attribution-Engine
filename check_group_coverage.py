"""
check_mapping_precision.py

Given a list of mapped technique IDs (from the description mapper) and a
target group name, reports how many of those techniques are actually part
of that group's REAL signature (true positives) vs. noise (false positives),
and compares the true-positive count against the empirical reliability
floor from find_min_coverage.py.

Edit MAPPED_IDS and TARGET_GROUP below, then run:
    python check_mapping_precision.py
"""

import pathlib
import numpy as np
import joblib

ROOT = pathlib.Path(".").resolve()
ARTIFACTS = ROOT / "artifacts"

X = joblib.load(ARTIFACTS / "X.joblib")
y = np.array(joblib.load(ARTIFACTS / "y.joblib"))
feature_vocab = joblib.load(ARTIFACTS / "feature_vocab.joblib")
all_tids = sorted(feature_vocab, key=feature_vocab.get)

# ── EDIT THESE ──
MAPPED_IDS = [
    "T1621", "T1555.006", "T1111", "T1098.001", "T1556.006", "T1021.008",
    "T1021.007", "T1550", "T1584", "T1136.003", "T1078.004", "T1195.003",
    "T1195.002", "T1668", "T1578.002", "T1557", "T1092", "T1651", "T1573",
    "T1587.001", "T1087.004", "T1204.005", "T1542.002", "T1671", "T1665","T1496.001 ","T1552.007 ","T1048 ","T1538 ","T1586.003 ","T1059.009 ","T1098.003 ","T1556.007 ","T1585.003 ","T1098.005 "
]
TARGET_GROUP_KEYWORDS = ["APT29", "Cozy Bear"]
# ────────────────

real_label = next((l for l in set(y) if any(k.lower() in l.lower() for k in TARGET_GROUP_KEYWORDS)), None)
if not real_label:
    raise SystemExit(f"No group found matching {TARGET_GROUP_KEYWORDS}")

idx = np.where(y == real_label)[0]
row_sums = X[idx].sum(axis=1)
full_row = X[idx[np.argmax(row_sums)]]
real_sig = set(all_tids[i] for i, v in enumerate(full_row) if v > 0)

# root-collapse the mapped IDs, same as encode_incident() does
mapped_roots = sorted(set(t.split(".")[0] if "." in t else t for t in MAPPED_IDS))

true_positives  = [t for t in mapped_roots if t in real_sig]
false_positives = [t for t in mapped_roots if t not in real_sig]

print(f"Target group: {real_label}  (real signature size: {len(real_sig)})")
print(f"Mapped (root-collapsed, deduped): {len(mapped_roots)} unique techniques")
print()
print(f"✓ True positives  ({len(true_positives)}): {true_positives}")
print(f"✗ Noise / false positives ({len(false_positives)}): {false_positives}")
print()
print(f"Precision: {len(true_positives)/len(mapped_roots)*100:.0f}%  "
      f"(share of mapped techniques that are actually real for this group)")
print(f"Coverage : {len(true_positives)/len(real_sig)*100:.0f}%  "
      f"(share of the group's real signature you actually captured)")
print()
print("Compare true-positive count against your find_min_coverage.py sweep result")
print("for this group's 'min reliable N' — if you're close to or below it, that")
print("explains a weak/incorrect prediction even with mostly-relevant mapped IDs.")