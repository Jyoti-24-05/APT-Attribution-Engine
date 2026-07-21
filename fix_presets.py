"""
fix_presets.py

Regenerates every entry in app.py's PRESETS dict using REAL technique
signatures pulled from X.joblib/y.joblib (what the model actually trained
on), instead of the hand-typed lists that caused the APT28 bug.

For each group:
  1. Find its real label in y.
  2. Reconstruct its fullest known signature from X.
  3. Verify predict_top3() actually ranks that group #1 using this signature
     (if not, print a warning instead of silently shipping a broken preset).
  4. Print ready-to-paste Python for the corrected PRESETS dict.

Run:
    python fix_presets.py
"""

import pathlib
import numpy as np
import joblib

ROOT = pathlib.Path(".").resolve()
ARTIFACTS = ROOT / "artifacts"

import sys
sys.path.insert(0, str(ROOT))
from inference import predict_top3

X = joblib.load(ARTIFACTS / "X.joblib")
y = np.array(joblib.load(ARTIFACTS / "y.joblib"))
feature_vocab = joblib.load(ARTIFACTS / "feature_vocab.joblib")
all_tids = sorted(feature_vocab, key=feature_vocab.get)
all_labels = set(y)

# ── the groups your current PRESETS dict covers, with search keywords to
#    find their real label in y (handles alias mismatches like "Fancy Bear"
#    being stored as "APT28") ─────────────────────────────────────────────
PRESET_TARGETS = {
    "APT28 — Fancy Bear (Russia)": ["APT28", "Fancy Bear", "Sofacy"],
    "Lazarus Group (North Korea)": ["Lazarus"],
    "APT29 — Cozy Bear (Russia)":  ["APT29", "Cozy Bear"],
    "APT41 — Winnti (China)":      ["APT41", "Winnti"],
    "MuddyWater (Iran)":           ["MuddyWater"],
}

# how much of the real signature to keep — must be >= the 0.75 floor the
# model was trained on (MIN_KEEP_RATIO in notebook 1). Using 1.0 (full
# signature) is safest for a demo preset since it guarantees max confidence.
KEEP_RATIO = 1.0


def find_real_label(keywords):
    for label in all_labels:
        if any(kw.lower() in label.lower() for kw in keywords):
            return label
    return None


def reconstruct_signature(label):
    idx = np.where(y == label)[0]
    if len(idx) == 0:
        return []
    row_sums = X[idx].sum(axis=1)
    fullest_row = X[idx[np.argmax(row_sums)]]
    return [all_tids[i] for i, v in enumerate(fullest_row) if v > 0]


print("=" * 70)
print("Regenerating presets from real training data")
print("=" * 70)

new_presets = {}
warnings = []

for preset_name, keywords in PRESET_TARGETS.items():
    real_label = find_real_label(keywords)
    if real_label is None:
        warnings.append(f"'{preset_name}': no matching label found in y for {keywords} — SKIPPED")
        print(f"\n✗ {preset_name}: no match found for {keywords}")
        continue

    full_sig = reconstruct_signature(real_label)
    if not full_sig:
        warnings.append(f"'{preset_name}': matched label '{real_label}' but got 0 techniques — SKIPPED")
        continue

    n_keep = max(1, int(len(full_sig) * KEEP_RATIO))
    sig_subset = sorted(full_sig)[:n_keep]  # deterministic; full sig if KEEP_RATIO=1.0

    # verify it actually predicts correctly before shipping it
    result = predict_top3(sig_subset)
    top1 = result[0]
    is_correct = top1["group"] == real_label

    status = "✓" if is_correct else "✗ WRONG PREDICTION"
    print(f"\n{status} {preset_name}")
    print(f"    real label in y     : '{real_label}'")
    print(f"    real technique count: {len(full_sig)}")
    print(f"    preset uses         : {len(sig_subset)} techniques ({KEEP_RATIO*100:.0f}% keep)")
    print(f"    predicted #1        : {top1['group']} ({top1['confidence']:.4f})")

    if not is_correct:
        warnings.append(
            f"'{preset_name}': predicted '{top1['group']}' instead of '{real_label}' "
            f"even with {KEEP_RATIO*100:.0f}% of the real signature — needs manual review, "
            f"not just a bigger preset."
        )

    new_presets[preset_name] = sig_subset

print()
print("=" * 70)
print("Warnings")
print("=" * 70)
if warnings:
    for w in warnings:
        print(f"⚠  {w}")
else:
    print("None — all presets verified correct.")

print()
print("=" * 70)
print("Paste this into app.py, replacing the old PRESETS dict")
print("=" * 70)
print()
print("PRESETS = {")
for name, tids in new_presets.items():
    print(f'    "{name}": {tids!r},')
print('    "Minimal incident — 2 techniques": ["T1059", "T1003"],  # kept as-is (generic stress test)')
print("}")