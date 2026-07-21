

import json
import pathlib
import numpy as np
import joblib

ROOT = pathlib.Path(".").resolve()
ARTIFACTS = ROOT / "artifacts"

# ── the numbers you're trying to reconcile ──────────────────────────────
NB2_PROBE_COUNT = 15
NB2_PROBE_SAMPLE_IDS = {
    "T1003", "T1005", "T1007", "T1021", "T1036",
    "T1049", "T1057", "T1059", "T1114", "T1119",
}

APP_PRESET = {
    "T1059", "T1078", "T1566", "T1053", "T1027",
    "T1105", "T1036", "T1016", "T1057", "T1083",
}

print("=" * 70)
print("1. group_tech_counts.json  (notebook 1 EDA output)")
print("=" * 70)
counts = json.load(open(ARTIFACTS / "group_tech_counts.json"))
apt28_matches = {k: v for k, v in counts.items()
                 if "APT28" in k or "Fancy" in k or "Sofacy" in k}
print(f"Matching keys: {apt28_matches}")
if not apt28_matches:
    print("⚠  NO MATCH FOUND. Check exact group naming — print all keys:")
    print(sorted(counts.keys())[:30], "...")

print()
print("=" * 70)
print("2. Reconstructing APT28's signature from X.joblib / y.joblib")
print("   (this is literally what the model was trained on)")
print("=" * 70)
X = joblib.load(ARTIFACTS / "X.joblib")
y = np.array(joblib.load(ARTIFACTS / "y.joblib"))
feature_vocab = joblib.load(ARTIFACTS / "feature_vocab.joblib")
all_tids = sorted(feature_vocab, key=feature_vocab.get)

candidates = [n for n in set(y) if "APT28" in n or "Fancy" in n or "Sofacy" in n]
print(f"Label candidates in y: {candidates}")

if candidates:
    apt28_name = candidates[0]
    idx = np.where(y == apt28_name)[0]
    row_sums = X[idx].sum(axis=1)

    print(f"Number of training rows for '{apt28_name}': {len(idx)}")
    print(f"Technique-count per row (min/median/max): "
          f"{row_sums.min():.0f} / {np.median(row_sums):.0f} / {row_sums.max():.0f}")

    full_row = X[idx[np.argmax(row_sums)]]
    full_tids = set(all_tids[i] for i, v in enumerate(full_row) if v > 0)
    print(f"\nFullest reconstructed signature has {len(full_tids)} techniques:")
    print(sorted(full_tids))
else:
    full_tids = set()
    print("⚠  No matching label found in y at all — group may have been")
    print("   dropped during training (e.g. filtered for having <3 techniques,")
    print("   or a naming mismatch between notebook 1 and notebook 2).")

print()
print("=" * 70)
print("3. Compare against your notebook 2 probe output")
print("=" * 70)
print(f"NB2 probe said APT28 has {NB2_PROBE_COUNT} techniques.")
print(f"X/y reconstruction says: {len(full_tids)} techniques.")
if len(full_tids) == NB2_PROBE_COUNT:
    print("✓ Counts MATCH — 15 is confirmed as APT28's real trained signature size.")
elif full_tids:
    print(f"✗ MISMATCH — {len(full_tids)} (from X/y) vs {NB2_PROBE_COUNT} (from nb2 probe).")
    print("  Possible causes: different `y` label spelling/alias picked up a")
    print("  different group, or nb2's probe used a different filter/source than")
    print("  the X/y artifacts loaded here.")
    

overlap_probe = full_tids & NB2_PROBE_SAMPLE_IDS
print(f"\nOverlap between reconstructed signature and nb2's 10 sample IDs: "
      f"{len(overlap_probe)}/10 → {sorted(overlap_probe)}")
missing_from_recon = NB2_PROBE_SAMPLE_IDS - full_tids
if missing_from_recon:
    print(f"⚠  These nb2 sample IDs are NOT in the X/y reconstruction: {sorted(missing_from_recon)}")
    print("   → strong signal the two are looking at different data/groups.")

print()
print("=" * 70)
print("4. Compare against app.py's hardcoded PRESETS['APT28 — Fancy Bear (Russia)']")
print("=" * 70)
overlap_preset = full_tids & APP_PRESET
coverage = len(overlap_preset) / len(full_tids) * 100 if full_tids else 0
print(f"App preset techniques   : {sorted(APP_PRESET)}")
print(f"Overlap with real sig   : {len(overlap_preset)}/{len(APP_PRESET)} → {sorted(overlap_preset)}")
print(f"Preset covers {coverage:.0f}% of APT28's real (reconstructed) signature "
      f"({len(full_tids)} techniques total).")

not_in_real_sig = APP_PRESET - full_tids
if not_in_real_sig:
    print(f"⚠  Preset contains IDs NOT in APT28's real signature at all: {sorted(not_in_real_sig)}")
    print("   → these were likely hand-typed/guessed, not pulled from real STIX data.")

print()
print("=" * 70)
print("5. Verdict")
print("=" * 70)
if full_tids and len(full_tids) < 20:
    ratio = len(APP_PRESET & full_tids) / len(full_tids)
    print(f"APT28's REAL signature is small ({len(full_tids)} techniques), so the")
    print(f"10-item preset actually covers ~{ratio*100:.0f}% of it — NOT wildly sparse.")
    print("If the preset still contains IDs absent from the real signature (see")
    print("section 4), that's the more likely bug: the preset was hand-typed from")
    print("memory/research, not generated from this project's own STIX data —")
    print("so it's testing a signature that doesn't fully match what the model learned.")
elif full_tids:
    print(f"APT28's real signature has {len(full_tids)} techniques — the 10-item")
    print("preset is a small subset of that, consistent with the earlier")
    print("sparse-input hypothesis.")