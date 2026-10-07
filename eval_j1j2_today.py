import os
import glob
import joblib
import librosa
import numpy as np
from collections import Counter
from datetime import datetime
from scipy.stats import kurtosis

SR = 16000
WINDOW = int(0.20 * SR)
DETECT_STEP = int(0.15 * SR)
DUAL_MIN_EVENT_GAP_SEC = 0.45
DUAL_CLUSTER_SEC = 0.20
DUAL_MAX_EVENTS = 2
THRESHOLD = 0.60
ROOT = os.path.join("sessions", "J1_J2")
MODEL = joblib.load(os.path.join(ROOT, "model", "click_detector.pkl"))


def extract_features(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=13, n_fft=512, hop_length=256
    )
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    crest = peak / (rms + 1e-8)
    k = kurtosis(audio)
    return np.concatenate(
        [np.mean(mfcc, axis=1), np.std(mfcc, axis=1), [rms, crest, k]]
    )


def detect_dual(audio, threshold=THRESHOLD):
    starts = list(range(0, len(audio) - WINDOW + 1, DETECT_STEP))
    if not starts:
        return 0, "NO", "NO", []
    feats = np.asarray(
        [extract_features(audio[s : s + WINDOW]) for s in starts],
        dtype=np.float64,
    )
    probabilities = MODEL.predict_proba(feats)[:, 1]
    n = len(probabilities)

    peak_idxs = []
    for i in range(n):
        p = float(probabilities[i])
        if p < threshold:
            continue
        left = float(probabilities[i - 1]) if i > 0 else -1.0
        right = float(probabilities[i + 1]) if i + 1 < n else -1.0
        if p + 1e-9 >= left and p + 1e-9 >= right:
            peak_idxs.append(i)

    clustered = []
    for idx in peak_idxs:
        t = starts[idx] / SR
        p = float(probabilities[idx])
        if not clustered:
            clustered.append((t, p))
            continue
        prev_t, prev_p = clustered[-1]
        if t - prev_t <= DUAL_CLUSTER_SEC:
            if p > prev_p:
                clustered[-1] = (t, p)
        else:
            clustered.append((t, p))

    ranked = sorted(clustered, key=lambda item: item[1], reverse=True)
    chosen = []
    for t, p in ranked:
        if any(abs(t - ct) < DUAL_MIN_EVENT_GAP_SEC for ct, _cp in chosen):
            continue
        chosen.append((t, p))
        if len(chosen) >= DUAL_MAX_EVENTS:
            break
    events = sorted(chosen, key=lambda item: item[0])

    count = len(events)
    j1 = "YES" if count >= 1 else "NO"
    j2 = "YES" if count >= 2 else "NO"
    return count, j1, j2, events


def scenario(j1, j2):
    if j1 == "YES" and j2 == "YES":
        return "BOTH"
    if j1 == "YES":
        return "ONE"
    return "NONE"


sets = [
    ("TWO_CLICK", os.path.join(ROOT, "training_audio", "TWO_CLICK_FILES"), "BOTH"),
    ("ONE_CLICK", os.path.join(ROOT, "training_audio", "ONE_CLICK_FILES"), "ONE"),
    ("NO_CLICK", os.path.join(ROOT, "training_audio", "NO_CLICK_FILES"), "NONE"),
]

print(
    f"J1_J2 training eval | threshold={THRESHOLD} | "
    f"min_gap={DUAL_MIN_EVENT_GAP_SEC} | max_events={DUAL_MAX_EVENTS} | "
    f"rule=top2_strongest+gap"
)
print("=" * 100)

summary = {}
all_rows = []

for name, folder, expected in sets:
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    ok = 0
    print(f"\n### {name}  expected={expected}  n={len(files)}")
    print(
        f"{'file':<48} {'clicks':>6} {'J1':>4} {'J2':>4} "
        f"{'pred':>6} {'ok':>4}  events"
    )
    for path in files:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        n, j1, j2, events = detect_dual(audio)
        pred = scenario(j1, j2)
        match = pred == expected
        ok += int(match)
        ev = ", ".join(f"{t:.2f}s/{p:.2f}" for t, p in events[:4])
        mark = "OK" if match else "MISS"
        print(
            f"{os.path.basename(path):<48} {n:>6} {j1:>4} {j2:>4} "
            f"{pred:>6} {mark:>4}  {ev}"
        )
        all_rows.append((name, expected, pred, match, n))
    summary[name] = (ok, len(files))

print("\n" + "=" * 100)
print("SUMMARY vs folder labels")
total_ok = total_n = 0
for name, (ok, n) in summary.items():
    total_ok += ok
    total_n += n
    pct = (100.0 * ok / n) if n else 0.0
    print(f"  {name:<10} {ok}/{n} correct ({pct:.1f}%)")
pct = (100.0 * total_ok / total_n) if total_n else 0.0
print(f"  TOTAL      {total_ok}/{total_n} correct ({pct:.1f}%)")

print("\nPredicted distribution by true folder:")
for true in ("BOTH", "ONE", "NONE"):
    preds = Counter(r[2] for r in all_rows if r[1] == true)
    print(f"  true {true}: {dict(preds)}")
