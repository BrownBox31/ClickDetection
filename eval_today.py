import os
import glob
import joblib
import librosa
import numpy as np
from scipy.stats import kurtosis

SR = 16000
WINDOW = int(0.20 * SR)
DETECT_STEP = int(0.15 * SR)
EVENT_STEP = int(0.05 * SR)
EVENT_MERGE_SEC = 0.70
EVENT_PERCENTILE = 90
EVENT_MIN_SCORE = 40.0

model = joblib.load(os.path.join("sessions", "J3", "model", "click_detector.pkl"))


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


def window_score(window):
    rms = np.sqrt(np.mean(window ** 2))
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-8)
    k = kurtosis(window)
    return float(crest * max(k, 1.0))


def detect_events(audio):
    starts = list(range(0, len(audio) - WINDOW + 1, EVENT_STEP))
    if len(starts) < 3:
        return []
    scores = np.array(
        [window_score(audio[s : s + WINDOW]) for s in starts], dtype=np.float64
    )
    thr = max(float(np.percentile(scores, EVENT_PERCENTILE)), EVENT_MIN_SCORE)
    peaks = []
    for i in range(1, len(scores) - 1):
        if (
            scores[i] >= thr
            and scores[i] >= scores[i - 1]
            and scores[i] >= scores[i + 1]
        ):
            peaks.append((starts[i] / SR, scores[i]))
    merged = []
    for t, sc in peaks:
        if not merged or t - merged[-1][0] > EVENT_MERGE_SEC:
            merged.append([t, sc])
        elif sc > merged[-1][1]:
            merged[-1] = [t, sc]
    return [(t, sc) for t, sc in merged]


def region_start(audio):
    duration = len(audio) / SR
    events = detect_events(audio)
    if events:
        rs = max(0.0, events[-1][0] - 0.5)
        rule = "from_last_event"
    else:
        rs = max(0.0, duration * 0.6)
        rule = "fallback"
    rs = min(rs, max(0.0, duration - (WINDOW / SR)))
    return rs, events, rule


def predict_file(path, threshold):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < WINDOW:
        return "NO", 0.0, 0, "short"
    rs, events, rule = region_start(audio)
    starts = [
        s
        for s in range(0, len(audio) - WINDOW + 1, DETECT_STEP)
        if s / SR >= rs
    ]
    if not starts:
        all_s = list(range(0, len(audio) - WINDOW + 1, DETECT_STEP))
        starts = all_s[-max(1, min(8, len(all_s))) :]
    feats = np.asarray(
        [extract_features(audio[s : s + WINDOW]) for s in starts], dtype=np.float64
    )
    probs = model.predict_proba(feats)[:, 1]
    mx = float(np.max(probs))
    final = "YES" if mx >= threshold else "NO"
    return final, mx, len(starts), rule


def eval_set(files, true_label, threshold):
    tp = tn = fp = fn = 0
    rows = []
    for path in files:
        pred, mx, nwin, rule = predict_file(path, threshold)
        rows.append((os.path.basename(path), pred, mx))
        if true_label == "YES":
            if pred == "YES":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "NO":
                tn += 1
            else:
                fp += 1
    return tp, tn, fp, fn, rows


today = "20260929"
detect_dir = os.path.join("sessions", "J3", "detection_runs", "audio")
click_dir = os.path.join("sessions", "J3", "training_audio", "CLICK_FILES")
noclick_dir = os.path.join("sessions", "J3", "training_audio", "NO_CLICK_FILES")

detect_files = sorted(glob.glob(os.path.join(detect_dir, f"detect_{today}*.wav")))
run1, run2 = [], []
for path in detect_files:
    name = os.path.basename(path)
    parts = name.replace(".wav", "").split("_")
    hhmmss = int(parts[2]) if len(parts) >= 3 else 0
    (run2 if hhmmss >= 123630 else run1).append(path)

click_today = sorted(glob.glob(os.path.join(click_dir, f"*{today}*.wav")))
noclick_today = sorted(glob.glob(os.path.join(noclick_dir, f"*{today}*.wav")))

print("=" * 72)
print(
    f"Today eval | detect={len(detect_files)} "
    f"click={len(click_today)} noclick={len(noclick_today)}"
)
print(f"Run1={len(run1)} Run2={len(run2)} (Run2 GT = FITTED)")
print("=" * 72)

for thr in (0.45, 0.50, 0.55, 0.60):
    print(f"\n### Threshold {thr:.2f}")
    tp, tn, fp, fn, _ = eval_set(run2, "YES", thr)
    print(
        f"RUN2 (physically fitted, n={len(run2)}): "
        f"Fitted={tp} missed={fn} recall={tp / max(1, len(run2)):.1%}"
    )
    tp2, tn2, fp2, fn2, _ = eval_set(click_today, "YES", thr)
    _, tn3, fp3, _, _ = eval_set(noclick_today, "NO", thr)
    total = len(click_today) + len(noclick_today)
    correct = tp2 + tn3
    print(f"TRAINING TODAY labeled (n={total}):")
    print(
        f"  Fitted:     correct={tp2}/{len(click_today)} missed={fn2}"
    )
    print(
        f"  Not-fitted: correct={tn3}/{len(noclick_today)} false_fitted={fp3}"
    )
    print(
        f"  Accuracy={correct / max(1, total):.1%}  "
        f"Fitted_recall={tp2 / max(1, len(click_today)):.1%}  "
        f"NotFitted_spec={tn3 / max(1, len(noclick_today)):.1%}"
    )

thr = 0.50
print("\n" + "=" * 72)
print("DETAIL @ threshold 0.50")
print("=" * 72)

print("\n--- RUN 1 detections (no full GT) ---")
yes = no = 0
for path in run1:
    pred, mx, nwin, rule = predict_file(path, thr)
    yes += pred == "YES"
    no += pred == "NO"
    print(f"  {os.path.basename(path)}  {pred:3}  maxP={mx:.3f}")
print(f"Run1 summary @0.50: Fitted={yes} NotFitted={no}")

print("\n--- RUN 2 detections (GT=FITTED) ---")
misses = []
for path in run2:
    pred, mx, nwin, rule = predict_file(path, thr)
    mark = "OK" if pred == "YES" else "MISS"
    if pred != "YES":
        misses.append((os.path.basename(path), mx))
    print(f"  {os.path.basename(path)}  {pred:3}  maxP={mx:.3f}  {mark}")
print(f"Run2 misses @0.50: {len(misses)}/{len(run2)}")
for name, mx in misses:
    print(f"  MISS {name} maxP={mx:.3f}")
