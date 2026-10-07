import os
import glob
import joblib
import librosa
import numpy as np
from datetime import datetime
from scipy.stats import kurtosis

SR = 16000
WINDOW = int(0.2 * SR)
DETECT_STEP = int(0.15 * SR)
MERGE = 0.55
THR = 0.50
TODAY = datetime.now().strftime("%Y%m%d")
ROOT = os.path.join("sessions", "J1_J2")
MODEL = joblib.load(os.path.join(ROOT, "model", "click_detector.pkl"))


def feats(a):
    mfcc = librosa.feature.mfcc(y=a, sr=SR, n_mfcc=13, n_fft=512, hop_length=256)
    rms = np.sqrt(np.mean(a ** 2))
    peak = np.max(np.abs(a))
    crest = peak / (rms + 1e-8)
    k = kurtosis(a)
    return np.concatenate([np.mean(mfcc, 1), np.std(mfcc, 1), [rms, crest, k]])


def dual(audio):
    starts = list(range(0, len(audio) - WINDOW + 1, DETECT_STEP))
    if not starts:
        return 0, [], 0.0, 0.0, 0.0, np.array([])
    X = np.asarray([feats(audio[s : s + WINDOW]) for s in starts], dtype=np.float64)
    probs = MODEL.predict_proba(X)[:, 1]
    pos = []
    for i, s in enumerate(starts):
        if probs[i] >= THR:
            w = audio[s : s + WINDOW]
            pos.append(
                (
                    s / SR,
                    float(probs[i]),
                    float(np.sqrt(np.mean(w ** 2))),
                    float(np.max(np.abs(w))),
                )
            )
    merged = []
    for t, p, r, pk in pos:
        if not merged or t - merged[-1][0] > MERGE:
            merged.append([t, p, r, pk])
        elif p > merged[-1][1]:
            merged[-1] = [t, p, r, pk]
    return (
        len(merged),
        merged,
        float(np.max(probs)),
        float(np.sqrt(np.mean(audio ** 2))),
        float(np.max(np.abs(audio))),
        probs,
        starts,
    )


def stats(subset, key):
    vals = [r[key] for r in subset]
    return float(np.mean(vals)), float(np.median(vals)), float(np.min(vals)), float(np.max(vals))


sets = [
    ("TWO", "BOTH", os.path.join(ROOT, "training_audio", "TWO_CLICK_FILES")),
    ("ONE", "ONE", os.path.join(ROOT, "training_audio", "ONE_CLICK_FILES")),
    ("NONE", "NONE", os.path.join(ROOT, "training_audio", "NO_CLICK_FILES")),
]

rows = []
for label, expected, folder in sets:
    for path in sorted(glob.glob(os.path.join(folder, f"*{TODAY}*.wav"))):
        audio, _ = librosa.load(path, sr=SR, mono=True)
        n, ev, mx, rms, peak, probs, starts = dual(audio)
        pred = "BOTH" if n >= 2 else ("ONE" if n == 1 else "NONE")
        rows.append(
            {
                "label": label,
                "expected": expected,
                "pred": pred,
                "ok": pred == expected,
                "n": n,
                "mx": mx,
                "rms": rms,
                "peak": peak,
                "dur": len(audio) / SR,
                "file": os.path.basename(path),
                "path": path,
                "ev": ev,
                "probs": probs,
                "starts": starts,
            }
        )

print("LEVEL COMPARISON (file RMS / peak)")
print(f"{'group':<28} n  rms_mean  rms_med  peak_mean  peak_med  maxProb_mean")
for label, _, _ in sets:
    for kind, cond in [("OK", True), ("MISS", False)]:
        sub = [r for r in rows if r["label"] == label and r["ok"] == cond]
        if not sub:
            continue
        rm, rd, _, _ = stats(sub, "rms")
        pm, pd, _, _ = stats(sub, "peak")
        mm, _, _, _ = stats(sub, "mx")
        print(f"{label + ' ' + kind:<28} {len(sub):>2}  {rm:.4f}   {rd:.4f}   {pm:.3f}     {pd:.3f}     {mm:.3f}")

print("\nOVERALL OK vs MISS levels")
for kind, cond in [("OK", True), ("MISS", False)]:
    sub = [r for r in rows if r["ok"] == cond]
    rm, rd, rlo, rhi = stats(sub, "rms")
    pm, pd, plo, phi = stats(sub, "peak")
    print(
        f"{kind}: n={len(sub)} RMS mean={rm:.4f} med={rd:.4f} "
        f"range=[{rlo:.4f},{rhi:.4f}]  peak mean={pm:.3f} med={pd:.3f}"
    )

print("\nTWO_CLICK predicted ONE — 2nd-best peak away from 1st event")
print(f"{'file':<40} rms    peak   maxP  event1        2nd_best")
for r in rows:
    if r["label"] != "TWO" or r["pred"] != "ONE":
        continue
    t0 = r["ev"][0][0] if r["ev"] else -999
    candidates = []
    for i, s in enumerate(r["starts"]):
        t = s / SR
        if abs(t - t0) >= MERGE:
            candidates.append((float(r["probs"][i]), t))
    candidates.sort(reverse=True)
    second = candidates[0] if candidates else (0.0, 0.0)
    e1 = f"{r['ev'][0][0]:.2f}/{r['ev'][0][1]:.2f}" if r["ev"] else "-"
    print(
        f"{r['file']:<40} {r['rms']:.4f} {r['peak']:.3f}  {r['mx']:.2f}  "
        f"{e1:<12} 2nd={second[1]:.2f}s p={second[0]:.2f}"
    )

print("\nONE_CLICK predicted BOTH — 2nd event details")
print(f"{'file':<40} rms    peak   events")
for r in rows:
    if r["label"] != "ONE" or r["pred"] != "BOTH":
        continue
    evs = ", ".join(f"{t:.2f}s p={p:.2f} rms={rm:.3f}" for t, p, rm, pk in r["ev"])
    print(f"{r['file']:<40} {r['rms']:.4f} {r['peak']:.3f}  {evs}")

print("\nONE_CLICK predicted NONE")
for r in rows:
    if r["label"] != "ONE" or r["pred"] != "NONE":
        continue
    print(
        f"{r['file']:<40} rms={r['rms']:.4f} peak={r['peak']:.3f} "
        f"maxP={r['mx']:.2f} dur={r['dur']:.2f}s"
    )

print("\nMiss rate by peak loudness tercile")
peaks = np.array([r["peak"] for r in rows])
qs = np.quantile(peaks, [0.33, 0.66])
bins = [
    ("quiet", peaks <= qs[0]),
    ("mid", (peaks > qs[0]) & (peaks <= qs[1])),
    ("loud", peaks > qs[1]),
]
for name, mask in bins:
    sub = [r for r, m in zip(rows, mask) if m]
    miss = sum(1 for r in sub if not r["ok"])
    print(
        f"  {name}: n={len(sub)} miss={miss}/{len(sub)} "
        f"({100 * miss / len(sub):.0f}%)  peak cuts={qs[0]:.3f}/{qs[1]:.3f}"
    )

# How many TWO misses have a near-threshold 2nd peak?
near = []
for r in rows:
    if r["label"] != "TWO" or r["pred"] != "ONE":
        continue
    t0 = r["ev"][0][0] if r["ev"] else -999
    best = 0.0
    for i, s in enumerate(r["starts"]):
        t = s / SR
        if abs(t - t0) >= MERGE:
            best = max(best, float(r["probs"][i]))
    near.append(best)
if near:
    near = np.array(near)
    print("\nTWO->ONE: 2nd-best probability away from 1st click")
    print(f"  mean={near.mean():.2f} med={np.median(near):.2f} "
          f"in [0.30,0.50)={np.sum((near>=0.30)&(near<0.50))}/{len(near)} "
          f">=0.50 should have been 2nd={np.sum(near>=0.50)}/{len(near)}")
