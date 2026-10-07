import time
import joblib
import librosa
import numpy as np
from scipy.stats import kurtosis

SR = 16000
WINDOW = int(0.20 * SR)
STEP = int(0.10 * SR)
DETECT_THRESHOLD = 0.8

MODEL_PATH = "model/click_detector.pkl"
AUDIO_FILE = "training_audio/COMBINE_FILES/all_test_combined.wav"

model = joblib.load(MODEL_PATH)

audio, _ = librosa.load(AUDIO_FILE, sr=SR, mono=True)

def extract_features(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=13,
        n_fft=512,
        hop_length=256
    )

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    crest = peak / (rms + 1e-8)
    k = kurtosis(audio)

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [rms, crest, k]
    ])

for start in range(0, len(audio) - WINDOW + 1, STEP):
    window = audio[start:start + WINDOW]

    start_time = time.perf_counter()

    features = extract_features(window).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]

    processing_time = time.perf_counter() - start_time

    result = "YES" if probability >= DETECT_THRESHOLD else "NO"

    print(
        f"Audio: {start / SR:.2f}s | "
        f"Result: {result} | "
        f"Probability: {probability:.3f} | "
        f"Processing: {processing_time:.3f}s"
    )