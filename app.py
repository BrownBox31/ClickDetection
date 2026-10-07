import io
import os
import wave
from datetime import datetime

import av
import streamlit as st
import joblib
import librosa
import numpy as np
import sounddevice as sd
from scipy.stats import kurtosis
from streamlit_mic_recorder import mic_recorder

SR = 16000
WINDOW = int(0.20 * SR)
STEP = int(0.10 * SR)
DETECT_THRESHOLD_DEFAULT = 0.8
MODEL_PATH = "model/click_detector.pkl"

TRAINING_FOLDERS = {
    "Click": "training_audio/CLICK_FILES",
    "No click": "training_audio/NO_CLICK_FILES",
}
CAPTURE_MIN = 30
CAPTURE_MAX = 40
CAPTURE_PREFIX = {
    "Click": "click_",
    "No click": "noclick_",
}

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

def get_microphone():
    try:
        index = sd.default.device[0]

        if index is None or index < 0:
            return "Not Connected", False

        device = sd.query_devices(index)

        if device["max_input_channels"] > 0:
            return device["name"], True

    except Exception:
        pass

    return "Not Connected", False

def extract_audio(audio_bytes):
    container = av.open(io.BytesIO(audio_bytes))
    parts = []
    sample_rate = SR

    for frame in container.decode(audio=0):
        data = frame.to_ndarray()

        if data.ndim > 1:
            data = np.mean(data, axis=0)

        parts.append(data.astype(np.float32))
        sample_rate = frame.sample_rate

    if not parts:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(parts)

    if sample_rate != SR:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SR
        )

    return audio.astype(np.float32)

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

def detect_audio(audio, threshold):
    results = []

    for start in range(0, len(audio) - WINDOW + 1, STEP):
        window = audio[start:start + WINDOW]

        start_time = start / SR
        processing_start = __import__("time").perf_counter()

        features = extract_features(window).reshape(1, -1)
        probability = model.predict_proba(features)[0][1]

        processing_time = __import__("time").perf_counter() - processing_start

        result = "YES" if probability >= threshold else "NO"

        results.append(
            f"Audio: {start_time:.2f}s | "
            f"Result: {result} | "
            f"Probability: {probability:.3f} | "
            f"Processing: {processing_time:.3f}s"
        )

    return results

def captured_count(label):
    folder = TRAINING_FOLDERS[label]
    prefix = CAPTURE_PREFIX[label]

    if not os.path.isdir(folder):
        return 0

    return len([
        name for name in os.listdir(folder)
        if name.lower().endswith(".wav") and name.lower().startswith(prefix)
    ])

def capture_status(label, count):
    if count >= CAPTURE_MAX:
        return f"{label}: {count} / {CAPTURE_MAX} (full)"

    if count >= CAPTURE_MIN:
        return f"{label}: {count} / {CAPTURE_MAX} (ready)"

    return f"{label}: {count} / {CAPTURE_MIN}–{CAPTURE_MAX}"

def wav_duration(path):
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / handle.getframerate()

def save_training_recording(audio, label):
    folder = TRAINING_FOLDERS[label]
    os.makedirs(folder, exist_ok=True)

    prefix = "click" if label == "Click" else "noclick"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{prefix}_{stamp}.wav")
    suffix = 2

    while os.path.exists(path):
        path = os.path.join(folder, f"{prefix}_{stamp}_{suffix}.wav")
        suffix += 1

    audio_bytes = audio["bytes"]

    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("Recording is not a WAV file")

    with open(path, "wb") as handle:
        handle.write(audio_bytes)

    return path, wav_duration(path)

st.set_page_config(
    page_title="Wire Fit Detection",
    page_icon="🔊"
)

st.title("Wire Fit Detection")

mode = st.radio(
    "Mode",
    ["Detection", "Training"],
    horizontal=True
)

mic_name, connected = get_microphone()

if connected:
    st.success("Microphone: " + mic_name + " | Connected")
else:
    st.error("Microphone: Not Connected")

if mode == "Training":
    label = st.radio(
        "Save recording as",
        ["Click", "No click"],
        horizontal=True
    )

    click_saved = captured_count("Click")
    noclick_saved = captured_count("No click")
    selected_saved = click_saved if label == "Click" else noclick_saved
    st.write(capture_status("Click", click_saved))
    st.write(capture_status("No click", noclick_saved))
    st.caption(
        "Record 30 to 40 files for each label. "
        "Choose Click or No click, then press START and STOP. "
        "Each recording is saved as its own WAV file."
    )

    if "saved_audio_ids" not in st.session_state:
        st.session_state.saved_audio_ids = set()

    if "saved_files" not in st.session_state:
        st.session_state.saved_files = []

    if selected_saved >= CAPTURE_MAX:
        st.warning(
            f"{label} already has {CAPTURE_MAX} recordings. "
            "Switch the label to keep recording the other set."
        )
        audio = None
    else:
        audio = mic_recorder(
            start_prompt="START",
            stop_prompt="STOP",
            just_once=False,
            use_container_width=True,
            format="wav",
            key="training_recorder"
        )

    if audio is not None and audio["id"] not in st.session_state.saved_audio_ids:
        st.session_state.saved_audio_ids.add(audio["id"])

        try:
            path, duration = save_training_recording(audio, label)
        except Exception as error:
            st.error(f"Could not save recording: {error}")
        else:
            st.session_state.saved_files.append(path)
            st.success(
                f"Saved {os.path.basename(path)} "
                f"({duration:.1f}s) as {label}"
            )
            st.audio(audio["bytes"], format="audio/wav")

    if st.session_state.saved_files:
        st.subheader("Saved this session")

        for path in reversed(st.session_state.saved_files):
            st.write(path)

    st.stop()

detect_threshold = st.slider(
    "Detection threshold",
    min_value=0.50,
    max_value=0.99,
    value=DETECT_THRESHOLD_DEFAULT,
    step=0.05,
    help="A window is YES when probability is at least this value."
)

audio = mic_recorder(
    start_prompt="START",
    stop_prompt="STOP",
    just_once=True,
    use_container_width=True,
    key="detection_recorder"
)

if model is None:
    st.error("Model file not found. Train a model before using Detection.")
    st.stop()

if audio is not None:
    audio_array = extract_audio(audio["bytes"])

    if len(audio_array) >= WINDOW:
        results = detect_audio(audio_array, detect_threshold)

        st.subheader("Detection Results")
        st.caption(f"Threshold: {detect_threshold:.2f}")

        for result in results:
            st.write(result)

        probabilities = [
            float(result.split("Probability: ")[1].split(" |")[0])
            for result in results
        ]

        if any(p >= detect_threshold for p in probabilities):
            st.success("YES - Wire is fitted")
        else:
            st.error("NO - Wire is not fitted")