import glob
import librosa
import numpy as np
import soundfile as sf
import os

SR = 16000
INPUT_FOLDER = "training_audio/COMBINE_FILES"
OUTPUT_FILE = "training_audio/COMBINE_FILES/all_test_combined.wav"

files = sorted(
    glob.glob(os.path.join(INPUT_FOLDER, "*.wav"))
)

audio_parts = []

for file in files:
    if os.path.basename(file) == "all_test_combined.wav":
        continue

    audio, _ = librosa.load(file, sr=SR, mono=True)
    audio_parts.append(audio)

if not audio_parts:
    raise ValueError("No WAV files found")

combined_audio = np.concatenate(audio_parts)

sf.write(OUTPUT_FILE, combined_audio, SR)

print("Combined file saved:", OUTPUT_FILE)
print("Number of files combined:", len(audio_parts))
print("Duration:", len(combined_audio) / SR, "seconds")