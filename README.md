# Wire Fit Detection

A machine learning-based audio detection system that detects whether a wire is fitted by identifying a target click sound in noisy audio.

## Approach

Microphone → Audio Recording → MFCC Features → Random Forest Model → YES/NO

## Features

- Live microphone recording
- START and STOP controls
- Microphone name display
- MFCC feature extraction
- RMS, Crest Factor and Kurtosis features
- Random Forest-based click detection
- Window-by-window detection
- Probability output
- Final YES/NO result

## Project Structure

Practice/
├── training_audio/
│   ├── CLICK_FILES/
│   ├── NO_CLICK_FILES/
│   └── COMBINE_FILES/
├── model/
│   └── click_detector.pkl
├── train_model.py
├── live_detect.py
├── app.py
├── requirements.txt
└── README.md

## Model

The trained Random Forest model is saved as:

model/click_detector.pkl

## Detection Process

1. Click START.
2. Record live audio.
3. Click STOP.
4. Audio is divided into 0.2-second windows with a 0.1-second step.
5. MFCC, RMS, Crest Factor and Kurtosis features are extracted from each window.
6. The Random Forest model predicts the click probability.
7. Results are displayed for every audio window.
8. If any window detects the click, the final result is YES.
9. Otherwise, the final result is NO.

## Run Application

streamlit run app.py

## Final Output

YES - Wire is fitted

or

NO - Wire is not fitted