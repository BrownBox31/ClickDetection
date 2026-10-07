import argparse
import os
import sys
import glob
import joblib
import librosa
import numpy as np
from datetime import datetime
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SR = 16000
WINDOW = int(0.20 * SR)
STEP = int(0.02 * SR)
EVENT_STEP = int(0.05 * SR)
EVENT_MERGE_SEC = 0.70
EVENT_PERCENTILE = 90
EVENT_MIN_SCORE = 40.0

if getattr(sys, "frozen", False):
    _APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_ROOT = os.path.join(_APP_DIR, "sessions")
DEFAULT_SESSION = "J3"


def session_paths(session_id):
    root = os.path.join(SESSIONS_ROOT, session_id)
    return {
        "root": root,
        "click_dir": os.path.join(root, "training_audio", "CLICK_FILES"),
        "non_click_dir": os.path.join(root, "training_audio", "NO_CLICK_FILES"),
        "one_click_dir": os.path.join(root, "training_audio", "ONE_CLICK_FILES"),
        "two_click_dir": os.path.join(root, "training_audio", "TWO_CLICK_FILES"),
        "model_path": os.path.join(root, "model", "click_detector.pkl"),
    }


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


def window_score(window):
    rms = np.sqrt(np.mean(window ** 2))
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-8)
    k = kurtosis(window)
    return float(crest * max(k, 1.0))


def get_windows(audio, step=STEP):
    starts = []
    windows = []

    for start in range(0, len(audio) - WINDOW + 1, step):
        starts.append(start)
        windows.append(audio[start:start + WINDOW])

    return starts, windows


def detect_events(audio):
    starts, windows = get_windows(audio, step=EVENT_STEP)

    if not windows:
        return []

    scores = np.array([window_score(window) for window in windows], dtype=np.float64)
    threshold = max(float(np.percentile(scores, EVENT_PERCENTILE)), EVENT_MIN_SCORE)

    peaks = []

    for index in range(1, len(scores) - 1):
        score = scores[index]

        if (
            score >= threshold
            and score >= scores[index - 1]
            and score >= scores[index + 1]
        ):
            peaks.append((starts[index] / SR, score))

    merged = []

    for time_sec, score in peaks:
        if not merged or time_sec - merged[-1][0] > EVENT_MERGE_SEC:
            merged.append([time_sec, score])
        elif score > merged[-1][1]:
            merged[-1] = [time_sec, score]

    return [(time_sec, score) for time_sec, score in merged]


def region_from_last_event(audio):
    """
    Anchor on the last screw-like event only.
    Do not assume exactly 4 screws; unscrews earlier are ignored.
    """
    duration = len(audio) / SR
    events = detect_events(audio)

    if len(events) >= 1:
        last_event_time = events[-1][0]
        region_start = max(0.0, last_event_time - 0.5)
        rule = "from_last_event"
    else:
        region_start = max(0.0, duration * 0.6)
        rule = "fallback_last_40_percent"

    region_start = min(region_start, max(0.0, duration - (WINDOW / SR)))
    return region_start, events, rule


def select_click_windows(audio):
    starts, windows = get_windows(audio, step=STEP)
    region_start, events, rule = region_from_last_event(audio)

    selected = []
    scores = []

    for start, window in zip(starts, windows):
        if start / SR >= region_start:
            selected.append(window)
            scores.append(window_score(window))

    if not selected:
        selected = windows[-max(1, min(10, len(windows))):]
        scores = [window_score(window) for window in selected]
        rule = rule + "+short_tail"

    scores = np.array(scores, dtype=np.float64)
    keep = max(1, int(len(selected) * 0.35))
    chosen = np.argsort(scores)[-keep:]
    chosen_windows = [selected[index] for index in chosen]

    return chosen_windows, region_start, events, rule


def dual_peak_candidates(audio):
    """
    J1_J2 only: find local energy peaks on the full clip (no screw region).
    """
    starts, windows = get_windows(audio, step=EVENT_STEP)
    if len(windows) < 3:
        return [], starts, windows, np.array([])

    scores = np.array([window_score(window) for window in windows], dtype=np.float64)
    threshold = float(np.percentile(scores, 80))

    peaks = []
    for index in range(1, len(scores) - 1):
        score = scores[index]
        if (
            score >= threshold
            and score >= scores[index - 1]
            and score >= scores[index + 1]
        ):
            peaks.append([starts[index], starts[index] / SR, score])

    # Collapse nearby peaks; keep strongest in each short cluster.
    clustered = []
    for start, time_sec, score in peaks:
        if not clustered:
            clustered.append([start, time_sec, score])
            continue
        _prev_start, prev_t, prev_score = clustered[-1]
        if time_sec - prev_t <= 0.20:
            if score > prev_score:
                clustered[-1] = [start, time_sec, score]
        else:
            clustered.append([start, time_sec, score])

    return clustered, starts, windows, scores


def pick_dual_seat_peaks(clustered, max_events):
    """Keep strongest peaks with min gap; at most max_events seating clicks."""
    ranked = sorted(clustered, key=lambda item: item[2], reverse=True)
    chosen = []
    for start, time_sec, score in ranked:
        if any(abs(time_sec - ct) < 0.45 for _cs, ct, _cscore in chosen):
            continue
        chosen.append([start, time_sec, score])
        if len(chosen) >= max_events:
            break
    chosen.sort(key=lambda item: item[1])
    return chosen


def windows_near_starts(audio, peak_starts, neighbor_steps=1):
    """Positive windows: peak center plus nearby frames."""
    selected = []
    seen = set()
    for peak_start in peak_starts:
        center = int(peak_start)
        for delta in range(-neighbor_steps, neighbor_steps + 1):
            start = center + delta * EVENT_STEP
            if start < 0 or start + WINDOW > len(audio):
                continue
            if start in seen:
                continue
            seen.add(start)
            selected.append(audio[start:start + WINDOW])
    return selected


def build_dual_j1j2_samples(one_files, two_files, no_files):
    """
    J1_J2 training without J3 screw-region cropping.
    - Two-click files: top 2 strong peaks = seating positives; other peaks = hard negatives
    - One-click files: top 1 strong peak = seating positive; other peaks = hard negatives
    - No-click files: subsampled windows as negatives (prefer energetic handling)
    """
    X = []
    y = []

    def add_windows(windows, label):
        for window in windows:
            X.append(extract_features(window))
            y.append(label)

    for file in two_files:
        audio, _ = librosa.load(file, sr=SR, mono=True)
        clustered, _starts, _windows, scores = dual_peak_candidates(audio)
        seats = pick_dual_seat_peaks(clustered, max_events=2)
        seat_times = {round(t, 2) for _s, t, _sc in seats}
        pos = windows_near_starts(audio, [s for s, _t, _sc in seats], neighbor_steps=1)
        hard_neg_starts = [
            s for s, t, _sc in clustered if round(t, 2) not in seat_times
        ]
        neg = windows_near_starts(audio, hard_neg_starts, neighbor_steps=0)
        # Fill with sparse background negatives away from seats.
        all_starts, all_windows = get_windows(audio, step=STEP * 5)
        background = []
        for start, window in zip(all_starts, all_windows):
            t = start / SR
            if any(abs(t - st) < 0.35 for _s, st, _sc in seats):
                continue
            background.append(window)
        if len(background) > 40:
            idx = np.linspace(0, len(background) - 1, 40).astype(int)
            background = [background[i] for i in idx]
        add_windows(pos, 1)
        add_windows(neg + background, 0)
        print(
            f"TWO {os.path.basename(file)} | peaks={len(clustered)} | "
            f"seats={len(seats)} | pos={len(pos)} | neg={len(neg) + len(background)}"
        )

    for file in one_files:
        audio, _ = librosa.load(file, sr=SR, mono=True)
        clustered, _starts, _windows, scores = dual_peak_candidates(audio)
        seats = pick_dual_seat_peaks(clustered, max_events=1)
        seat_times = {round(t, 2) for _s, t, _sc in seats}
        pos = windows_near_starts(audio, [s for s, _t, _sc in seats], neighbor_steps=1)
        hard_neg_starts = [
            s for s, t, _sc in clustered if round(t, 2) not in seat_times
        ]
        neg = windows_near_starts(audio, hard_neg_starts, neighbor_steps=0)
        all_starts, all_windows = get_windows(audio, step=STEP * 5)
        background = []
        for start, window in zip(all_starts, all_windows):
            t = start / SR
            if any(abs(t - st) < 0.35 for _s, st, _sc in seats):
                continue
            background.append(window)
        if len(background) > 40:
            idx = np.linspace(0, len(background) - 1, 40).astype(int)
            background = [background[i] for i in idx]
        add_windows(pos, 1)
        add_windows(neg + background, 0)
        print(
            f"ONE {os.path.basename(file)} | peaks={len(clustered)} | "
            f"seats={len(seats)} | pos={len(pos)} | neg={len(neg) + len(background)}"
        )

    for file in no_files:
        audio, _ = librosa.load(file, sr=SR, mono=True)
        starts, windows = get_windows(audio, step=STEP)
        if not windows:
            continue
        scores = np.array([window_score(window) for window in windows], dtype=np.float64)
        # Prefer louder/handling windows as useful negatives, plus a sparse full-file sample.
        keep_hi = max(40, int(len(windows) * 0.40))
        hi_idx = np.argsort(scores)[-keep_hi:]
        spaced = np.linspace(0, len(windows) - 1, min(120, len(windows))).astype(int)
        chosen = sorted(set(hi_idx.tolist() + spaced.tolist()))
        add_windows([windows[i] for i in chosen], 0)
        # Extra hard negatives on energy peaks (intentional handling).
        clustered, _s, _w, _sc = dual_peak_candidates(audio)
        peak_neg = windows_near_starts(
            audio, [s for s, _t, _sc in clustered], neighbor_steps=1
        )
        # Repeat peak hard-negatives so handling is not drowned by quiet frames.
        add_windows(peak_neg, 0)
        add_windows(peak_neg, 0)
        print(
            f"NO  {os.path.basename(file)} | neg_windows={len(chosen)} | "
            f"peak_hard_neg={len(peak_neg)}x2"
        )

    return X, y


def load_files(folder, today_only=False):
    files = glob.glob(os.path.join(folder, "*.wav"))
    if not today_only:
        return files

    today = datetime.now().date()
    selected = []
    for path in files:
        modified = datetime.fromtimestamp(os.path.getmtime(path)).date()
        if modified == today:
            selected.append(path)
    return selected


def build_samples(files, label):
    X = []
    y = []

    for file in files:
        audio, _ = librosa.load(file, sr=SR, mono=True)

        if label == 1:
            windows, region_start, events, rule = select_click_windows(audio)
            print(
                f"CLICK {os.path.basename(file)} | "
                f"events={len(events)} | "
                f"region_start={region_start:.2f}s | "
                f"rule={rule} | "
                f"windows={len(windows)}"
            )
        else:
            _, windows = get_windows(audio, step=STEP)

            if len(windows) > 400:
                index = np.linspace(0, len(windows) - 1, 400).astype(int)
                windows = [windows[i] for i in index]

        for window in windows:
            X.append(extract_features(window))
            y.append(label)

    return X, y


def train_session(session_id, today_only=False):
    paths = session_paths(session_id)
    click_dir = paths["click_dir"]
    non_click_dir = paths["non_click_dir"]
    one_click_dir = paths["one_click_dir"]
    two_click_dir = paths["two_click_dir"]
    model_path = paths["model_path"]

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for folder in (click_dir, non_click_dir, one_click_dir, two_click_dir):
        os.makedirs(folder, exist_ok=True)

    if session_id == "J1_J2":
        one_files = load_files(one_click_dir, today_only=today_only)
        two_files = load_files(two_click_dir, today_only=today_only)
        no_files = load_files(non_click_dir, today_only=today_only)
        click_files = one_files + two_files
        non_click_files = no_files

        if len(click_files) < 2:
            raise ValueError(
                f"Need at least 2 One/Two click files under "
                f"{one_click_dir} or {two_click_dir}"
            )
        if len(non_click_files) < 2:
            raise ValueError(f"Need at least 2 non-click files in {non_click_dir}")

        one_train, one_test = train_test_split(
            one_files, test_size=0.25, random_state=42
        ) if len(one_files) >= 4 else (one_files, [])
        two_train, two_test = train_test_split(
            two_files, test_size=0.25, random_state=42
        ) if len(two_files) >= 4 else (two_files, [])
        no_train, no_test = train_test_split(
            non_click_files, test_size=0.25, random_state=42
        )

        print(f"Session: {session_id}")
        if today_only:
            print("Using today's audio files only")
        print("One-click files:", len(one_files))
        print("Two-click files:", len(two_files))
        print("No-click files:", len(non_click_files))
        print("Building J1_J2 samples on full clip (no screw-region crop)...")
        X_train, y_train = build_dual_j1j2_samples(one_train, two_train, no_train)
        print("Building test samples...")
        X_test, y_test = build_dual_j1j2_samples(one_test, two_test, no_test)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Non-click", "Click"],
            zero_division=0
        ))
        print("One train/test:", len(one_train), len(one_test))
        print("Two train/test:", len(two_train), len(two_test))
        print("No train/test:", len(no_train), len(no_test))
        print(
            "Train windows: click", int(np.sum(y_train == 1)),
            "non-click", int(np.sum(y_train == 0))
        )
        print(
            "Test windows: click", int(np.sum(y_test == 1)),
            "non-click", int(np.sum(y_test == 0))
        )
        joblib.dump(model, model_path)
        print("Model saved:", model_path)
        return model_path

    # J3 / single-connector path (unchanged).
    click_files = load_files(click_dir, today_only=today_only)
    non_click_files = load_files(non_click_dir, today_only=today_only)

    if len(click_files) < 2:
        raise ValueError(f"Need at least 2 click files in {click_dir}")

    if len(non_click_files) < 2:
        raise ValueError(f"Need at least 2 non-click files in {non_click_dir}")

    click_train, click_test = train_test_split(
        click_files,
        test_size=0.25,
        random_state=42
    )

    non_click_train, non_click_test = train_test_split(
        non_click_files,
        test_size=0.25,
        random_state=42
    )

    print(f"Session: {session_id}")
    if today_only:
        print("Using today's audio files only")
    print("Click files:", len(click_files))
    print("Non-click files:", len(non_click_files))
    print("Building training samples with post-screw connector region...")
    X_train_click, y_train_click = build_samples(click_train, 1)
    X_train_non, y_train_non = build_samples(non_click_train, 0)

    print("Building test samples...")
    X_test_click, y_test_click = build_samples(click_test, 1)
    X_test_non, y_test_non = build_samples(non_click_test, 0)

    X_train = np.array(X_train_click + X_train_non)
    y_train = np.array(y_train_click + y_train_non)

    X_test = np.array(X_test_click + X_test_non)
    y_test = np.array(y_test_click + y_test_non)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-click", "Click"],
        zero_division=0
    ))

    print("Click training files:", len(click_train))
    print("Click test files:", len(click_test))
    print("Non-click training files:", len(non_click_train))
    print("Non-click test files:", len(non_click_test))
    print("Train windows: click", len(X_train_click), "non-click", len(X_train_non))
    print("Test windows: click", len(X_test_click), "non-click", len(X_test_non))

    joblib.dump(model, model_path)
    print("Model saved:", model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train connector fit detector")
    parser.add_argument(
        "--session",
        default=DEFAULT_SESSION,
        help="Training session id, e.g. J3, J1_J2, Front, or Rear"
    )
    parser.add_argument(
        "--today-only",
        action="store_true",
        help="Train using only audio files modified today"
    )
    args = parser.parse_args()
    train_session(args.session, today_only=args.today_only)


if __name__ == "__main__":
    main()
