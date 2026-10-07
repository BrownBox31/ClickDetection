import os
import sys
import json
import shutil
import threading
import time
import subprocess
from datetime import datetime
from tkinter import END, BOTH, LEFT, RIGHT, W, E, X, Y, DoubleVar
from tkinter import StringVar, BooleanVar, Tk, Text, messagebox, Frame, Label, Canvas, Button
from tkinter import ttk, filedialog, Toplevel, Entry

import joblib
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from PIL import Image, ImageTk
from scipy.stats import kurtosis

SR = 16000
CHANNELS = 1
WINDOW = int(0.20 * SR)
STEP = int(0.10 * SR)
DETECT_STEP = int(0.15 * SR)  # 0.15 s windows; denser than 0.20, still faster than 0.10
EVENT_STEP = int(0.05 * SR)
EVENT_MERGE_SEC = 0.70
EVENT_PERCENTILE = 90
EVENT_MIN_SCORE = 40.0
DETECT_THRESHOLD_DEFAULT = 0.60
QUIET_ROOM_THRESHOLD_OFFSET = 0.10
QUIET_ROOM_THRESHOLD_FLOOR = 0.55
QUIET_ROOM_EVENT_MIN_SCORE = 30.0
WAVE_DISPLAY_SECONDS = 8.0
WAVE_DISPLAY_SAMPLES = int(WAVE_DISPLAY_SECONDS * SR)
RESULT_TEXT_HEIGHT = 40
WAVE_PLOT_HEIGHT = 170
WAVE_CTRL_HEIGHT = 52
RESULT_PANEL_HEIGHT = RESULT_TEXT_HEIGHT + WAVE_PLOT_HEIGHT + WAVE_CTRL_HEIGHT
DECISION_MARK_FG = "#F5D76E"
PLAYHEAD_FG = "#FFFFFF"
# Dual-click event rules (J1 then J2). No screw sequence — seating clicks only.
DUAL_MIN_EVENT_GAP_SEC = 0.45
DUAL_CLUSTER_SEC = 0.20
DUAL_MAX_EVENTS = 2
# Second seat must stay close to the strongest click so handling spikes are dropped.
DUAL_SECONDARY_MIN_RATIO = 0.70
WARN_BG = "#3D2F0F"
WARN_FG = "#F0C24B"
PLOT_AXIS = "#5B6A86"
PLOT_GRID = "#243049"
PLOT_PAD_L = 52
PLOT_PAD_R = 16
PLOT_PAD_T = 12
PLOT_PAD_B = 28

# Packaged exe: keep writable data next to the executable (not inside _internal).
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_ROOT = os.path.join(APP_DIR, "sessions")
LOGO_PATH = os.path.join(APP_DIR, "assets", "bbi_logo.png")
ICON_PATH = os.path.join(APP_DIR, "assets", "bbi_logo.ico")
# Bundled assets fallback for PyInstaller one-dir builds.
if getattr(sys, "frozen", False):
    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass:
        _bundled_logo = os.path.join(_meipass, "assets", "bbi_logo.png")
        _bundled_icon = os.path.join(_meipass, "assets", "bbi_logo.ico")
        if os.path.exists(_bundled_logo):
            LOGO_PATH = _bundled_logo
        if os.path.exists(_bundled_icon):
            ICON_PATH = _bundled_icon

SESSION_OPTIONS = [
    ("J3", "J3 Connector"),
    ("J1_J2", "J1 + J2 Connectors"),
    ("Front", "Front"),
    ("Rear", "Rear"),
]

CAPTURE_PREFIX = {
    "Click": "native_click_",
    "No click": "native_noclick_",
    "Two click": "native_twoclick_",
    "One click": "native_oneclick_",
}

DUAL_TRAIN_LABELS = ("Two click", "One click", "No click")
SINGLE_TRAIN_LABELS = ("Click", "No click")
TEACHING_BATCH_FILE = "teaching_batch.json"
DAILY_COUNTER_FILE = "daily_counter.json"

# License / redeem / online-time gate (isolated module — does not affect detection).
from license_gate import (  # noqa: E402
    POC_EXPIRE_DATE,
    evaluate_poc_license,
    poc_license_path,
    redeem_unlock_code,
)

# Brand palette
BG = "#0B1220"
CARD = "#121A2B"
CARD_ALT = "#182235"
ACCENT = "#2F6BFF"
TEXT = "#E8EEF8"
MUTED = "#93A0B8"
OK_BG = "#0F3D2E"
OK_FG = "#3DDC97"
BAD_BG = "#3D1218"
BAD_FG = "#FF6B7A"
IDLE_BG = "#1A2233"
IDLE_FG = "#AEB8CC"


def session_paths(session_id):
    root = os.path.join(SESSIONS_ROOT, session_id)
    return {
        "root": root,
        "click_dir": os.path.join(root, "training_audio", "CLICK_FILES"),
        "noclick_dir": os.path.join(root, "training_audio", "NO_CLICK_FILES"),
        "oneclick_dir": os.path.join(root, "training_audio", "ONE_CLICK_FILES"),
        "twoclick_dir": os.path.join(root, "training_audio", "TWO_CLICK_FILES"),
        "model_path": os.path.join(root, "model", "click_detector.pkl"),
        "audio_dir": os.path.join(root, "detection_runs", "audio"),
        "log_dir": os.path.join(root, "detection_runs", "logs"),
    }


def ensure_session_dirs(session_id):
    paths = session_paths(session_id)
    for key in ("click_dir", "noclick_dir", "oneclick_dir", "twoclick_dir", "audio_dir", "log_dir"):
        os.makedirs(paths[key], exist_ok=True)
    os.makedirs(os.path.dirname(paths["model_path"]), exist_ok=True)
    return paths


class NativeWireFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wire Fit Detection | Brown Box Innovations")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)
        self.root.configure(bg=BG)

        if os.path.exists(ICON_PATH):
            try:
                self.root.iconbitmap(ICON_PATH)
            except Exception:
                pass

        self.model = None
        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self.saved_files = []
        self.lock = threading.Lock()
        self.logo_image = None
        self.input_devices = []
        self.wave_dirty = False
        self.recorded_samples = 0
        self._record_poll_job = None
        self._last_stream_status = ""
        self.poc_locked = False
        self.poc_info = None

        self.mode_var = StringVar(value="Detection")
        self.label_var = StringVar(value="Click")
        self.device_var = StringVar(value="")
        self.status_var = StringVar(value="Ready")
        self.mic_var = StringVar(value="Microphone: checking...")
        self.click_count_var = StringVar(value="")
        self.noclick_count_var = StringVar(value="")
        self.mid_count_var = StringVar(value="")
        self.session_status_var = StringVar(value="")
        self.daily_count_var = StringVar(value="Today: 0 detections")
        self.help_var = StringVar(value="")
        self.result_var = StringVar(value="Awaiting detection")
        self.poc_var = StringVar(value="")
        self.busy = BooleanVar(value=False)
        self.threshold_var = DoubleVar(value=DETECT_THRESHOLD_DEFAULT)
        self.quiet_room_var = BooleanVar(value=False)
        self.effective_thr_var = StringVar(value="")
        self.session_var = StringVar(value=SESSION_OPTIONS[0][1])
        self.session_id = SESSION_OPTIONS[0][0]
        self._prev_session_id = None
        self.last_detection_audio = None
        self.last_detection_log = None
        self.last_detection_final = None
        self.last_detection_dual = None
        self.last_detection_marked = False
        self.wave_samples = np.zeros(0, dtype=np.float32)
        self.wave_pending = False
        self.showing_waveform = False
        self.showing_fixed_wave = False
        self.fixed_wave_audio = None
        self.decision_markers = []
        self.playhead_sec = None
        self.playback_active = False
        self.playback_start_perf = None
        self.playback_duration = 0.0
        self.loaded_audio_name = ""
        self.banner_state = "IDLE"
        self.banner_fg = IDLE_FG
        self.banner_bg = IDLE_BG

        self.configure_styles()
        self.build_ui()
        self.refresh_devices()
        self.on_session_change()
        self.apply_poc_license()
        self._schedule_poc_refresh()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        field_bg = "#F4F7FC"
        field_fg = "#0B1220"

        style.configure(".", background=BG, foreground=TEXT, fieldbackground=CARD)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 13))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 12))
        style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 13))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 28))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 13))
        style.configure("Brand.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 12, "bold"))
        style.configure("TLabelframe", background=CARD, foreground=TEXT, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=CARD, foreground=MUTED, font=("Segoe UI", 12, "bold"))
        style.configure("TRadiobutton", background=CARD, foreground=TEXT, font=("Segoe UI", 13))
        style.configure("TButton", font=("Segoe UI Semibold", 13), padding=(14, 10))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 16), padding=(28, 14))
        style.configure("Big.TButton", font=("Segoe UI Semibold", 13), padding=(16, 10))
        style.configure("MarkFitted.TButton", font=("Segoe UI Semibold", 13), padding=(14, 12))
        style.configure("MarkOneFitted.TButton", font=("Segoe UI Semibold", 13), padding=(14, 12))
        style.configure("MarkNotFitted.TButton", font=("Segoe UI Semibold", 13), padding=(14, 12))
        style.map("TRadiobutton", background=[("active", CARD)], foreground=[("disabled", MUTED)])

        style.configure(
            "TCombobox",
            fieldbackground=field_bg,
            background=field_bg,
            foreground=field_fg,
            arrowcolor=field_fg,
            insertcolor=field_fg,
            padding=8
        )
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", field_bg),
                ("!disabled", field_bg),
                ("disabled", "#2A3448"),
            ],
            foreground=[
                ("readonly", field_fg),
                ("!disabled", field_fg),
                ("disabled", MUTED),
            ],
            selectbackground=[("readonly", ACCENT), ("!disabled", ACCENT)],
            selectforeground=[("readonly", "#FFFFFF"), ("!disabled", "#FFFFFF")],
            background=[("readonly", field_bg), ("!disabled", field_bg)],
        )

        style.configure(
            "TSpinbox",
            fieldbackground=field_bg,
            background=field_bg,
            foreground=field_fg,
            arrowcolor=field_fg,
            insertcolor=field_fg,
            padding=8
        )
        style.map(
            "TSpinbox",
            fieldbackground=[("!disabled", field_bg)],
            foreground=[("!disabled", field_fg)],
            background=[("!disabled", field_bg)],
        )

        self.root.option_add("*TCombobox*Listbox.background", field_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", field_fg)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
        self.root.option_add("*TCombobox*Listbox.font", "{Segoe UI} 13")

    def apply_poc_license(self, show_warning=True):
        """UI/action lock only — does not change detection or delete audio."""
        was_locked = bool(self.poc_locked)
        info = evaluate_poc_license()
        self.poc_info = info
        self.poc_locked = not info["ok"]
        self.poc_var.set(info["message"])
        if hasattr(self, "poc_label"):
            self.poc_label.configure(fg=BAD_FG if self.poc_locked else MUTED)
        if hasattr(self, "redeem_button"):
            # Always allow opening redeem dialog (also useful near expiry).
            self.redeem_button.configure(state="normal")

        if self.poc_locked:
            try:
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="disabled")
            except Exception:
                pass
            try:
                self.train_button.configure(state="disabled")
            except Exception:
                pass
            try:
                if hasattr(self, "import_train_button"):
                    self.import_train_button.configure(state="disabled")
            except Exception:
                pass
            self.set_mark_buttons_enabled(False)
            self.status_var.set("POC locked")
            if not was_locked:
                self.log(info["message"])
            if show_warning and not was_locked:
                messagebox.showwarning("POC license", info["message"])
        else:
            if was_locked:
                self.log(info["message"])
            elif show_warning:
                self.log(info["message"])
            if not self.recording and not self.busy.get():
                try:
                    self.start_button.configure(state="normal")
                except Exception:
                    pass

    def ensure_poc_allowed(self, action="continue"):
        # Re-check every action so expiry locks even if the app stayed open.
        self.apply_poc_license(show_warning=False)
        if not self.poc_locked:
            return True
        message = (self.poc_info or {}).get("message") or (
            "POC is locked. Detect / Train / Record are unavailable. "
            "All audio remains on disk."
        )
        messagebox.showwarning("POC license", f"Cannot {action}.\n\n{message}")
        return False

    def _schedule_poc_refresh(self):
        self.root.after(10000, self._poc_refresh_tick)

    def _poc_refresh_tick(self):
        try:
            if not self.root.winfo_exists():
                return
            was_locked = bool(self.poc_locked)
            self.apply_poc_license(show_warning=False)
            if self.poc_locked and not was_locked:
                messagebox.showwarning(
                    "POC license",
                    (self.poc_info or {}).get("message") or "POC expired.",
                )
        except Exception:
            pass
        try:
            if self.root.winfo_exists():
                self.root.after(10000, self._poc_refresh_tick)
        except Exception:
            pass

    def open_redeem_dialog(self):
        win = Toplevel(self.root)
        win.title("Redeem license code")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        Label(
            win,
            text="Enter one-time redeem code",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 12),
        ).pack(anchor=W, padx=16, pady=(16, 6))
        Label(
            win,
            text="Each code extends the POC by 15 days and cannot be reused.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor=W, padx=16, pady=(0, 10))

        code_var = StringVar(value="")
        entry = Entry(
            win,
            textvariable=code_var,
            width=42,
            font=("Consolas", 12),
        )
        entry.pack(padx=16, pady=(0, 12))
        entry.focus_set()

        def submit():
            result = redeem_unlock_code(code_var.get())
            if not result.get("ok"):
                messagebox.showerror("Redeem failed", result.get("message") or "Invalid code.", parent=win)
                return
            messagebox.showinfo("Redeem OK", result.get("message"), parent=win)
            win.destroy()
            self.apply_poc_license(show_warning=False)
            if not self.poc_locked:
                self.on_mode_change()

        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill=X, padx=16, pady=(0, 16))
        Button(
            btn_row,
            text="Redeem",
            command=submit,
            bg=ACCENT,
            fg="#FFFFFF",
            activebackground="#4A7FFF",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=14,
            pady=6,
        ).pack(side=LEFT)
        Button(
            btn_row,
            text="Cancel",
            command=win.destroy,
            bg=CARD_ALT,
            fg=TEXT,
            activebackground="#243049",
            activeforeground=TEXT,
            relief="flat",
            padx=14,
            pady=6,
        ).pack(side=LEFT, padx=(8, 0))
        win.bind("<Return>", lambda _e: submit())

    def build_ui(self):
        outer = Frame(self.root, bg=BG)
        outer.pack(fill=BOTH, expand=True)

        self.main_canvas = Canvas(outer, bg=BG, highlightthickness=0, bd=0)
        self.main_scrollbar = ttk.Scrollbar(
            outer, orient="vertical", command=self.main_canvas.yview
        )
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        self.main_scrollbar.pack(side=RIGHT, fill=Y)
        self.main_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        shell = Frame(self.main_canvas, bg=BG)
        self._shell_window = self.main_canvas.create_window(
            (0, 0), window=shell, anchor="nw"
        )
        shell.bind("<Configure>", self._on_shell_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        content = Frame(shell, bg=BG)
        content.pack(fill=BOTH, expand=True, padx=18, pady=12)

        header = Frame(content, bg=BG)
        header.pack(fill=X, pady=(0, 8))

        logo_wrap = Frame(header, bg=BG)
        logo_wrap.pack(side=LEFT, padx=(0, 12))
        self.logo_label = Label(logo_wrap, bg=BG)
        self.logo_label.pack()
        self.load_logo()

        title_wrap = Frame(header, bg=BG)
        title_wrap.pack(side=LEFT, fill=X, expand=True)
        Label(
            title_wrap,
            text="Wire Fit Detection",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 18)
        ).pack(anchor=W)
        Label(
            title_wrap,
            text="Brown Box Innovations  ·  Connector seating check",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 11)
        ).pack(anchor=W, pady=(2, 0))
        self.poc_label = Label(
            title_wrap,
            textvariable=self.poc_var,
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 10),
        )
        self.poc_label.pack(anchor=W, pady=(2, 0))
        self.redeem_button = Button(
            title_wrap,
            text="Redeem code…",
            command=self.open_redeem_dialog,
            bg=CARD_ALT,
            fg=TEXT,
            activebackground="#243049",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            font=("Segoe UI", 9),
        )
        self.redeem_button.pack(anchor=W, pady=(6, 0))

        # Primary station view: result + graph + start/stop stay together.
        self.result_banner = Frame(content, bg=IDLE_BG, height=RESULT_PANEL_HEIGHT)
        self.result_banner.pack(fill=X, pady=(0, 8))
        self.result_banner.pack_propagate(False)

        self.result_text = Label(
            self.result_banner,
            textvariable=self.result_var,
            bg=IDLE_BG,
            fg=IDLE_FG,
            font=("Segoe UI Semibold", 20),
            pady=4
        )
        self.result_text.pack(fill=X)

        self.wave_canvas = Canvas(
            self.result_banner,
            bg=IDLE_BG,
            highlightthickness=0,
            bd=0,
            height=WAVE_PLOT_HEIGHT
        )
        self.wave_canvas.pack(fill=BOTH, expand=True)
        self.wave_canvas.bind("<Configure>", lambda _e: self.redraw_result_banner())

        self.wave_controls = Frame(self.result_banner, bg=IDLE_BG, height=WAVE_CTRL_HEIGHT)
        self.wave_controls.pack(fill=X, padx=10, pady=(2, 6))
        self.wave_controls.pack_propagate(False)

        btn_kwargs = dict(
            font=("Segoe UI Semibold", 12),
            fg="#FFFFFF",
            activeforeground="#FFFFFF",
            disabledforeground="#A8B4CC",
            relief="flat",
            bd=0,
            padx=16,
            pady=7,
            cursor="hand2",
        )
        self.start_button = Button(
            self.wave_controls,
            text="START",
            command=self.start_recording,
            bg="#1F9D63",
            activebackground="#27B875",
            **btn_kwargs,
        )
        self.start_button.pack(side=LEFT, pady=5)
        self.stop_button = Button(
            self.wave_controls,
            text="STOP",
            command=self.stop_recording,
            state="disabled",
            bg="#C44A57",
            activebackground="#E05A68",
            **btn_kwargs,
        )
        self.stop_button.pack(side=LEFT, padx=(8, 0), pady=5)

        sep = Frame(self.wave_controls, bg=PLOT_AXIS, width=1)
        sep.pack(side=LEFT, fill=Y, padx=12, pady=8)

        open_play_kwargs = dict(btn_kwargs)
        open_play_kwargs.update(bg=ACCENT, activebackground="#4A7FFF")
        self.open_audio_button = Button(
            self.wave_controls,
            text="Open…",
            command=self.open_audio_file,
            **open_play_kwargs,
        )
        self.open_audio_button.pack(side=LEFT, pady=5)
        self.play_button = Button(
            self.wave_controls,
            text="Play",
            command=self.toggle_playback,
            state="disabled",
            **open_play_kwargs,
        )
        self.play_button.pack(side=LEFT, padx=(8, 0), pady=5)
        self.play_status_var = StringVar(value="")
        self.play_status_label = Label(
            self.wave_controls,
            textvariable=self.play_status_var,
            bg=IDLE_BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            anchor=W,
        )
        self.play_status_label.pack(side=LEFT, fill=X, expand=True, padx=(12, 0))
        Label(
            self.wave_controls,
            textvariable=self.status_var,
            bg=IDLE_BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            anchor=E,
        ).pack(side=RIGHT, padx=(8, 4))
        self.loaded_audio_name = ""
        self.set_result_banner("IDLE")

        self.mark_bar = Frame(content, bg=BG)

        self.mark_single_row = Frame(self.mark_bar, bg=BG)
        self.mark_fitted_button = ttk.Button(
            self.mark_single_row,
            text="Mark as Fitted",
            style="MarkFitted.TButton",
            command=lambda: self.mark_detection("YES"),
            state="disabled"
        )
        self.mark_fitted_button.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        self.mark_not_fitted_button = ttk.Button(
            self.mark_single_row,
            text="Mark as Not Fitted",
            style="MarkNotFitted.TButton",
            command=lambda: self.mark_detection("NO"),
            state="disabled"
        )
        self.mark_not_fitted_button.pack(side=LEFT, fill=X, expand=True, padx=(8, 0))

        self.mark_dual_row = Frame(self.mark_bar, bg=BG)
        self.mark_two_button = ttk.Button(
            self.mark_dual_row,
            text="Mark as Two Fitted",
            style="MarkFitted.TButton",
            command=lambda: self.mark_detection("TWO"),
            state="disabled"
        )
        self.mark_two_button.pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
        self.mark_one_button = ttk.Button(
            self.mark_dual_row,
            text="Mark as One Fitted",
            style="MarkOneFitted.TButton",
            command=lambda: self.mark_detection("ONE"),
            state="disabled"
        )
        self.mark_one_button.pack(side=LEFT, fill=X, expand=True, padx=(6, 6))
        self.mark_none_button = ttk.Button(
            self.mark_dual_row,
            text="Mark as None Fitted",
            style="MarkNotFitted.TButton",
            command=lambda: self.mark_detection("NONE"),
            state="disabled"
        )
        self.mark_none_button.pack(side=LEFT, fill=X, expand=True, padx=(6, 0))

        self.mark_bar.pack(fill=X, pady=(0, 8))

        self.counter_bar = Frame(content, bg=CARD_ALT, padx=12, pady=6)
        self.counter_bar.pack(fill=X, pady=(0, 8))
        Label(
            self.counter_bar,
            textvariable=self.daily_count_var,
            bg=CARD_ALT,
            fg=TEXT,
            font=("Segoe UI Semibold", 12),
            anchor=W
        ).pack(side=LEFT, fill=X, expand=True)
        ttk.Button(
            self.counter_bar,
            text="Reset counter",
            style="Big.TButton",
            command=self.reset_daily_counter
        ).pack(side=RIGHT)

        body = Frame(content, bg=BG)
        body.pack(fill=X, pady=(0, 8))
        body.columnconfigure(0, weight=3, uniform="cols")
        body.columnconfigure(1, weight=2, uniform="cols")

        left = Frame(body, bg=CARD, padx=12, pady=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = Frame(body, bg=CARD, padx=12, pady=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        Label(left, text="Training session", bg=CARD, fg=MUTED, font=("Segoe UI", 11, "bold")).pack(anchor=W)
        session_names = [name for _, name in SESSION_OPTIONS]
        self.session_combo = ttk.Combobox(
            left,
            textvariable=self.session_var,
            values=session_names,
            state="readonly",
            width=36,
            font=("Segoe UI", 12)
        )
        self.session_combo.pack(anchor=W, fill=X, pady=(4, 0))
        self.session_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_session_change())
        Label(
            left,
            textvariable=self.session_status_var,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 11),
            wraplength=620,
            justify=LEFT
        ).pack(anchor=W, pady=(4, 0))

        Label(left, text="Microphone", bg=CARD, fg=MUTED, font=("Segoe UI", 11, "bold")).pack(anchor=W, pady=(10, 0))
        mic_row = Frame(left, bg=CARD)
        mic_row.pack(fill=X, pady=(4, 0))
        self.device_combo = ttk.Combobox(
            mic_row,
            textvariable=self.device_var,
            state="readonly",
            font=("Segoe UI", 12)
        )
        self.device_combo.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        ttk.Button(mic_row, text="Refresh", style="Big.TButton", command=self.refresh_devices).pack(side=LEFT)
        Label(
            left,
            textvariable=self.mic_var,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 11)
        ).pack(anchor=W, pady=(4, 0))

        self.click_count_label = Label(
            left,
            textvariable=self.click_count_var,
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12)
        )
        self.click_count_label.pack(anchor=W, pady=(10, 0))
        self.mid_count_label = Label(
            left,
            textvariable=self.mid_count_var,
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12)
        )
        self.mid_count_label.pack(anchor=W, pady=(2, 0))
        self.noclick_count_label = Label(
            left,
            textvariable=self.noclick_count_var,
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12)
        )
        self.noclick_count_label.pack(anchor=W, pady=(2, 0))
        self.help_label = Label(
            left,
            textvariable=self.help_var,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 11),
            wraplength=620,
            justify=LEFT
        )
        self.help_label.pack(anchor=W, pady=(6, 0))

        Label(right, text="Mode", bg=CARD, fg=MUTED, font=("Segoe UI", 11, "bold")).pack(anchor=W)
        radios = Frame(right, bg=CARD)
        radios.pack(anchor=W, pady=(4, 0))
        ttk.Radiobutton(
            radios,
            text="Detection",
            value="Detection",
            variable=self.mode_var,
            command=self.on_mode_change
        ).pack(side=LEFT, padx=(0, 14))
        ttk.Radiobutton(
            radios,
            text="Training",
            value="Training",
            variable=self.mode_var,
            command=self.on_mode_change
        ).pack(side=LEFT)

        self.mode_options = Frame(right, bg=CARD)
        self.mode_options.pack(fill=X, pady=(8, 0))

        self.label_frame = Frame(self.mode_options, bg=CARD)
        Label(
            self.label_frame,
            text="Save recording as",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 11, "bold")
        ).pack(anchor=W)

        self.single_label_radios = Frame(self.label_frame, bg=CARD)
        ttk.Radiobutton(
            self.single_label_radios,
            text="Fitted (connector seated)",
            value="Click",
            variable=self.label_var,
            command=self.refresh_counts
        ).pack(anchor=W, pady=(0, 2))
        ttk.Radiobutton(
            self.single_label_radios,
            text="Not fitted",
            value="No click",
            variable=self.label_var,
            command=self.refresh_counts
        ).pack(anchor=W)

        self.dual_label_radios = Frame(self.label_frame, bg=CARD)
        ttk.Radiobutton(
            self.dual_label_radios,
            text="Two click (J1 + J2 both seated)",
            value="Two click",
            variable=self.label_var,
            command=self.refresh_counts
        ).pack(anchor=W, pady=(0, 2))
        ttk.Radiobutton(
            self.dual_label_radios,
            text="One click (only first / J1 seated)",
            value="One click",
            variable=self.label_var,
            command=self.refresh_counts
        ).pack(anchor=W, pady=(0, 2))
        ttk.Radiobutton(
            self.dual_label_radios,
            text="No click (none seated)",
            value="No click",
            variable=self.label_var,
            command=self.refresh_counts
        ).pack(anchor=W)

        batch_row = Frame(self.label_frame, bg=CARD)
        batch_row.pack(anchor=W, fill=X, pady=(8, 0))
        ttk.Button(
            batch_row,
            text="New teaching batch",
            style="Big.TButton",
            command=self.reset_teaching_batch
        ).pack(anchor=W)

        self.threshold_frame = Frame(self.mode_options, bg=CARD)
        Label(
            self.threshold_frame,
            text="Detection threshold",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 11, "bold")
        ).pack(anchor=W)
        thr_row = Frame(self.threshold_frame, bg=CARD)
        thr_row.pack(anchor=W, pady=(4, 0))
        Label(
            thr_row,
            text="Probability ≥",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12)
        ).pack(side=LEFT, padx=(0, 8))
        ttk.Spinbox(
            thr_row,
            from_=0.50,
            to=0.99,
            increment=0.05,
            textvariable=self.threshold_var,
            width=8,
            format="%.2f",
            font=("Segoe UI", 12),
            command=self.update_effective_threshold_label,
        ).pack(side=LEFT)
        quiet_row = Frame(self.threshold_frame, bg=CARD)
        quiet_row.pack(anchor=W, fill=X, pady=(8, 0))
        ttk.Checkbutton(
            quiet_row,
            text="Quiet room mode",
            variable=self.quiet_room_var,
            command=self.on_quiet_room_toggle,
        ).pack(side=LEFT)
        Label(
            quiet_row,
            textvariable=self.effective_thr_var,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(side=LEFT, padx=(12, 0))
        Label(
            self.threshold_frame,
            text="Quiet room lowers effective click threshold (−0.10, floor 0.55) for softer seating sounds.",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=360,
            justify=LEFT,
        ).pack(anchor=W, pady=(4, 0))
        self.update_effective_threshold_label()

        train_row = Frame(right, bg=CARD)
        train_row.pack(fill=X, pady=(12, 0))
        self.train_button = ttk.Button(
            train_row,
            text="Train / Retrain model",
            style="Big.TButton",
            command=self.train_current_session
        )
        self.train_button.pack(side=LEFT)
        self.import_train_button = ttk.Button(
            train_row,
            text="Import training WAVs…",
            style="Big.TButton",
            command=self.import_training_wavs_menu
        )
        self.import_train_button.pack(side=LEFT, padx=(8, 0))
        self.open_train_folder_button = ttk.Button(
            train_row,
            text="Open folder",
            style="Big.TButton",
            command=self.open_training_folder
        )
        self.open_train_folder_button.pack(side=LEFT, padx=(8, 0))
        Label(
            right,
            text=(
                "Each session (J3 / J1+J2 / Front / Rear) has its own model. "
                "Import received WAVs into the selected session, then Train / Retrain."
            ),
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=360,
            justify=LEFT,
        ).pack(anchor=W, pady=(8, 0))

        results_card = Frame(content, bg=CARD, padx=12, pady=8)
        results_card.pack(fill=X, pady=(0, 0))
        Label(
            results_card,
            text="Activity log",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 11, "bold")
        ).pack(anchor=W, pady=(0, 4))
        text_wrap = Frame(results_card, bg=CARD)
        text_wrap.pack(fill=X)
        self.results = Text(
            text_wrap,
            wrap="word",
            height=7,
            bg=CARD_ALT,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Consolas", 11),
            padx=10,
            pady=8
        )
        self.results.pack(side=LEFT, fill=X, expand=True)
        scroll = ttk.Scrollbar(text_wrap, orient="vertical", command=self.results.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.results.configure(yscrollcommand=scroll.set)

        self.on_mode_change()

    def _on_shell_configure(self, _event=None):
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event):
        self.main_canvas.itemconfigure(self._shell_window, width=event.width)

    def _on_mousewheel(self, event):
        target = self.root.winfo_containing(event.x_root, event.y_root)
        if target is not None and target == self.results:
            self.results.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def load_logo(self):
        if not os.path.exists(LOGO_PATH):
            self.logo_label.configure(text="BBI", fg=ACCENT, font=("Segoe UI Semibold", 18))
            return

        image = Image.open(LOGO_PATH).convert("RGBA")
        image.thumbnail((56, 56), Image.Resampling.LANCZOS)
        self.logo_image = ImageTk.PhotoImage(image)
        self.logo_label.configure(image=self.logo_image)

    def current_session_id(self):
        selected = self.session_var.get()
        for session_id, name in SESSION_OPTIONS:
            if name == selected:
                return session_id
        return SESSION_OPTIONS[0][0]

    def paths(self):
        return ensure_session_dirs(self.session_id)

    def is_dual_session(self):
        return self.session_id == "J1_J2"

    def training_labels(self):
        return DUAL_TRAIN_LABELS if self.is_dual_session() else SINGLE_TRAIN_LABELS

    def folder_for_label(self, label):
        paths = self.paths()
        mapping = {
            "Click": paths["click_dir"],
            "No click": paths["noclick_dir"],
            "One click": paths["oneclick_dir"],
            "Two click": paths["twoclick_dir"],
        }
        if label not in mapping:
            raise ValueError(f"Unknown training label: {label}")
        return mapping[label]

    def sync_training_label_options(self):
        self.single_label_radios.pack_forget()
        self.dual_label_radios.pack_forget()
        if self.is_dual_session():
            self.dual_label_radios.pack(anchor=W, pady=(6, 0))
            if self.label_var.get() not in DUAL_TRAIN_LABELS:
                self.label_var.set("Two click")
            self.mid_count_label.pack(
                anchor=W, pady=(4, 0), before=self.noclick_count_label
            )
        else:
            self.single_label_radios.pack(anchor=W, pady=(6, 0))
            if self.label_var.get() not in SINGLE_TRAIN_LABELS:
                self.label_var.set("Click")
            self.mid_count_label.pack_forget()
            self.mid_count_var.set("")

    def format_dual_banner_text(self, j1, j2):
        j1_text = "FITTED" if j1 == "YES" else "NOT FITTED"
        j2_text = "FITTED" if j2 == "YES" else "NOT FITTED"
        return f"J1: {j1_text}    |    J2: {j2_text}"

    def dual_banner_colors(self, j1, j2):
        if j1 == "YES" and j2 == "YES":
            return OK_BG, OK_FG
        if j1 == "NO" and j2 == "NO":
            return BAD_BG, BAD_FG
        return WARN_BG, WARN_FG

    def apply_banner_colors(self, bg, fg):
        self.banner_bg = bg
        self.banner_fg = fg
        self.result_banner.configure(bg=bg)
        self.result_text.configure(bg=bg, fg=fg)
        self.wave_canvas.configure(bg=bg)
        self.wave_controls.configure(bg=bg)
        for child in self.wave_controls.winfo_children():
            if isinstance(child, Label):
                child.configure(bg=bg)

    def set_dual_result_banner(self, j1, j2):
        self.showing_waveform = False
        self.banner_state = "DUAL"
        text = self.format_dual_banner_text(j1, j2)
        bg, fg = self.dual_banner_colors(j1, j2)
        self.result_var.set(text)
        self.result_text.configure(font=("Segoe UI Semibold", 18))
        self.apply_banner_colors(bg, fg)
        self.redraw_result_banner()

    def set_result_banner(self, state):
        self.showing_waveform = False
        self.banner_state = state
        if isinstance(state, dict) and state.get("dual"):
            self.set_dual_result_banner(state.get("j1", "NO"), state.get("j2", "NO"))
            return
        if state == "YES":
            self.result_var.set("CONNECTOR FITTED")
            bg, fg = OK_BG, OK_FG
            self.result_text.configure(font=("Segoe UI Semibold", 20))
        elif state == "NO":
            self.result_var.set("CONNECTOR NOT FITTED")
            bg, fg = BAD_BG, BAD_FG
            self.result_text.configure(font=("Segoe UI Semibold", 20))
        else:
            self.result_var.set("Awaiting detection")
            bg, fg = IDLE_BG, IDLE_FG
            self.result_text.configure(font=("Segoe UI Semibold", 20))
            self.stop_playback(clear_playhead=True, redraw=False)
            self.showing_fixed_wave = False
            self.fixed_wave_audio = None
            self.decision_markers = []
            self.loaded_audio_name = ""
            self.sync_play_button_state()
        self.apply_banner_colors(bg, fg)
        self.redraw_result_banner()

    def redraw_result_banner(self):
        if self.showing_waveform:
            self._draw_live_waveform()
        elif self.showing_fixed_wave and self.fixed_wave_audio is not None:
            self._draw_fixed_waveform(self.fixed_wave_audio)
        else:
            self._draw_empty_plot()

    def _draw_empty_plot(self):
        canvas = self.wave_canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 2)
        h = max(canvas.winfo_height(), 2)
        canvas.create_text(
            w // 2,
            h // 2,
            text="Waveform appears here after recording",
            fill=MUTED,
            font=("Segoe UI", 13)
        )

    def _plot_rect(self, canvas_w, canvas_h):
        left = PLOT_PAD_L
        right = max(left + 40, canvas_w - PLOT_PAD_R)
        top = PLOT_PAD_T
        bottom = max(top + 40, canvas_h - PLOT_PAD_B)
        return left, top, right, bottom

    def _draw_axes(
        self,
        canvas,
        left,
        top,
        right,
        bottom,
        duration_sec,
        y_min,
        y_max,
        y_label,
        show_x_label=True,
    ):
        canvas.create_rectangle(left, top, right, bottom, outline=PLOT_AXIS, width=1)
        # Horizontal mid / grid
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + (bottom - top) * frac
            canvas.create_line(left, y, right, y, fill=PLOT_GRID, width=1)
            value = y_max - (y_max - y_min) * frac
            canvas.create_text(
                left - 6,
                y,
                text=f"{value:.2f}",
                fill=MUTED,
                font=("Segoe UI", 9),
                anchor="e"
            )

        # Time ticks on X
        tick_count = 5 if duration_sec >= 1.0 else 4
        for i in range(tick_count + 1):
            frac = i / tick_count
            x = left + (right - left) * frac
            t = duration_sec * frac
            canvas.create_line(x, bottom, x, bottom + 4, fill=PLOT_AXIS, width=1)
            canvas.create_line(x, top, x, bottom, fill=PLOT_GRID, width=1)
            if show_x_label:
                canvas.create_text(
                    x,
                    bottom + 10,
                    text=f"{t:.1f}s",
                    fill=MUTED,
                    font=("Segoe UI", 9),
                    anchor="n"
                )

        canvas.create_text(
            14,
            (top + bottom) / 2,
            text=y_label,
            fill=MUTED,
            font=("Segoe UI", 9),
            angle=90,
            anchor="center"
        )
        if show_x_label:
            canvas.create_text(
                (left + right) / 2,
                bottom + 22,
                text="Time (s)",
                fill=MUTED,
                font=("Segoe UI", 9),
                anchor="n"
            )

    def _time_to_x(self, t_sec, left, right, duration):
        if duration <= 0:
            return left
        frac = max(0.0, min(1.0, float(t_sec) / duration))
        return left + frac * (right - left)

    def _draw_decision_markers(self, canvas, left, top, right, bottom, duration):
        for marker in self.decision_markers:
            t = float(marker.get("t", 0.0))
            label = str(marker.get("label", ""))
            prob = marker.get("prob")
            negative = bool(marker.get("negative"))
            color = BAD_FG if negative else DECISION_MARK_FG
            x = self._time_to_x(t, left, right, duration)
            canvas.create_line(x, top, x, bottom, fill=color, width=2, dash=(5, 3))
            canvas.create_polygon(
                x - 7, top + 1,
                x + 7, top + 1,
                x, top + 12,
                fill=color,
                outline=color,
            )
            text = label if prob is None else f"{label} {float(prob):.2f}"
            canvas.create_text(
                x,
                top + 14,
                text=text,
                fill=color,
                font=("Segoe UI Semibold", 10),
                anchor="n",
            )

    def _draw_playhead(self, canvas, left, top, right, bottom, duration):
        if self.playhead_sec is None or duration <= 0:
            return
        x = self._time_to_x(self.playhead_sec, left, right, duration)
        canvas.create_line(x, top, x, bottom, fill=PLAYHEAD_FG, width=2)
        canvas.create_oval(x - 4, top - 2, x + 4, top + 6, fill=PLAYHEAD_FG, outline=PLAYHEAD_FG)

    def _draw_live_waveform(self):
        canvas = self.wave_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 4 or h < 4:
            return

        left, top, right, bottom = self._plot_rect(w, h)
        with self.lock:
            samples = self.wave_samples.copy()

        duration = WAVE_DISPLAY_SECONDS
        if len(samples) >= 2:
            duration = len(samples) / SR

        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        y_lim = max(peak * 1.1, 0.05)
        self._draw_axes(canvas, left, top, right, bottom, duration, -y_lim, y_lim, "Amplitude")

        if len(samples) < 2:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Listening...",
                fill=MUTED,
                font=("Segoe UI Semibold", 18)
            )
            return

        points = max(int(right - left), 2)
        idx = np.linspace(0, len(samples) - 1, points).astype(np.int32)
        ys = samples[idx]
        mid = (top + bottom) / 2.0
        amp = (bottom - top) / 2.0 / y_lim
        coords = []
        for i, sample in enumerate(ys):
            x = left + i * (right - left) / (points - 1)
            y = mid - float(sample) * amp
            y = min(bottom - 1, max(top + 1, y))
            coords.extend((x, y))
        if len(coords) >= 4:
            canvas.create_line(*coords, fill=ACCENT, width=2, smooth=True)

    def _draw_fixed_waveform(self, audio):
        """Draw fixed capture amplitude waveform with decision markers and playhead."""
        canvas = self.wave_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 4 or h < 4:
            return

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(samples) < 2:
            self._draw_empty_plot()
            return

        duration = len(samples) / SR
        left, top, right, bottom = self._plot_rect(w, h)

        peak = float(np.max(np.abs(samples)))
        y_lim = max(peak * 1.05, 0.05)
        self._draw_axes(
            canvas, left, top, right, bottom, duration, -y_lim, y_lim, "Amplitude",
        )

        points = max(int(right - left), 2)
        idx = np.linspace(0, len(samples) - 1, points).astype(np.int32)
        ys = samples[idx]
        mid = (top + bottom) / 2.0
        amp = (bottom - top) / 2.0 / y_lim
        coords = []
        for i, sample in enumerate(ys):
            x = left + i * (right - left) / (points - 1)
            y = mid - float(sample) * amp
            y = min(bottom - 1, max(top + 1, y))
            coords.extend((x, y))
        if len(coords) >= 4:
            canvas.create_line(*coords, fill=ACCENT, width=2, smooth=True)

        self._draw_decision_markers(canvas, left, top, right, bottom, duration)
        self._draw_playhead(canvas, left, top, right, bottom, duration)

        rms = float(np.sqrt(np.mean(samples ** 2)))
        info = f"Peak amp {peak:.3f}  ·  RMS {rms:.3f}  ·  {duration:.2f}s"
        canvas.create_text(
            right,
            top + 2,
            text=info,
            fill=MUTED,
            font=("Segoe UI", 9),
            anchor="ne"
        )

    def show_fixed_waveform(self, audio, markers=None):
        self.stop_playback(clear_playhead=True, redraw=False)
        self.showing_waveform = False
        self.showing_fixed_wave = True
        self.fixed_wave_audio = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if markers is not None:
            self.decision_markers = list(markers)
        self.sync_play_button_state()
        self.redraw_result_banner()

    def sync_play_button_state(self):
        can_play = (
            self.showing_fixed_wave
            and self.fixed_wave_audio is not None
            and len(self.fixed_wave_audio) >= 2
            and not self.recording
        )
        state = "normal" if can_play else "disabled"
        if hasattr(self, "play_button"):
            self.play_button.configure(state=state)
            if not self.playback_active:
                self.play_button.configure(text="Play")
        if hasattr(self, "open_audio_button"):
            open_state = "disabled" if (self.recording or self.busy.get()) else "normal"
            self.open_audio_button.configure(state=open_state)
        if not self.playback_active and hasattr(self, "play_status_var"):
            if can_play and self.loaded_audio_name:
                self.play_status_var.set(self.loaded_audio_name)
            elif not can_play:
                self.play_status_var.set("")

    def open_audio_file(self):
        if self.recording or self.busy.get():
            return
        paths = self.paths()
        candidates = [
            paths.get("audio_dir"),
            paths.get("twoclick_dir"),
            paths.get("oneclick_dir"),
            paths.get("noclick_dir"),
            paths.get("click_dir"),
            APP_DIR,
        ]
        initial = next((p for p in candidates if p and os.path.isdir(p)), APP_DIR)
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select audio file to play",
            initialdir=initial,
            filetypes=[
                ("WAV audio", "*.wav"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        self.load_selected_audio_file(selected)

    def load_selected_audio_file(self, path):
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception as error:
            messagebox.showerror("Open audio", f"Could not read file:\n{error}")
            return
        if len(audio) < 2:
            messagebox.showwarning("Open audio", "Audio file is too short to play.")
            return

        self.stop_playback(clear_playhead=True, redraw=False)
        self.clear_pending_detection_mark()
        self.loaded_audio_name = os.path.basename(path)
        self.decision_markers = []
        self.show_fixed_waveform(audio, markers=[])
        self.result_text.configure(font=("Segoe UI Semibold", 18))
        self.result_var.set(f"Loaded: {self.loaded_audio_name}")
        self.apply_banner_colors(IDLE_BG, IDLE_FG)
        self.play_status_var.set(self.loaded_audio_name)
        self.log(f"Opened audio for review: {path}")

        if self.mode_var.get() == "Detection" and self.model is not None:
            try:
                threshold = self.get_detect_threshold()
                _results, final, markers = self.detect_audio(audio, threshold)
                self.decision_markers = list(markers or [])
                if isinstance(final, dict) and final.get("dual"):
                    self.set_dual_result_banner(final.get("j1", "NO"), final.get("j2", "NO"))
                else:
                    self.set_result_banner(final)
                if self.fixed_wave_audio is not None:
                    self.showing_fixed_wave = True
                    self.redraw_result_banner()
                if self.decision_markers:
                    marks = ", ".join(
                        f"{m['label']}@{m['t']:.2f}s" for m in self.decision_markers
                    )
                    self.log(f"Review markers: {marks}")
                self.play_status_var.set(
                    f"{self.loaded_audio_name}  ·  reviewed with current model"
                )
            except Exception as error:
                self.log(f"Could not run detection on opened file: {error}")
                self.play_status_var.set(self.loaded_audio_name)

        self.sync_play_button_state()

    def toggle_playback(self):
        if self.playback_active:
            self.stop_playback(clear_playhead=True)
        else:
            self.start_playback()

    def start_playback(self):
        if self.fixed_wave_audio is None or len(self.fixed_wave_audio) < 2:
            return
        if self.recording:
            return
        self.stop_playback(clear_playhead=True, redraw=False)
        audio = np.asarray(self.fixed_wave_audio, dtype=np.float32).reshape(-1)
        self.playback_duration = len(audio) / float(SR)
        self.playback_start_perf = time.perf_counter()
        self.playhead_sec = 0.0
        self.playback_active = True
        self.play_button.configure(text="Stop", state="normal")
        prefix = f"{self.loaded_audio_name}  ·  " if self.loaded_audio_name else ""
        self.play_status_var.set(f"{prefix}Playing...")
        try:
            sd.play(audio, SR, blocking=False)
        except Exception as error:
            self.playback_active = False
            self.playhead_sec = None
            self.play_button.configure(text="Play")
            self.play_status_var.set(self.loaded_audio_name)
            messagebox.showerror("Playback", f"Could not play audio:\n{error}")
            return
        self.redraw_result_banner()
        self.root.after(33, self._tick_playhead)

    def _tick_playhead(self):
        if not self.playback_active:
            return
        elapsed = time.perf_counter() - (self.playback_start_perf or 0.0)
        prefix = f"{self.loaded_audio_name}  ·  " if self.loaded_audio_name else ""
        if elapsed >= self.playback_duration:
            self.playhead_sec = self.playback_duration
            self.redraw_result_banner()
            self.stop_playback(clear_playhead=False)
            self.play_status_var.set(f"{prefix}Playback finished")
            return
        self.playhead_sec = elapsed
        self.play_status_var.set(
            f"{prefix}Playing  {elapsed:.2f}s / {self.playback_duration:.2f}s"
        )
        self.redraw_result_banner()
        self.root.after(33, self._tick_playhead)

    def stop_playback(self, clear_playhead=True, redraw=True):
        was_active = self.playback_active
        self.playback_active = False
        if was_active:
            try:
                sd.stop()
            except Exception:
                pass
        if clear_playhead:
            self.playhead_sec = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Play")
        self.sync_play_button_state()
        if redraw and self.showing_fixed_wave:
            self.redraw_result_banner()

    def _queue_wave_draw(self):
        if self.wave_pending:
            return
        self.wave_pending = True
        self.root.after(33, self._flush_wave_draw)

    def _flush_wave_draw(self):
        self.wave_pending = False
        if self.showing_waveform and self.recording:
            self._draw_live_waveform()

    def _start_record_poll(self):
        self._stop_record_poll()
        self._record_poll_job = self.root.after(50, self._poll_recording_ui)

    def _stop_record_poll(self):
        job = self._record_poll_job
        self._record_poll_job = None
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass

    def _poll_recording_ui(self):
        self._record_poll_job = None
        if not self.recording:
            return
        with self.lock:
            dirty = self.wave_dirty
            self.wave_dirty = False
            samples = int(self.recorded_samples)
            status = self._last_stream_status
        elapsed = samples / float(SR)
        self.result_var.set(f"Recording… {elapsed:.1f}s")
        status_text = f"Recording… {elapsed:.1f}s"
        if status:
            status_text += f"  ·  {status}"
        self.status_var.set(status_text)
        if dirty and self.showing_waveform:
            self._draw_live_waveform()
        self._record_poll_job = self.root.after(50, self._poll_recording_ui)

    def begin_live_waveform(self):
        self.stop_playback(clear_playhead=True, redraw=False)
        with self.lock:
            self.wave_samples = np.zeros(0, dtype=np.float32)
            self.wave_dirty = False
            self.recorded_samples = 0
            self._last_stream_status = ""
        self.showing_waveform = True
        self.showing_fixed_wave = False
        self.fixed_wave_audio = None
        self.decision_markers = []
        self.loaded_audio_name = ""
        self.banner_state = "LIVE"
        self.result_var.set("Recording… 0.0s")
        self.result_text.configure(font=("Segoe UI Semibold", 18))
        self.apply_banner_colors(IDLE_BG, MUTED)
        self.sync_play_button_state()
        self._draw_live_waveform()
        self._start_record_poll()

    def log(self, message):
        self.results.insert(END, message + "\n")
        self.results.see(END)

    def clear_results(self):
        self.results.delete("1.0", END)

    def load_model(self):
        model_path = self.paths()["model_path"]
        try:
            self.model = joblib.load(model_path)
            self.session_status_var.set(f"Model ready · {model_path}")
            self.status_var.set("Ready")
        except Exception:
            self.model = None
            self.session_status_var.set(
                f"No model yet for this session · capture training audio, then Train model"
            )

    def list_input_devices(self):
        devices = []
        try:
            host_devices = sd.query_devices()
        except Exception:
            return devices

        for index, device in enumerate(host_devices):
            if device["max_input_channels"] > 0:
                devices.append((index, f"{index}: {device['name']}", device))
        return devices

    def refresh_devices(self):
        self.input_devices = self.list_input_devices()
        names = [name for _, name, _ in self.input_devices]
        self.device_combo["values"] = names

        if not names:
            self.device_var.set("")
            self.mic_var.set("Microphone: Not Connected")
            return

        default_index = None
        try:
            default_index = sd.default.device[0]
        except Exception:
            default_index = None

        selected = names[0]
        for index, name, _ in self.input_devices:
            if index == default_index:
                selected = name
                break

        self.device_var.set(selected)
        self.mic_var.set("Microphone: " + selected.split(": ", 1)[-1] + " | Connected")

    def selected_device_index(self):
        selected = self.device_var.get()
        for index, name, _ in self.input_devices:
            if name == selected:
                return index
        return None

    def teaching_batch_path(self):
        return os.path.join(self.paths()["root"], TEACHING_BATCH_FILE)

    def daily_counter_path(self):
        return os.path.join(self.paths()["root"], DAILY_COUNTER_FILE)

    def get_teaching_batch_start_ts(self):
        path = self.teaching_batch_path()
        if not os.path.exists(path):
            return 0.0
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            return float(data.get("started_ts", 0.0))
        except Exception:
            return 0.0

    def get_daily_counter_start_ts(self):
        path = self.daily_counter_path()
        if not os.path.exists(path):
            return 0.0
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            return float(data.get("started_ts", 0.0))
        except Exception:
            return 0.0

    def reset_teaching_batch(self, silent=False):
        ensure_session_dirs(self.session_id)
        started = datetime.now()
        payload = {
            "session": self.session_id,
            "started_at": started.isoformat(timespec="seconds"),
            "started_ts": started.timestamp(),
        }
        with open(self.teaching_batch_path(), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.refresh_counts()
        if not silent:
            self.log(
                f"Teaching counter reset for {self.session_var.get()} "
                f"at {payload['started_at']}. Older files are kept for training."
            )
            self.status_var.set("Teaching counter reset")

    def reset_daily_counter(self):
        ensure_session_dirs(self.session_id)
        started = datetime.now()
        payload = {
            "session": self.session_id,
            "started_at": started.isoformat(timespec="seconds"),
            "started_ts": started.timestamp(),
        }
        with open(self.daily_counter_path(), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.refresh_daily_counter()
        self.log(
            f"Daily counter reset for {self.session_var.get()} "
            f"at {payload['started_at']}. Detection logs are kept on disk."
        )
        self.status_var.set("Daily counter reset")

    def ensure_teaching_batch(self):
        if not os.path.exists(self.teaching_batch_path()):
            # First use: count all existing files until user resets.
            payload = {
                "session": self.session_id,
                "started_at": datetime.fromtimestamp(0).isoformat(timespec="seconds"),
                "started_ts": 0.0,
            }
            with open(self.teaching_batch_path(), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

    def ensure_daily_counter(self):
        if not os.path.exists(self.daily_counter_path()):
            payload = {
                "session": self.session_id,
                "started_at": datetime.fromtimestamp(0).isoformat(timespec="seconds"),
                "started_ts": 0.0,
            }
            with open(self.daily_counter_path(), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

    def captured_count(self, label, current_batch_only=True):
        folder = self.folder_for_label(label)
        prefix = CAPTURE_PREFIX[label]
        if not os.path.isdir(folder):
            return 0

        start_ts = self.get_teaching_batch_start_ts() if current_batch_only else 0.0
        count = 0
        for name in os.listdir(folder):
            if not name.lower().endswith(".wav"):
                continue
            if not name.lower().startswith(prefix):
                continue
            path = os.path.join(folder, name)
            if current_batch_only and os.path.getmtime(path) < start_ts:
                continue
            count += 1
        return count

    def capture_status(self, label, count):
        if self.is_dual_session():
            names = {
                "No click": "No click",
                "One click": "One click",
                "Two click": "Two click",
            }
        else:
            names = {
                "Click": "Fitted",
                "No click": "Not fitted",
            }
        name = names.get(label, label)
        return f"{name}: {count} this teaching batch (unlimited)"

    def refresh_counts(self):
        if self.is_dual_session():
            self.click_count_var.set(
                self.capture_status("Two click", self.captured_count("Two click"))
            )
            self.mid_count_var.set(
                self.capture_status("One click", self.captured_count("One click"))
            )
            self.noclick_count_var.set(
                self.capture_status("No click", self.captured_count("No click"))
            )
        else:
            self.click_count_var.set(
                self.capture_status("Click", self.captured_count("Click"))
            )
            self.mid_count_var.set("")
            self.noclick_count_var.set(
                self.capture_status("No click", self.captured_count("No click"))
            )

    def on_session_change(self):
        previous = self._prev_session_id
        self.session_id = self.current_session_id()
        ensure_session_dirs(self.session_id)
        self.ensure_teaching_batch()
        self.ensure_daily_counter()

        # Reset the teaching counter when switching connector teaching sessions.
        if previous is not None and previous != self.session_id:
            self.reset_teaching_batch(silent=True)
            self.log(
                f"Switched teaching session to {self.session_var.get()}. "
                "Capture counter reset to 0."
            )

        self._prev_session_id = self.session_id
        self.clear_pending_detection_mark()
        self.refresh_counts()
        self.refresh_daily_counter()
        self.load_model()
        self.set_result_banner("IDLE")
        self.status_var.set(f"Session: {self.session_var.get()}")
        self.on_mode_change()

    def on_mode_change(self):
        self.label_frame.pack_forget()
        self.threshold_frame.pack_forget()
        self.mark_bar.pack_forget()
        self.sync_training_label_options()
        self.sync_mark_button_layout()

        if self.mode_var.get() == "Training":
            self.label_frame.pack(fill=X, pady=(8, 0))
            self.train_button.configure(
                state="disabled" if self.poc_locked else "normal"
            )
            if hasattr(self, "import_train_button"):
                self.import_train_button.configure(
                    state="disabled" if self.poc_locked else "normal"
                )
            if hasattr(self, "open_train_folder_button"):
                self.open_train_folder_button.configure(state="normal")
            self.set_mark_buttons_enabled(False)
            if self.is_dual_session():
                self.help_var.set(
                    "Capture Two click, One click, and No click takes. "
                    "One+Two teach the click sound; No click teaches silence/noise. "
                    "Or Import training WAVs received from site, then Train / Retrain. "
                    "Use New teaching batch to reset the on-screen counter."
                )
            else:
                self.help_var.set(
                    "Capture fitted and not-fitted takes for this session only "
                    f"({self.session_var.get()}). "
                    "Or Import training WAVs received from site, then Train / Retrain. "
                    "Each session keeps its own model and audio folders."
                )
            self.set_result_banner("IDLE")
        else:
            self.threshold_frame.pack(fill=X, pady=(8, 0))
            self.train_button.configure(state="disabled")
            if hasattr(self, "import_train_button"):
                self.import_train_button.configure(state="disabled")
            self.mark_bar.pack(fill=X, pady=(0, 10), before=self.counter_bar)
            self.sync_mark_buttons_state()
            if self.is_dual_session():
                self.help_var.set(
                    "J1+J2: one recording. Clicks are counted in time order — "
                    "1st click = J1, 2nd = J2. If wrong, mark Two / One / None Fitted "
                    "to save the clip into the matching training folder."
                )
            else:
                self.help_var.set(
                    "Detection uses the selected session model. "
                    "After each run, use Mark as Fitted / Mark as Not Fitted "
                    "to correct the result and save that audio for training."
                )
        self.refresh_counts()

    def sync_mark_button_layout(self):
        self.mark_single_row.pack_forget()
        self.mark_dual_row.pack_forget()
        if self.is_dual_session():
            self.mark_dual_row.pack(fill=X)
        else:
            self.mark_single_row.pack(fill=X)

    def clear_pending_detection_mark(self):
        self.last_detection_audio = None
        self.last_detection_log = None
        self.last_detection_final = None
        self.last_detection_dual = None
        self.last_detection_marked = False
        self.set_mark_buttons_enabled(False)

    def set_mark_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for name in (
            "mark_fitted_button",
            "mark_not_fitted_button",
            "mark_two_button",
            "mark_one_button",
            "mark_none_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.configure(state=state)

    def sync_mark_buttons_state(self):
        can_mark = (
            not self.poc_locked
            and self.mode_var.get() == "Detection"
            and not self.recording
            and not self.busy.get()
            and self.last_detection_audio
            and not self.last_detection_marked
            and os.path.exists(self.last_detection_audio)
        )
        self.set_mark_buttons_enabled(can_mark)

    def mark_detection(self, operator_label):
        if not self.ensure_poc_allowed("mark detection"):
            return
        if self.mode_var.get() != "Detection":
            return
        if self.last_detection_marked or not self.last_detection_audio:
            return
        if self.recording or self.busy.get():
            return

        dual_labels = {"TWO", "ONE", "NONE"}
        single_labels = {"YES", "NO"}
        if self.is_dual_session():
            if operator_label not in dual_labels:
                return
        elif operator_label not in single_labels:
            return

        if not os.path.exists(self.last_detection_audio):
            messagebox.showerror("Mark failed", "Last detection audio file is missing.")
            self.clear_pending_detection_mark()
            return

        if self.is_dual_session():
            training_map = {
                "TWO": "Two click",
                "ONE": "One click",
                "NONE": "No click",
            }
            training_label = training_map[operator_label]
            dual_banner = {
                "TWO": ("YES", "YES", 2),
                "ONE": ("YES", "NO", 1),
                "NONE": ("NO", "NO", 0),
            }
            kind_map = {
                "TWO": "Two fitted",
                "ONE": "One fitted",
                "NONE": "None fitted",
            }
        else:
            training_label = "Click" if operator_label == "YES" else "No click"
            kind_map = {"YES": "Fitted", "NO": "Not fitted"}

        folder = self.folder_for_label(training_label)
        prefix = CAPTURE_PREFIX[training_label].rstrip("_")
        dest_path, _ = self.unique_stamp_paths(folder, prefix, "wav")

        try:
            shutil.copy2(self.last_detection_audio, dest_path)
        except Exception as error:
            messagebox.showerror("Mark failed", f"Could not copy audio to training:\n{error}")
            return

        if self.last_detection_log and os.path.exists(self.last_detection_log):
            try:
                with open(self.last_detection_log, "a", encoding="utf-8") as handle:
                    handle.write(
                        f"operator_label: {operator_label}\n"
                        f"operator_training_label: {training_label}\n"
                        f"operator_marked_at: {datetime.now().isoformat(timespec='seconds')}\n"
                        f"operator_training_file: {os.path.abspath(dest_path)}\n"
                    )
            except Exception as error:
                self.log(f"Warning: could not update detection log ({error})")

        self.last_detection_marked = True
        self.set_mark_buttons_enabled(False)

        if self.is_dual_session():
            j1, j2, click_count = dual_banner[operator_label]
            self.last_detection_dual = {
                "dual": True,
                "j1": j1,
                "j2": j2,
                "click_count": click_count,
            }
            self.set_dual_result_banner(j1, j2)
            if self.fixed_wave_audio is not None:
                self.showing_fixed_wave = True
                self.redraw_result_banner()
        else:
            self.set_result_banner(operator_label)

        self.refresh_counts()
        self.refresh_daily_counter()

        kind = kind_map[operator_label]
        if self.is_dual_session() and isinstance(self.last_detection_final, dict):
            model_kind = self.format_dual_banner_text(
                self.last_detection_final.get("j1", "NO"),
                self.last_detection_final.get("j2", "NO"),
            )
        else:
            model_kind = "Fitted" if self.last_detection_final == "YES" else "Not fitted"
        self.log(
            f"Operator marked {kind} (model: {model_kind}) — saved to training: "
            f"{os.path.basename(dest_path)}"
        )
        self.status_var.set(f"Marked as {kind}")


    def on_quiet_room_toggle(self):
        self.update_effective_threshold_label()
        if self.quiet_room_var.get():
            self.log(
                "Quiet room mode ON — softer clicks accepted "
                f"(effective threshold {self.get_detect_threshold():.2f})."
            )
        else:
            self.log(
                "Quiet room mode OFF — normal threshold "
                f"({self.get_detect_threshold():.2f})."
            )

    def update_effective_threshold_label(self):
        try:
            base = float(self.threshold_var.get())
        except (TypeError, ValueError):
            base = DETECT_THRESHOLD_DEFAULT
        base = min(0.99, max(0.50, base))
        if self.quiet_room_var.get():
            effective = max(QUIET_ROOM_THRESHOLD_FLOOR, round(base - QUIET_ROOM_THRESHOLD_OFFSET, 2))
            self.effective_thr_var.set(f"Effective ≥ {effective:.2f}")
        else:
            self.effective_thr_var.set(f"Effective ≥ {base:.2f}")

    def get_detect_threshold(self):
        try:
            value = float(self.threshold_var.get())
        except (TypeError, ValueError):
            value = DETECT_THRESHOLD_DEFAULT
        value = min(0.99, max(0.50, value))
        self.threshold_var.set(round(value, 2))
        if self.quiet_room_var.get():
            return round(max(QUIET_ROOM_THRESHOLD_FLOOR, value - QUIET_ROOM_THRESHOLD_OFFSET), 2)
        return value

    def get_event_min_score(self):
        if self.quiet_room_var.get():
            return QUIET_ROOM_EVENT_MIN_SCORE
        return EVENT_MIN_SCORE

    def unique_stamp_paths(self, folder, prefix, extension):
        os.makedirs(folder, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"{prefix}_{stamp}.{extension}")
        suffix = 2
        while os.path.exists(path):
            path = os.path.join(folder, f"{prefix}_{stamp}_{suffix}.{extension}")
            suffix += 1
        return path, stamp

    def write_wav(self, path, audio):
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        out = audio
        if peak > 1.0:
            out = audio / peak
        sf.write(path, out, SR, subtype="PCM_16")
        return len(audio) / SR

    def save_detection_run(self, audio, results, final, threshold):
        paths = self.paths()
        os.makedirs(paths["audio_dir"], exist_ok=True)
        os.makedirs(paths["log_dir"], exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(paths["audio_dir"], f"detect_{stamp}.wav")
        log_path = os.path.join(paths["log_dir"], f"detect_{stamp}.log")
        suffix = 2
        while os.path.exists(audio_path) or os.path.exists(log_path):
            audio_path = os.path.join(paths["audio_dir"], f"detect_{stamp}_{suffix}.wav")
            log_path = os.path.join(paths["log_dir"], f"detect_{stamp}_{suffix}.log")
            suffix += 1

        duration = self.write_wav(audio_path, audio)
        lines = [
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            "mode: Detection",
            f"session: {self.session_id}",
            f"session_name: {self.session_var.get()}",
            f"microphone: {self.device_var.get()}",
            f"sample_rate: {SR}",
            f"detect_threshold: {threshold}",
            f"quiet_room_mode: {'yes' if self.quiet_room_var.get() else 'no'}",
            f"ui_threshold: {float(self.threshold_var.get()):.2f}",
            f"duration_sec: {duration:.3f}",
            f"audio_file: {os.path.abspath(audio_path)}",
            f"model: {os.path.abspath(self.paths()['model_path'])}",
        ]

        if isinstance(final, dict) and final.get("dual"):
            j1 = final.get("j1", "NO")
            j2 = final.get("j2", "NO")
            click_count = int(final.get("click_count", 0))
            if j1 == "YES" and j2 == "YES":
                summary = "BOTH FITTED"
                overall = "YES"
            elif j1 == "YES" or j2 == "YES":
                summary = "ONE FITTED"
                overall = "PARTIAL"
            else:
                summary = "NONE FITTED"
                overall = "NO"
            lines.extend([
                f"detection_mode: dual_j1_j2",
                f"click_count: {click_count}",
                f"j1_result: {j1}",
                f"j2_result: {j2}",
                f"final_result: {overall}",
                f"final_message: {summary}",
                f"final_display: {self.format_dual_banner_text(j1, j2)}",
                "assignment_rule: 1st_click=J1, 2nd_click=J2",
            ])
        else:
            lines.extend([
                "detection_mode: single",
                f"final_result: {final}",
                "final_message: "
                f"{'CONNECTOR FITTED' if final == 'YES' else 'CONNECTOR NOT FITTED'}",
            ])

        lines.extend(["", "window_results:"])
        if results:
            lines.extend(results)
        else:
            lines.append("Recording too short for detection.")

        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return audio_path, log_path, duration

    def audio_callback(self, indata, frames, time_info, status):
        # Keep this callback tiny and never touch Tk from the audio thread.
        # Calling root.after here was flooding the UI and aborting captures early.
        if status:
            self._last_stream_status = str(status)
        with self.lock:
            if not self.recording:
                return
            self.audio_chunks.append(indata.copy())
            flat = indata.reshape(-1).astype(np.float32)
            self.recorded_samples += int(flat.shape[0])
            if len(self.wave_samples) == 0:
                self.wave_samples = flat.copy()
            else:
                self.wave_samples = np.concatenate((self.wave_samples, flat))
                if len(self.wave_samples) > WAVE_DISPLAY_SAMPLES:
                    self.wave_samples = self.wave_samples[-WAVE_DISPLAY_SAMPLES:].copy()
            self.wave_dirty = True

    def start_recording(self):
        if self.recording or self.busy.get():
            return
        if not self.ensure_poc_allowed("record"):
            return

        device_index = self.selected_device_index()
        if device_index is None:
            messagebox.showerror("Microphone", "No input microphone selected.")
            return

        if self.mode_var.get() == "Detection" and self.model is None:
            messagebox.showerror(
                "Model missing",
                "No trained model for this session yet.\n"
                "Switch to Training, capture samples, then click Train model."
            )
            return

        self.clear_results()
        self.clear_pending_detection_mark()
        with self.lock:
            self.audio_chunks = []
            self.recorded_samples = 0
            self.wave_dirty = False
            self._last_stream_status = ""
        self.recording = True
        self.busy.set(True)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Recording… 0.0s")
        self.begin_live_waveform()

        try:
            self.stream = sd.InputStream(
                samplerate=SR,
                channels=CHANNELS,
                dtype="float32",
                device=device_index,
                blocksize=1024,
                latency="high",
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as error:
            self.recording = False
            self._stop_record_poll()
            self.busy.set(False)
            self.showing_waveform = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_var.set("Ready")
            self.set_result_banner("IDLE")
            messagebox.showerror("Recording error", str(error))

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self._stop_record_poll()
        self.showing_waveform = False
        self.stop_button.configure(state="disabled")
        self.status_var.set("Processing...")
        self.result_var.set("Processing...")
        self.result_text.configure(font=("Segoe UI Semibold", 18))
        self.apply_banner_colors(IDLE_BG, IDLE_FG)

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None

        with self.lock:
            chunks = list(self.audio_chunks)
            self.audio_chunks = []
            sample_count = int(self.recorded_samples)

        if not chunks:
            self.set_result_banner("IDLE")
            self.finish_idle("No audio captured")
            messagebox.showwarning("Recording", "No audio was captured.")
            return

        audio = np.concatenate(chunks, axis=0).reshape(-1).astype(np.float32)
        duration = len(audio) / float(SR)
        self.log(
            f"Captured {duration:.2f}s "
            f"({len(audio)} samples from {len(chunks)} blocks, "
            f"counter={sample_count})"
        )
        if duration < 1.5:
            self.log(
                "Warning: capture is very short. Keep recording through the full "
                "connector action, then press STOP."
            )
        self.decision_markers = []
        self.loaded_audio_name = "Last recording"
        self.show_fixed_waveform(audio, markers=[])
        self.play_status_var.set(f"Last recording  ·  {duration:.2f}s")
        mode = self.mode_var.get()
        label = self.label_var.get()
        threshold = self.get_detect_threshold()
        threading.Thread(
            target=self.process_recording,
            args=(audio, mode, label, threshold),
            daemon=True
        ).start()

    def process_recording(self, audio, mode, label, threshold):
        try:
            if mode == "Training":
                path, duration = self.save_training_recording(audio, label)
                self.root.after(0, self.on_training_saved, path, duration, label)
            else:
                results, final, markers = self.detect_audio(audio, threshold)
                audio_path, log_path, duration = self.save_detection_run(
                    audio, results, final, threshold
                )
                self.root.after(
                    0,
                    self.on_detection_done,
                    results,
                    final,
                    markers,
                    audio_path,
                    log_path,
                    duration,
                    threshold
                )
        except Exception as error:
            self.root.after(0, self.on_process_error, str(error))

    def save_training_recording(self, audio, label):
        folder = self.folder_for_label(label)
        prefix = CAPTURE_PREFIX[label].rstrip("_")
        path, _ = self.unique_stamp_paths(folder, prefix, "wav")
        duration = self.write_wav(path, audio)
        return path, duration

    def extract_features(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio, sr=SR, n_mfcc=13, n_fft=512, hop_length=256
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

    def window_score(self, window):
        rms = np.sqrt(np.mean(window ** 2))
        peak = np.max(np.abs(window))
        crest = peak / (rms + 1e-8)
        k = kurtosis(window)
        return float(crest * max(k, 1.0))

    def detect_events(self, audio):
        scores = []
        times = []
        for start in range(0, len(audio) - WINDOW + 1, EVENT_STEP):
            window = audio[start:start + WINDOW]
            scores.append(self.window_score(window))
            times.append(start / SR)
        if len(scores) < 3:
            return []
        scores = np.asarray(scores, dtype=np.float64)
        min_score = self.get_event_min_score()
        threshold = max(float(np.percentile(scores, EVENT_PERCENTILE)), min_score)
        peaks = []
        for index in range(1, len(scores) - 1):
            score = scores[index]
            if score >= threshold and score >= scores[index - 1] and score >= scores[index + 1]:
                peaks.append((times[index], score))
        merged = []
        for time_sec, score in peaks:
            if not merged or time_sec - merged[-1][0] > EVENT_MERGE_SEC:
                merged.append([time_sec, score])
            elif score > merged[-1][1]:
                merged[-1] = [time_sec, score]
        return [(time_sec, score) for time_sec, score in merged]

    def detection_region_start(self, audio):
        """
        Anchor on the last screw-like event only.
        Unscrews and earlier spikes are ignored; we do not assume exactly 4 screws.
        Scan from slightly before that last event through the end so the
        connector seating sound is included whether it is the last event
        itself or comes just after it.
        """
        duration = len(audio) / SR
        events = self.detect_events(audio)

        if len(events) >= 1:
            last_event_time = events[-1][0]
            region_start = max(0.0, last_event_time - 0.5)
            rule = "from_last_event"
        else:
            region_start = max(0.0, duration * 0.6)
            rule = "fallback_last_40_percent"

        region_start = min(region_start, max(0.0, duration - (WINDOW / SR)))
        return region_start, events, rule

    def count_today_detections(self):
        paths = self.paths()
        log_dir = paths["log_dir"]
        today = datetime.now().strftime("%Y%m%d")
        fitted = 0
        not_fitted = 0
        partial = 0
        total = 0
        start_ts = self.get_daily_counter_start_ts()

        if not os.path.isdir(log_dir):
            return total, fitted, not_fitted, partial

        for name in os.listdir(log_dir):
            if not name.lower().endswith(".log"):
                continue
            if not name.startswith(f"detect_{today}"):
                continue

            path = os.path.join(log_dir, name)
            try:
                if os.path.getmtime(path) < start_ts:
                    continue
            except Exception:
                continue

            total += 1
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
            except Exception:
                continue

            if "operator_label: TWO" in text:
                fitted += 1
            elif "operator_label: ONE" in text:
                partial += 1
            elif "operator_label: NONE" in text:
                not_fitted += 1
            elif "operator_label: YES" in text:
                fitted += 1
            elif "operator_label: NO" in text:
                not_fitted += 1
            elif "final_result: PARTIAL" in text or "final_message: ONE FITTED" in text:
                partial += 1
            elif "final_result: YES" in text or "final_message: BOTH FITTED" in text:
                fitted += 1
            elif "final_result: NO" in text or "final_message: NONE FITTED" in text:
                not_fitted += 1

        return total, fitted, not_fitted, partial

    def refresh_daily_counter(self):
        total, fitted, not_fitted, partial = self.count_today_detections()
        if self.is_dual_session():
            self.daily_count_var.set(
                f"Today ({self.session_var.get()}): {total} detections  ·  "
                f"Both {fitted}  ·  One {partial}  ·  None {not_fitted}"
            )
        else:
            self.daily_count_var.set(
                f"Today ({self.session_var.get()}): {total} detections  ·  "
                f"Fitted {fitted}  ·  Not fitted {not_fitted}"
            )

    def detect_model_click_events(self, audio, threshold):
        """
        J1_J2 seating clicks only (no screw sequence).
        Find local probability peaks above threshold, collapse nearby peaks,
        then keep at most DUAL_MAX_EVENTS of the strongest peaks that are at
        least DUAL_MIN_EVENT_GAP_SEC apart. Time-order them as J1 then J2 so
        weaker handling spikes do not invent extra seats.
        """
        starts = list(range(0, len(audio) - WINDOW + 1, DETECT_STEP))
        if not starts:
            return [], []

        feature_rows = [self.extract_features(audio[start:start + WINDOW]) for start in starts]
        features = np.asarray(feature_rows, dtype=np.float64)
        probabilities = self.model.predict_proba(features)[:, 1]

        window_rows = []
        for start, probability in zip(starts, probabilities):
            probability = float(probability)
            result = "YES" if probability >= threshold else "NO"
            window_rows.append(
                f"Audio: {start / SR:.2f}s | Result: {result} | Probability: {probability:.3f}"
            )

        peak_idxs = []
        n = len(probabilities)
        for i in range(n):
            p = float(probabilities[i])
            if p < threshold:
                continue
            left = float(probabilities[i - 1]) if i > 0 else -1.0
            right = float(probabilities[i + 1]) if i + 1 < n else -1.0
            if p + 1e-9 >= left and p + 1e-9 >= right:
                peak_idxs.append(i)

        # Collapse peaks within a short cluster; keep the strongest.
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

        # Keep strongest peaks first, enforcing min gap; cap at J1+J2.
        # Secondary seat must be near the primary probability so handling
        # spikes just above threshold do not invent a fake second click.
        ranked = sorted(clustered, key=lambda item: item[1], reverse=True)
        chosen = []
        primary_p = ranked[0][1] if ranked else 0.0
        secondary_floor = max(threshold, primary_p * DUAL_SECONDARY_MIN_RATIO)
        for t, p in ranked:
            if any(abs(t - ct) < DUAL_MIN_EVENT_GAP_SEC for ct, _cp in chosen):
                continue
            if chosen and p < secondary_floor:
                continue
            chosen.append((t, p))
            if len(chosen) >= DUAL_MAX_EVENTS:
                break

        events = sorted(chosen, key=lambda item: item[0])
        return events, window_rows

    def markers_from_dual_events(self, events):
        markers = []
        for index, (time_sec, probability) in enumerate(events):
            if index == 0:
                label = "J1"
            elif index == 1:
                label = "J2"
            else:
                label = f"E{index + 1}"
            markers.append({
                "t": float(time_sec),
                "label": label,
                "prob": float(probability),
            })
        return markers

    def markers_from_single_decision(self, starts, probabilities, threshold, final):
        if len(starts) == 0 or len(probabilities) == 0:
            return []
        probs = np.asarray(probabilities, dtype=np.float64)
        if final == "YES":
            yes_idxs = [i for i, p in enumerate(probs) if p >= threshold]
            if not yes_idxs:
                return []
            i = max(yes_idxs, key=lambda idx: probs[idx])
            return [{
                "t": starts[i] / SR,
                "label": "FITTED",
                "prob": float(probs[i]),
            }]
        i = int(np.argmax(probs))
        return [{
            "t": starts[i] / SR,
            "label": "NO",
            "prob": float(probs[i]),
            "negative": True,
        }]

    def detect_audio_dual(self, audio, threshold):
        overall_start = time.perf_counter()
        predict_start = time.perf_counter()
        events, window_rows = self.detect_model_click_events(audio, threshold)
        predict_time = time.perf_counter() - predict_start

        click_count = len(events)
        j1 = "YES" if click_count >= 1 else "NO"
        j2 = "YES" if click_count >= 2 else "NO"
        markers = self.markers_from_dual_events(events)

        results = [
            f"Dual scan | click_events={click_count} | "
            f"min_gap={DUAL_MIN_EVENT_GAP_SEC:.2f}s | "
            f"max_events={DUAL_MAX_EVENTS} | "
            f"rule=top2_strongest+gap,1st=J1,2nd=J2 | "
            f"predict={predict_time:.3f}s | "
            f"total={time.perf_counter() - overall_start:.3f}s"
        ]
        for index, (time_sec, probability) in enumerate(events, start=1):
            label = "J1" if index == 1 else ("J2" if index == 2 else f"extra_{index}")
            results.append(
                f"Click event {index} ({label}): {time_sec:.2f}s | "
                f"Probability: {probability:.3f}"
            )
        if click_count == 0:
            results.append("No click events above threshold.")
        elif click_count == 1:
            results.append("Only one click — counted as J1 FITTED, J2 NOT FITTED.")
        else:
            results.append("Two clicks — J1 FITTED, J2 FITTED.")

        results.append("--- window detail ---")
        results.extend(window_rows)

        final = {
            "dual": True,
            "j1": j1,
            "j2": j2,
            "click_count": click_count,
            "events": events,
        }
        return results, final, markers

    def detect_audio(self, audio, threshold):
        if self.is_dual_session():
            return self.detect_audio_dual(audio, threshold)

        overall_start = time.perf_counter()
        region_start, events, rule = self.detection_region_start(audio)
        starts = [
            start for start in range(0, len(audio) - WINDOW + 1, DETECT_STEP)
            if start / SR >= region_start
        ]
        if not starts:
            all_starts = list(range(0, len(audio) - WINDOW + 1, DETECT_STEP))
            starts = all_starts[-max(1, min(8, len(all_starts))):]
            rule = rule + "+short_tail"

        feature_rows = [self.extract_features(audio[start:start + WINDOW]) for start in starts]
        features = np.asarray(feature_rows, dtype=np.float64)
        predict_start = time.perf_counter()
        probabilities = self.model.predict_proba(features)[:, 1]
        predict_time = time.perf_counter() - predict_start

        results = []
        for start, probability in zip(starts, probabilities):
            probability = float(probability)
            result = "YES" if probability >= threshold else "NO"
            results.append(
                f"Audio: {start / SR:.2f}s | Result: {result} | Probability: {probability:.3f}"
            )

        final = "YES" if any(p >= threshold for p in probabilities) else "NO"
        markers = self.markers_from_single_decision(starts, probabilities, threshold, final)
        total_time = time.perf_counter() - overall_start
        results.insert(
            0,
            f"Scan region: {region_start:.2f}s onward | events={len(events)} | "
            f"rule={rule} | windows={len(starts)} | "
            f"predict={predict_time:.3f}s | total={total_time:.3f}s"
        )
        return results, final, markers

    def on_training_saved(self, path, duration, label):
        self.saved_files.append(path)
        self.refresh_counts()
        kind_names = {
            "Click": "Fitted",
            "No click": "No click",
            "One click": "One click",
            "Two click": "Two click",
        }
        kind = kind_names.get(label, label)
        self.log(f"Saved {os.path.basename(path)} ({duration:.1f}s) as {kind}")
        self.log(path)
        # Keep fixed waveform; only reset the status text.
        self.result_var.set(f"Saved as {kind}")
        self.result_text.configure(font=("Segoe UI Semibold", 18))
        self.apply_banner_colors(IDLE_BG, IDLE_FG)
        self.sync_play_button_state()
        self.redraw_result_banner()
        self.finish_idle("Saved")

    def on_detection_done(self, results, final, markers, audio_path, log_path, duration, threshold):
        self.last_detection_audio = audio_path
        self.last_detection_log = log_path
        self.last_detection_final = final
        self.last_detection_marked = False
        self.loaded_audio_name = os.path.basename(audio_path) if audio_path else "Last recording"
        self.decision_markers = list(markers or [])

        if isinstance(final, dict) and final.get("dual"):
            self.last_detection_dual = final
            self.set_dual_result_banner(final.get("j1", "NO"), final.get("j2", "NO"))
        else:
            self.last_detection_dual = None
            self.set_result_banner(final)

        # Keep the captured waveform visible under the result with decision pointers.
        if self.fixed_wave_audio is not None:
            self.showing_fixed_wave = True
            self.sync_play_button_state()
            self.redraw_result_banner()

        self.log(f"Session: {self.session_var.get()}")
        self.log(f"Threshold: {threshold:.2f}")
        if not results:
            self.log("Recording too short for detection.")
        else:
            self.log("Detection Results")
            for row in results:
                self.log(row)
            if isinstance(final, dict) and final.get("dual"):
                self.log(self.format_dual_banner_text(final.get("j1", "NO"), final.get("j2", "NO")))
                self.log(
                    f"Click count: {final.get('click_count', 0)} "
                    "(1st=J1, 2nd=J2)"
                )
            else:
                self.log("CONNECTOR FITTED" if final == "YES" else "CONNECTOR NOT FITTED")
        if self.decision_markers:
            marks = ", ".join(
                f"{m['label']}@{m['t']:.2f}s" for m in self.decision_markers
            )
            self.log(f"Decision markers: {marks}")
        self.log("")
        self.log(f"Saved audio ({duration:.1f}s): {audio_path}")
        self.log(f"Saved log: {log_path}")
        if self.is_dual_session():
            self.log(
                "Use Mark as Two / One / None Fitted if you need to correct this result."
            )
        else:
            self.log(
                "Use Mark as Fitted / Mark as Not Fitted if you need to correct this result."
            )
        self.refresh_daily_counter()
        self.finish_idle("Done")
        self.sync_mark_buttons_state()
        self.sync_play_button_state()

    def on_process_error(self, message):
        self.log("Error: " + message)
        self.set_result_banner("IDLE")
        self.clear_pending_detection_mark()
        self.finish_idle("Error")
        messagebox.showerror("Processing error", message)

    def training_wav_count(self, label):
        """Count all WAV files in a label folder (includes imported files)."""
        folder = self.folder_for_label(label)
        if not os.path.isdir(folder):
            return 0
        return sum(
            1
            for name in os.listdir(folder)
            if name.lower().endswith(".wav")
        )

    def import_training_wavs_menu(self):
        if self.poc_locked:
            self.ensure_poc_allowed("import training files")
            return
        if self.mode_var.get() != "Training":
            messagebox.showinfo(
                "Training mode",
                "Switch to Training mode first, then import WAVs for the selected session.",
            )
            return

        if self.is_dual_session():
            choices = [
                ("Two click (both fitted)", "Two click"),
                ("One click (one fitted)", "One click"),
                ("No click", "No click"),
            ]
        else:
            choices = [
                ("Fitted / Click", "Click"),
                ("Not fitted / No click", "No click"),
            ]

        win = Toplevel(self.root)
        win.title("Import training WAVs")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        Label(
            win,
            text=f"Import into session: {self.session_var.get()}",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 12),
        ).pack(anchor=W, padx=16, pady=(16, 8))
        Label(
            win,
            text="Choose which label folder to copy the received WAV files into.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor=W, padx=16, pady=(0, 10))

        def pick(label):
            win.destroy()
            self.import_training_wavs(label)

        for title, label in choices:
            Button(
                win,
                text=title,
                command=lambda lab=label: pick(lab),
                bg=ACCENT,
                fg="#FFFFFF",
                activebackground="#4A7FFF",
                activeforeground="#FFFFFF",
                relief="flat",
                padx=14,
                pady=7,
                cursor="hand2",
            ).pack(fill=X, padx=16, pady=4)
        Button(
            win,
            text="Cancel",
            command=win.destroy,
            bg=CARD_ALT,
            fg=TEXT,
            relief="flat",
            padx=14,
            pady=7,
        ).pack(fill=X, padx=16, pady=(8, 16))

    def import_training_wavs(self, label):
        files = filedialog.askopenfilenames(
            title=f"Select WAV files for {label} ({self.session_var.get()})",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not files:
            return

        dest_dir = self.folder_for_label(label)
        os.makedirs(dest_dir, exist_ok=True)
        prefix = CAPTURE_PREFIX[label]
        copied = 0
        for src in files:
            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                base = os.path.splitext(os.path.basename(src))[0]
                dest_name = f"{prefix}import_{stamp}_{base}.wav"
                dest = os.path.join(dest_dir, dest_name)
                shutil.copy2(src, dest)
                copied += 1
            except Exception as error:
                self.log(f"Import failed for {src}: {error}")

        self.refresh_counts()
        self.log(
            f"Imported {copied} WAV(s) as '{label}' into {self.session_var.get()}."
        )
        self.log(dest_dir)
        messagebox.showinfo(
            "Import complete",
            f"Copied {copied} file(s) into:\n{dest_dir}\n\n"
            "Click Train / Retrain model when ready.",
        )

    def open_training_folder(self):
        paths = self.paths()
        root = paths["root"]
        train_root = os.path.join(root, "training_audio")
        os.makedirs(train_root, exist_ok=True)
        try:
            os.startfile(train_root)
        except Exception as error:
            messagebox.showerror("Open folder", str(error))

    def train_current_session(self):
        if self.busy.get() or self.recording:
            return
        if not self.ensure_poc_allowed("train"):
            return

        if self.is_dual_session():
            one_n = self.training_wav_count("One click")
            two_n = self.training_wav_count("Two click")
            none_n = self.training_wav_count("No click")
            positive_n = one_n + two_n
            if positive_n < 2 or none_n < 2:
                messagebox.showwarning(
                    "Not enough data",
                    "Need at least 2 click recordings (One and/or Two click) "
                    "and 2 No click recordings in this session before training.\n\n"
                    f"Current: Two click={two_n}, One click={one_n}, No click={none_n}\n\n"
                    "Use Import training WAVs… if files were received separately."
                )
                return
        else:
            click_n = self.training_wav_count("Click")
            none_n = self.training_wav_count("No click")
            if click_n < 2 or none_n < 2:
                messagebox.showwarning(
                    "Not enough data",
                    "Need at least 2 fitted and 2 not-fitted recordings in this "
                    f"session ({self.session_var.get()}) before training.\n\n"
                    f"Current: Fitted={click_n}, Not fitted={none_n}\n\n"
                    "Use Import training WAVs… if files were received separately."
                )
                return

        model_exists = os.path.exists(self.paths()["model_path"])
        action = "Retrain" if model_exists else "Train"
        if not messagebox.askyesno(
            f"{action} model",
            f"{action} model for {self.session_var.get()}?\n\n"
            "This uses all WAV files currently in this session's training folders "
            "and overwrites this session's model only "
            "(other sessions are not changed).",
        ):
            return

        self.busy.set(True)
        self.status_var.set(f"{action}ing model...")
        self.log(f"{action}ing model for session {self.session_var.get()}...")
        session_id = self.session_id

        def worker():
            try:
                from train_model import train_session
                path = train_session(session_id)
                self.root.after(0, self.on_train_done, path, None)
            except Exception as error:
                self.root.after(0, self.on_train_done, None, str(error))

        threading.Thread(target=worker, daemon=True).start()

    def on_train_done(self, path, error):
        self.busy.set(False)
        if error:
            self.status_var.set("Training failed")
            self.log("Training error: " + error)
            messagebox.showerror("Training failed", error)
            return
        self.load_model()
        self.log("Model saved: " + path)
        self.status_var.set("Model trained")
        messagebox.showinfo(
            "Training complete",
            f"Model saved for {self.session_var.get()} only.\n{path}",
        )

    def finish_idle(self, status):
        self.busy.set(False)
        self.start_button.configure(
            state="disabled" if self.poc_locked else "normal"
        )
        self.stop_button.configure(state="disabled")
        self.status_var.set(status)
        self.sync_play_button_state()

    def on_close(self):
        self.recording = False
        self._stop_record_poll()
        self.stop_playback(clear_playhead=True, redraw=False)
        try:
            self.root.unbind_all("<MouseWheel>")
        except Exception:
            pass
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.root.destroy()


def resolve_desktop_dir():
    candidates = []
    try:
        import ctypes.wintypes
        from ctypes import windll, byref, create_unicode_buffer

        CSIDL_DESKTOP = 0
        buf = create_unicode_buffer(260)
        if windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOP, None, 0, buf) == 0:
            candidates.append(buf.value)
    except Exception:
        pass

    home = os.path.expanduser("~")
    userprofile = os.environ.get("USERPROFILE", home)
    onedrive = os.environ.get("OneDrive", os.path.join(userprofile, "OneDrive"))
    candidates.extend(
        [
            os.path.join(onedrive, "Desktop"),
            os.path.join(userprofile, "OneDrive", "Desktop"),
            os.path.join(home, "Desktop"),
            os.path.join(userprofile, "Desktop"),
        ]
    )
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return os.path.join(userprofile, "Desktop")


def create_desktop_shortcut():
    desktop = resolve_desktop_dir()
    os.makedirs(desktop, exist_ok=True)

    # Keep a single shortcut only.
    shortcut = os.path.join(desktop, "Wire Fit Detection.lnk")
    for stale_name in ("Wire Fit.lnk", "WireFit.lnk", "wire fit.lnk"):
        stale = os.path.join(desktop, stale_name)
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except Exception:
                pass

    icon = ICON_PATH if os.path.exists(ICON_PATH) else ""

    # Packaged build: shortcut points at the exe (no Python / source files).
    if getattr(sys, "frozen", False):
        target = sys.executable
        arguments = ""
        working = APP_DIR
    else:
        bat = os.path.join(APP_DIR, "start_wire_fit.bat")
        python_candidates = [
            r"C:\Python314\python.exe",
            os.path.join(os.environ.get("LocalAppData", ""), "Programs", "Python", "Python314", "python.exe"),
            os.path.join(os.environ.get("LocalAppData", ""), "Programs", "Python", "Python313", "python.exe"),
            os.path.join(os.environ.get("LocalAppData", ""), "Programs", "Python", "Python312", "python.exe"),
            shutil.which("python") or "",
        ]
        python_exe = next((p for p in python_candidates if p and os.path.isfile(p)), "python")

        with open(bat, "w", encoding="utf-8") as handle:
            handle.write("@echo off\r\n")
            handle.write("setlocal EnableExtensions\r\n")
            handle.write('cd /d "%~dp0"\r\n')
            handle.write(f'set "APP_PY=%~dp0native_app.py"\r\n')
            handle.write('if not exist "%APP_PY%" (\r\n')
            handle.write("  echo Could not find native_app.py\r\n")
            handle.write("  pause\r\n")
            handle.write("  exit /b 1\r\n")
            handle.write(")\r\n")
            handle.write(f'set "PYTHON_EXE={python_exe}"\r\n')
            handle.write('if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"\r\n')
            handle.write('"%PYTHON_EXE%" "%APP_PY%"\r\n')
            handle.write("if errorlevel 1 (\r\n")
            handle.write("  echo.\r\n")
            handle.write("  echo Wire Fit Detection failed to start.\r\n")
            handle.write("  pause\r\n")
            handle.write("  exit /b 1\r\n")
            handle.write(")\r\n")

        target = python_exe if os.path.isfile(python_exe) else bat
        arguments = f'"{os.path.join(APP_DIR, "native_app.py")}"' if os.path.isfile(python_exe) else ""
        working = APP_DIR

    ps = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{shortcut.replace("'", "''")}')
$Shortcut.TargetPath = '{target.replace("'", "''")}'
$Shortcut.Arguments = '{arguments.replace("'", "''")}'
$Shortcut.WorkingDirectory = '{working.replace("'", "''")}'
$Shortcut.IconLocation = '{icon.replace("'", "''")}'
$Shortcut.Description = 'Wire Fit Detection by Brown Box Innovations'
$Shortcut.WindowStyle = 1
$Shortcut.Save()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        check=False,
        capture_output=True,
        text=True,
    )
    return shortcut


def main():
    for session_id, _ in SESSION_OPTIONS:
        ensure_session_dirs(session_id)

    try:
        create_desktop_shortcut()
    except Exception:
        pass

    root = Tk()
    NativeWireFitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
