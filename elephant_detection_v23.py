# -*- coding: utf-8 -*-*

"""

Elephant Detection with MSP430 LED Control

      + Independent Analytics & Prediction window

      + Fully resizable (ribbon-style) windows

      + Proportionally scaled, card-style analytics

      + 30-min P(presence) window

      + Hour-of-day chart from log detection data

      + History line 2 = past + present combined            (v24)

Author: Ami

Date: Sep 26, 2026

"""

import asyncio
import cv2
import ctypes

import csv

import json

import math

import os

import statistics

import sys

import threading

import time

from collections import deque, defaultdict

from datetime import datetime, timedelta

import numpy as np

import serial

import serial.tools.list_ports

from ultralytics import YOLO

# ============================================================*

#  CONFIGURATION*

# ============================================================*

# Raspberry Pi 5 CPU-optimized default. Ultralytics downloads this model on first run.
# For more accuracy (at lower FPS), change to yolov8s.pt.
MODEL_PATH          = "yolov8n.pt"
INFERENCE_SIZE      = 416       # try 320 for speed, 512 for more small/distant elephants
MAX_DETECTIONS      = 10
CAMERA_WIDTH        = 640
CAMERA_HEIGHT       = 480
ANALYTICS_FPS       = 2.0       # expensive dashboard redraw rate; camera remains responsive

CONF_THRESHOLD      = 0.50

CAMERA_INDICES      = (0,1,2)          # force index 1 (external/USB camera)*

LOG_DIR             = "elephant_detections"

TEXT_LOG            = os.path.join(LOG_DIR, "detections.log")

CSV_LOG             = os.path.join(LOG_DIR, "events.csv")

REPORT_TXT          = os.path.join(LOG_DIR, "analytics_report.txt")

STATE_JSON          = os.path.join(LOG_DIR, "analytics_state.json")

PLOT_PNG            = os.path.join(LOG_DIR, "session_summary.png")

REQUIRE_MSP430           = True

SEND_PREDICTIVE_CAUTION  = False

SERIAL_BAUD              = 9600

# --- ESP32 BLE train control ---
TRAIN_BLE_DEVICE_NAME    = "ElephantTrainESP32"
TRAIN_BLE_SERVICE_UUID   = "7d2a0001-8e5f-4f3a-9b1c-6d7e8f901234"
TRAIN_BLE_CHAR_UUID      = "7d2a0002-8e5f-4f3a-9b1c-6d7e8f901234"
TRAIN_BLE_HEARTBEAT_S    = 0.5
TRAIN_CLEAR_BEFORE_RESUME_S = 2.0  # avoid RUN pulses from one-frame detection flicker
TRAIN_BLE_ENABLE         = True

# --- Analytics tuning ---*

MIN_DETECTION_S     = 0.5

SAMPLE_INTERVAL_S   = 0.25

RATE_WINDOW_S       = 300

CONF_WINDOW_N       = 200

PROB_HISTORY_N      = 1800          # 30 min at 1 sample/sec*

DECAY_TAU_S         = 120.0

PRIOR_SMOOTHING_S   = 60.0

W_RECENCY, W_RATE, W_PRIOR = 0.45, 0.25, 0.30

REPORT_INTERVAL_S   = 5.0

RUN_ID = time.strftime("%Y%m%d-%H%M%S")

# ============================================================*

#  WINDOWS*

# ============================================================*

CAMERA_WINDOW        = "Elephant Detection Feed"

ANALYTICS_WINDOW     = "Elephant Detection Analytics"

CAMERA_DEFAULT_SIZE  = (800, 600)

ANALYTICS_DEFAULT_SZ = (1200, 900)

# ============================================================*

#  ANALYTICS SCALING REFERENCE*

#     s = min(W/900, H/700), clamped to [0.55, 2.00]*

# ============================================================*

AN_BASE_W, AN_BASE_H = 900, 700

AN_S_MIN,  AN_S_MAX  = 0.55, 2.00

AN_PAD          = 14

AN_TITLE_H      = 34

AN_FOOTER_H     = 18

AN_CARD_PAD     = 12

AN_ROW_H        = 20

AN_SEC_H        = 18

AN_GAP          = 8

AN_CHART_MIN_H  = 70

AN_F_TITLE      = 0.60

AN_F_SECTION    = 0.48

AN_F_BODY       = 0.44

AN_F_SMALL      = 0.36

AN_F_TICK       = 0.32

# ============================================================*

#  COLORS (BGR)*

# ============================================================*

C_BG            = (24, 24, 24)

C_TITLE_BG      = (20, 20, 20)

C_CARD_BG       = (34, 34, 34)

C_CARD_BORDER   = (56, 56, 56)

C_DIVIDER       = (70, 70, 70)

C_ACCENT        = (170, 150, 100)

C_LABEL         = (140, 140, 140)

C_VALUE         = (225, 225, 225)

C_VALUE_DIM     = (185, 185, 185)

C_CHART_BG      = (30, 30, 30)

C_CHART_GRID    = (52, 52, 52)

# hour-chart bar colors*

C_HOUR_OBS      = (85, 85, 85)      # gray — observed time*

C_HOUR_OBS_CUR  = (120, 120, 120)   # lighter gray for current hour*

C_HOUR_DET      = (90, 60, 170)     # red — detected time*

C_HOUR_DET_CUR  = (60, 60, 220)     # brighter red for current hour*

RISK_BGR = {

    "LOW":      (110, 200, 110),

    "MEDIUM":   (0, 205, 235),

    "HIGH":     (0, 140, 240),

    "CRITICAL": (60, 60, 220),

}

# ============================================================*

#  HELPERS*

# ============================================================*

def iso(epoch):

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))

def hms(seconds):

    seconds = int(max(0, seconds))

    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

def fmt_dur_short(sec):

    """Compact duration for chart axis: 45s, 12m, 3h20, ..."""

    sec = int(max(0, sec))

    if sec < 60:   return f"{sec}s"

    if sec < 3600: return f"{sec // 60}m"

    h = sec // 3600

    m = (sec % 3600) // 60

    return f"{h}h{m:02d}" if m else f"{h}h"

def split_by_hour(start_epoch, end_epoch):

    cur = start_epoch

    while cur < end_epoch - 1e-9:

        dt = datetime.fromtimestamp(cur)

        nxt = (dt.replace(minute=0, second=0, microsecond=0)

               + timedelta(hours=1)).timestamp()

        chunk_end = min(end_epoch, nxt)

        yield dt.hour, chunk_end - cur

        cur = chunk_end

def get_window_size(name, default):

    try:

        rect = cv2.getWindowImageRect(name)

        if rect and len(rect) == 4:

            _, _, w, h = rect

            if w > 1 and h > 1:

                return int(w), int(h)

    except Exception:

        pass

    return default


def make_borderless_split_windows(camera_title, analytics_title):
    """Remove native title/borders and fit camera + analytics to the desktop.

    Windows-only helper. OpenCV keeps the normal-window fallback on other OSes.
    The two client areas are positioned edge-to-edge with no unused white bands.
    """
    if os.name != "nt":
        return

    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    left_w = screen_w // 2
    right_w = screen_w - left_w

    # Window style flags from WinUser.h; omit caption, sizing frame and borders.
    GWL_STYLE = -16
    WS_CAPTION = 0x00C00000
    WS_THICKFRAME = 0x00040000
    WS_BORDER = 0x00800000
    WS_DLGFRAME = 0x00400000
    WS_MINIMIZEBOX = 0x00020000
    WS_MAXIMIZEBOX = 0x00010000
    SWP_NOZORDER = 0x0004
    SWP_FRAMECHANGED = 0x0020
    SWP_SHOWWINDOW = 0x0040

    def style_and_place(title, x, width):
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        style = get_long(hwnd, GWL_STYLE)
        style &= ~(WS_CAPTION | WS_THICKFRAME | WS_BORDER | WS_DLGFRAME |
                   WS_MINIMIZEBOX | WS_MAXIMIZEBOX)
        set_long(hwnd, GWL_STYLE, style)
        user32.SetWindowPos(hwnd, 0, x, 0, width, screen_h,
                            SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW)
        return True

    # Retry briefly because some OpenCV backends register their HWND asynchronously.
    for _ in range(10):
        cam_ok = style_and_place(camera_title, 0, left_w)
        ana_ok = style_and_place(analytics_title, left_w, right_w)
        if cam_ok and ana_ok:
            break
        time.sleep(0.1)

# ============================================================*

#  HISTORY LOADING*

# ============================================================*

def load_history(csv_path):

    hist = {

        "n_runs": 0,

        "run_intervals": [],

        "det_intervals": [],

        "hour_observed": defaultdict(float),

        "hour_detected": defaultdict(float),

    }

    if not os.path.exists(csv_path):

        return hist

    try:

        with open(csv_path, newline="", encoding="utf-8") as f:

            for row in csv.DictReader(f):

                ev = (row.get("event") or "").strip().upper()

                try:

                    s = float(row["start_epoch"]); e = float(row["end_epoch"])

                except (KeyError, TypeError, ValueError):

                    continue

                if e <= s:

                    continue

                if ev == "RUN":

                    hist["n_runs"] += 1

                    hist["run_intervals"].append((s, e))

                    for h, secs in split_by_hour(s, e):

                        hist["hour_observed"][h] += secs

                elif ev == "DETECTION":

                    hist["det_intervals"].append((s, e))

                    for h, secs in split_by_hour(s, e):

                        hist["hour_detected"][h] += secs

    except Exception as exc:

        print(f"⚠️  Could not parse history ({exc}); starting fresh.")

    return hist

def derive_gaps(hist):

    gaps = []

    for rs, re in hist["run_intervals"]:

        dets = sorted(d for d in hist["det_intervals"] if d[0] >= rs and d[1] <= re)

        prev = rs

        for ds, de in dets:

            if ds > prev:

                gaps.append(ds - prev)

            prev = max(prev, de)

        if re > prev:

            gaps.append(re - prev)

    return gaps

# ============================================================*

#  STRUCTURED EVENT LOGGER*

# ============================================================*

class EventLogger:

    HEADER = ["run_id", "event", "start_epoch", "end_epoch", "duration_s",

              "peak_confidence", "mean_confidence", "start_iso", "end_iso"]

    def __init__(self, csv_path, text_path, run_id, model_path):

        self.csv_path, self.text_path, self.run_id = csv_path, text_path, run_id

        fresh = not os.path.exists(csv_path)

        self.fh = open(csv_path, "a", newline="", encoding="utf-8")

        self.w = csv.writer(self.fh)

        if fresh:

            self.w.writerow(self.HEADER)

            self.fh.flush()

        with open(self.text_path, "a", encoding="utf-8") as f:

            f.write(f"\n=== Run {run_id} started {iso(time.time())} | "

                    f"model={model_path} ===\n")

    def _row(self, event, s, e, peak=None, mean=None):

        self.w.writerow([self.run_id, event, f"{s:.3f}", f"{e:.3f}", f"{e - s:.3f}",

                         "" if peak is None else f"{peak:.3f}",

                         "" if mean is None else f"{mean:.3f}",

                         iso(s), iso(e)])

        self.fh.flush()

    def detection(self, s, e, peak, mean):

        self._row("DETECTION", s, e, peak, mean)

        with open(self.text_path, "a", encoding="utf-8") as f:

            f.write(f"{iso(s)} -> {iso(e)} | ELEPHANT {e - s:.1f}s | "

                    f"peak {peak:.2f} | mean {mean:.2f}\n")

    def run(self, s, e):

        self._row("RUN", s, e)

    def close(self):

        try:

            self.fh.close()

        except Exception:

            pass

# ============================================================*

#  LIVE ANALYTICS*

# ============================================================*

class LiveAnalytics:

    def __init__(self, run_id, history, logger):

        self.run_id = run_id

        self.history = history

        self.logger = logger

        self.t0 = time.time()

        self.streak_start = None

        self.streak_conf_sum = 0.0

        self.streak_conf_n = 0

        self.streak_conf_peak = 0.0

        self.detections = []

        self.gaps = []

        self.last_det_end = None

        self.gap_start = self.t0

        self.samples = deque()

        self.conf_hist = deque(maxlen=CONF_WINDOW_N)

        self.prob_history = deque(maxlen=PROB_HISTORY_N)

        self._last_push = 0.0

    def update(self, detected, conf, now):

        if now - self._last_push >= SAMPLE_INTERVAL_S:

            self.samples.append((now, 1 if detected else 0))

            self._last_push = now

        cutoff = now - RATE_WINDOW_S

        while self.samples and self.samples[0][0] < cutoff:

            self.samples.popleft()

        if detected:

            if self.streak_start is None:

                self.streak_start = now

                self.streak_conf_sum = 0.0

                self.streak_conf_n = 0

                self.streak_conf_peak = 0.0

                if self.gap_start is not None and now > self.gap_start:

                    self.gaps.append((self.gap_start, now))

                self.gap_start = None

            self.streak_conf_sum += conf

            self.streak_conf_n += 1

            self.streak_conf_peak = max(self.streak_conf_peak, conf)

            self.conf_hist.append(conf)

            self.last_det_end = now

        else:

            if self.streak_start is not None:

                self._close_streak(now)

                self.gap_start = now

    def _close_streak(self, now):

        dur = now - self.streak_start

        if dur >= MIN_DETECTION_S and self.streak_conf_n > 0:

            rec = (self.streak_start, now, self.streak_conf_peak,

                   self.streak_conf_sum / self.streak_conf_n)

            self.detections.append(rec)

            self.logger.detection(*rec)

        self.streak_start = None

    def flush(self):

        if self.streak_start is not None:

            self._close_streak(time.time())

            self.streak_start = None

    @property

    def recent_rate(self):

        if not self.samples:

            return 0.0

        return sum(v for _, v in self.samples) / len(self.samples)

    def session_detected_s(self, now=None):

        total = sum(e - s for s, e, _, _ in self.detections)

        if self.streak_start is not None:

            total += (now or time.time()) - self.streak_start

        return total

    def session_duration_s(self, now=None):

        return (now or time.time()) - self.t0

    def since_last_detection(self, now=None):

        if self.streak_start is not None:

            return 0.0

        if self.last_det_end is None:

            return None

        return (now or time.time()) - self.last_det_end

    def mean_conf(self):

        return statistics.fmean(self.conf_hist) if self.conf_hist else 0.0

    def peak_conf(self):

        return max(self.conf_hist) if self.conf_hist else 0.0

# ============================================================*

#  PRESENCE PREDICTOR*

# ============================================================*

class PresencePredictor:

    def __init__(self, history):

        self.history = history

        obs_total = sum(history["hour_observed"].values())

        det_total = sum(history["hour_detected"].values())

        self.base_rate = (det_total / obs_total) if obs_total > 1e-6 else 0.05

        self.hour_prior = {}

        for h in range(24):

            o = history["hour_observed"].get(h, 0.0)

            d = history["hour_detected"].get(h, 0.0)

            self.hour_prior[h] = (d + PRIOR_SMOOTHING_S * self.base_rate) / \
                                 (o + PRIOR_SMOOTHING_S)

        gaps = derive_gaps(history)

        self.median_gap = statistics.median(gaps) if gaps else None

        self.peak_hour = max(self.hour_prior, key=self.hour_prior.get) \
                         if obs_total > 0 else None

    def predict(self, now, detected, last_det_end, recent_rate):

        if detected:

            return 0.99, "CRITICAL", 1.0, self.hour_prior.get(

                datetime.fromtimestamp(now).hour, self.base_rate)

        hour = datetime.fromtimestamp(now).hour

        p_hour = self.hour_prior.get(hour, self.base_rate)

        p_rec = 0.0 if last_det_end is None else \
                math.exp(-max(0.0, now - last_det_end) / DECAY_TAU_S)

        p = W_RECENCY * p_rec + W_RATE * recent_rate + W_PRIOR * p_hour

        p = min(max(p, 0.0), 0.99)

        return p, risk_label(p), p_rec, p_hour

def risk_label(p):

    if p >= 0.70: return "CRITICAL"

    if p >= 0.45: return "HIGH"

    if p >= 0.20: return "MEDIUM"

    return "LOW"

# ============================================================*

#  MSP430 SERIAL*

# ============================================================*

class NullSerial:

    def write(self, *_): pass

    def close(self): pass

def find_msp430_uart():

    """

    Find the MSP430 Application UART.

    On this desktop, Windows identifies both MSP430 interfaces

    as 'USB Serial Device'. COM3 has been experimentally

    confirmed as the Application UART.

    """

    ports = serial.tools.list_ports.comports()

    # First try the original description-based detection*

    for p in ports:

        print(f"Found port: {p.device} - {p.description}")

        if "Application UART" in p.description or "MSP430" in p.description:

            print(f"✓ MSP430 Application UART found: {p.device}")

            return p.device

    # Desktop fallback:*

    # COM3 has been experimentally confirmed to control*

    # the MSP430 application LEDs.*

    for p in ports:

        if p.device == "COM3":

            print("✓ MSP430 Application UART found: COM3")

            return p.device

    return None

class SignalController:

    def __init__(self, ser):

        self.ser = ser

        self.state = None

    def set(self, state):

        if state == self.state:

            return

        self.state = state

        self.ser.write((state + "\n").encode("ascii"))

        if state == "ELEPHANT":

            print("➡️ Sent ELEPHANT   → RED   ■■■")

        elif state == "CAUTION":

            print("➡️ Sent CAUTION    → AMBER ■■■  (predictive)")

        else:

            print("➡️ Sent NOELEPHANT → GREEN ■■■")

# ============================================================*
#  ESP32 BLE TRAIN CONTROLLER*
# ============================================================*
class BLETrainController:
    """Non-blocking BLE STOP/RUN client with periodic heartbeats."""
    def __init__(self):
        self.device_name = TRAIN_BLE_DEVICE_NAME
        self.characteristic_uuid = TRAIN_BLE_CHAR_UUID.lower()
        self._desired_state = "STOP"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._thread_main,
            name="esp32-ble-train-control", daemon=True)
        self._thread.start()
        print(f"🚂 BLE train controller starting; looking for {self.device_name!r}")
        return self

    def set_state(self, state):
        state = str(state).strip().upper()
        if state not in ("STOP", "RUN"):
            raise ValueError("Train state must be STOP or RUN")
        with self._lock:
            self._desired_state = state

    def _thread_main(self):
        asyncio.run(self._ble_loop())

    async def _ble_loop(self):
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            print("⚠️ BLE train control requires Bleak. Install with: python3 -m pip install bleak")
            return
        while not self._stop_event.is_set():
            try:
                device = await BleakScanner.find_device_by_name(self.device_name, timeout=5.0)
                if device is None:
                    print(f"🔎 ESP32 {self.device_name!r} not found; retrying")
                    await asyncio.sleep(2.0)
                    continue
                async with BleakClient(device, timeout=10.0) as client:
                    print("✅ BLE connected to ESP32 train controller")
                    while client.is_connected and not self._stop_event.is_set():
                        with self._lock:
                            state = self._desired_state
                        await client.write_gatt_char(
                            self.characteristic_uuid, (state + "\n").encode("ascii"), response=True)
                        await asyncio.sleep(TRAIN_BLE_HEARTBEAT_S)
            except Exception as exc:
                if not self._stop_event.is_set():
                    print(f"⚠️ BLE train link unavailable ({exc}); retrying")
                    await asyncio.sleep(2.0)

    def stop(self):
        self.set_state("STOP")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=6.0)


# ============================================================*

#  THREADED LATEST-FRAME READER*

# ============================================================*

class LatestFrame:
    """Continuously read the camera in a background thread.

    Keeping only the newest frame avoids processing an increasingly stale
    camera buffer when inference is slower than the camera capture rate.
    """

    def __init__(self, capture):
        self.capture = capture
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stopped = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._update, name="camera-frame-reader", daemon=True
        )
        self._thread.start()
        return self

    def _update(self):
        while not self._stopped.is_set():
            ok, frame = self.capture.read()
            if ok and frame is not None and frame.size > 0:
                with self._lock:
                    self._frame = frame
                    self._ok = True
            else:
                # Avoid a busy loop if the camera temporarily stops delivering frames.
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def stop(self):
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ============================================================*

#  CAMERA AUTO-DETECT*

# ============================================================*

def open_camera(indices=CAMERA_INDICES):

    backends = (("DSHOW", cv2.CAP_DSHOW),

                ("MSMF",  cv2.CAP_MSMF),

                ("ANY",   cv2.CAP_ANY))

    print(f"🎥 Probing camera indices {indices} (forced order) …")

    for idx in indices:

        for name, flag in backends:

            cap = cv2.VideoCapture(idx, flag)

            if not cap.isOpened():

                cap.release()

                continue

            ret, frame = cap.read()

            if ret and frame is not None and frame.size > 0:

                print(f"✅ Camera opened: index={idx} backend={name} "

                      f"({frame.shape[1]}x{frame.shape[0]})")

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                return cap

            print(f"⚠️  index={idx} backend={name} opened but no frame")

            cap.release()

    return None

# ============================================================*

#  CAMERA-WINDOW RENDERER*

# ============================================================*

def prepare_camera_canvas(frame, results, model_names, win_w, win_h):

    h, w = frame.shape[:2]

    # Fill the viewport and center-crop; never add black side bars.
    scale = max(win_w / w, win_h / h)

    new_w = max(1, int(w * scale))

    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    detected = False

    best_conf = 0.0

    box_thickness = max(2, int(round(2 * scale)))

    font_box      = max(0.5, min(1.4, 0.6 * scale))

    font_status   = max(0.8, min(2.0, 1.0 * scale))

    for box in results[0].boxes:

        cls_id = int(box.cls[0].item())

        conf = float(box.conf[0].item())

        if model_names[cls_id] == "elephant" and conf > CONF_THRESHOLD:

            detected = True

            best_conf = max(best_conf, conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = int(x1 * scale); y1 = int(y1 * scale)

            x2 = int(x2 * scale); y2 = int(y2 * scale)

            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)

            cv2.putText(resized, f"Elephant {conf:.2f}",

                        (x1, max(int(15 * scale), y1 - 10)),

                        cv2.FONT_HERSHEY_SIMPLEX, font_box, (0, 0, 255),

                        box_thickness, cv2.LINE_AA)

    status_text = "Elephant Detected!" if detected else "No Elephant"

    color = (0, 255, 0) if detected else (255, 255, 255)

    cv2.putText(resized, status_text,

                (int(20 * scale), int(40 * scale)),

                cv2.FONT_HERSHEY_SIMPLEX, font_status, color,

                max(2, int(round(2 * scale))), cv2.LINE_AA)

    # Crop the scaled image to the current window size (desktop/Raspberry Pi).
    x_off = max(0, (new_w - win_w) // 2)
    y_off = max(0, (new_h - win_h) // 2)
    canvas = resized[y_off:y_off + win_h, x_off:x_off + win_w]
    if canvas.shape[1] != win_w or canvas.shape[0] != win_h:
        canvas = cv2.resize(canvas, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
    return canvas, detected, best_conf


# ============================================================*

#  ANALYTICS-WINDOW RENDERER*

# ============================================================*

def render_analytics_window(analytics, predictor, p, risk, p_rec, p_hour,
                            detected, now, W, H):
    """
    Responsive wildlife analytics dashboard.
    Layout switches between wide, compact and narrow modes based on the
    actual window dimensions. Text and cards are clipped to their panels.
    """
    W, H = max(420, int(W)), max(360, int(H))
    canvas = np.full((H, W, 3), (18, 22, 30), dtype=np.uint8)

    # Scale typography and spacing gently, while retaining a readable minimum.
    scale = min(W / 1100.0, H / 760.0)
    scale = max(0.62, min(1.35, scale))
    def S(v):
        return max(1, int(round(v * scale)))

    # Palette: deep navy, soft slate, cyan/teal accents and semantic status.
    BG = (18, 22, 30)
    HEADER = (24, 30, 41)
    PANEL = (29, 36, 48)
    PANEL_INNER = (24, 30, 41)
    BORDER = (57, 69, 86)
    GRID = (49, 60, 75)
    WHITE = (239, 244, 250)
    MUTED = (158, 172, 190)
    CYAN = (225, 177, 91)
    TEAL = (110, 211, 177)
    PURPLE = (190, 155, 255)
    RED = (95, 100, 255)
    GREEN = (100, 210, 143)
    AMBER = (70, 190, 255)

    def txt(text, x, y, size=0.48, color=WHITE, thick=1, max_width=None):
        text = str(text)
        font_scale = max(0.38, size * scale)
        thickness = max(1, int(round(thick * scale)))
        if max_width is not None:
            # Shrink long labels to fit the panel instead of overflowing.
            while font_scale > 0.36 and cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] > max_width:
                font_scale *= 0.92
        cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

    def panel(x, y, w, h, title=None, accent=CYAN):
        x, y, w, h = int(x), int(y), max(1, int(w)), max(1, int(h))
        cv2.rectangle(canvas, (x, y), (x+w, y+h), PANEL, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), BORDER, max(1, S(1)), cv2.LINE_AA)
        cv2.line(canvas, (x+S(12), y+S(3)), (x+w-S(12), y+S(3)), accent, S(2), cv2.LINE_AA)
        if title and h > S(28):
            txt(title, x+S(14), y+S(25), .43, accent, 1, w-S(28))

    def donut(cx, cy, radius, value, color, center_text, caption):
        radius = max(S(15), int(radius))
        thickness = max(3, S(7))
        cv2.circle(canvas, (int(cx), int(cy)), radius, GRID, thickness, cv2.LINE_AA)
        angle = int(360 * max(0.0, min(1.0, float(value))))
        if angle:
            cv2.ellipse(canvas, (int(cx), int(cy)), (radius, radius),
                        -90, 0, angle, color, thickness, cv2.LINE_AA)
        tw = cv2.getTextSize(str(center_text), cv2.FONT_HERSHEY_SIMPLEX,
                             max(.38, .55*scale), max(1, S(1)))[0][0]
        txt(center_text, cx-tw//2, cy+S(5), .55, WHITE, 2)
        txt(caption, cx-S(40), cy+radius+S(19), .30, MUTED, 1, S(110))

    def chart_grid(x, y, w, h, rows=4):
        for i in range(rows+1):
            gy = int(y + i*h/rows)
            cv2.line(canvas, (int(x), gy), (int(x+w), gy), GRID, 1, cv2.LINE_AA)

    def line_chart(x, y, w, h, values, color, maxval=1.0):
        chart_grid(x, y, w, h)
        vals = list(values)[-180:]
        if not vals or w < 3 or h < 3:
            txt("Collecting live samples…", x+S(10), y+max(S(20), h//2), .35, MUTED)
            return
        if len(vals) == 1:
            vals = [vals[0], vals[0]]
        pts = []
        for i, value in enumerate(vals):
            xx = int(x + i*(w-1)/max(1, len(vals)-1))
            yy = int(y+h - max(0, min(1, float(value)/maxval))*h)
            pts.append((xx, yy))
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False,
                      color, max(1, S(2)), cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], max(2, S(3)), color, -1, cv2.LINE_AA)

    def bar_chart(x, y, w, h, values, color):
        chart_grid(x, y, w, h, 3)
        vals = list(values)
        if not vals or w < 8 or h < 8:
            return
        mx = max(max(vals), 1.0)
        slot = w / len(vals)
        bw = max(1, int(slot*.62))
        for i, value in enumerate(vals):
            bh = int((max(0, value)/mx)*max(1, h-S(3)))
            bx = int(x+i*slot+(slot-bw)/2)
            if bh:
                cv2.rectangle(canvas, (bx, int(y+h-bh)), (bx+bw, int(y+h)), color, -1)
        for hr in (0, 6, 12, 18, 23):
            xx = int(x+hr*slot)
            txt(f"{hr:02d}h", xx, y+h+S(17), .28, MUTED, 1)

    pad = max(S(10), min(S(20), W//35))
    gap = max(S(7), min(S(14), W//75))
    header_h = max(S(62), int(H*.095))
    inner_w = W-2*pad
    is_narrow = W < 760
    is_short = H < 560

    # Header stays clean even when the window is narrow.
    cv2.rectangle(canvas, (0, 0), (W, header_h), HEADER, -1)
    cv2.circle(canvas, (pad+S(12), header_h//2), max(4, S(9)), CYAN, -1, cv2.LINE_AA)
    if is_narrow:
        txt("WILDLIFE ANALYTICS", pad+S(28), S(25), .43, CYAN, 1, inner_w-S(40))
        txt("Elephant monitoring dashboard", pad+S(28), S(47), .50, WHITE, 2, inner_w-S(40))
    else:
        txt("WILDLIFE MONITOR", pad+S(32), S(25), .39, CYAN, 1)
        txt("Elephant Detection  /  Live Analytics", pad+S(32), S(49), .62, WHITE, 2)
        txt(f"SESSION  {analytics.run_id}", max(pad+S(250), W-S(255)),
            S(38), .34, MUTED, 1, W//3)

    sess = max(0, analytics.session_duration_s(now))
    det_s = max(0, analytics.session_detected_s(now))
    pct = min(100.0, (det_s/sess*100) if sess > 0 else 0.0)
    detections = len(getattr(analytics, "detections", []))
    if getattr(analytics, "streak_start", None) is not None:
        detections += 1
    status = "ELEPHANT DETECTED" if detected else "NO ELEPHANT DETECTED"
    status_col = RED if detected else GREEN

    # Responsive KPI cards: 2x2 on narrow windows, 4 across on wide windows.
    content_y = header_h + pad
    kpi_gap = gap
    kpi_h = max(S(76), min(S(112), int(H*.13)))
    cols = 2 if is_narrow else 4
    rows = 2 if cols == 2 else 1
    kpi_w = (inner_w-(cols-1)*kpi_gap)//cols
    kpis = [
        ("LIVE STATUS", status, "Current camera result", status_col),
        ("DETECTIONS", str(detections), "This session", CYAN),
        ("SESSION TIME", fmt_dur_short(sess), "Monitoring duration", PURPLE),
        ("DETECTED TIME", f"{pct:.1f}%", "Share of session", TEAL),
    ]
    for i, (label, value, sub, accent) in enumerate(kpis):
        row, col = divmod(i, cols)
        x = pad + col*(kpi_w+kpi_gap)
        y = content_y + row*(kpi_h+kpi_gap)
        panel(x, y, kpi_w, kpi_h, label, accent)
        txt(value, x+S(13), y+S(52 if kpi_h > S(88) else 43),
            .57 if label != "LIVE STATUS" else .43, WHITE, 2, kpi_w-S(25))
        if kpi_h > S(88):
            txt(sub, x+S(13), y+kpi_h-S(13), .30, MUTED, 1, kpi_w-S(25))

    after_kpis = content_y + rows*kpi_h + (rows-1)*kpi_gap + gap
    # Compact windows prioritize live status and charts over decorative whitespace.
    remaining_h = H-after_kpis-pad
    if remaining_h < S(130):
        # In very short windows, keep the KPI row readable and show a compact risk strip.
        strip_h = max(S(55), remaining_h)
        panel(pad, after_kpis, inner_w, strip_h, "PRESENCE RISK", RISK_BGR.get(risk, AMBER))
        txt(f"{risk}  |  {float(p)*100:.0f}% estimated presence probability",
            pad+S(15), after_kpis+min(strip_h-S(8), S(48)), .42, WHITE, 1, inner_w-S(30))
        return canvas

    # Risk and session summary use stacked panels on narrow windows.
    summary_h = min(S(190), max(S(130), int(remaining_h*.30)))
    if is_narrow:
        risk_h = max(S(112), (summary_h-gap)//2)
        panel(pad, after_kpis, inner_w, risk_h, "PRESENCE RISK · NEXT 60 SECONDS",
              RISK_BGR.get(risk, AMBER))
        donut(pad+S(65), after_kpis+risk_h//2+S(2), S(32), p,
              RISK_BGR.get(risk, AMBER), f"{float(p)*100:.0f}%", "PROBABILITY")
        tx = pad+S(125)
        txt(f"Risk level: {risk}", tx, after_kpis+S(48), .43,
            RISK_BGR.get(risk, AMBER), 2, inner_w-S(140))
        txt(f"Recent activity  {float(p_rec):.2f}", tx, after_kpis+S(73), .34, WHITE, 1, inner_w-S(140))
        txt(f"Hourly baseline  {float(p_hour):.2f}", tx, after_kpis+S(96), .34, MUTED, 1, inner_w-S(140))
        sy = after_kpis+risk_h+gap
        sess_h = max(S(95), summary_h-risk_h-gap)
        panel(pad, sy, inner_w, sess_h, "SESSION BREAKDOWN", TEAL)
        txt(f"Detected  {fmt_dur_short(det_s)}  ({pct:.1f}%)",
            pad+S(14), sy+S(49), .40, TEAL, 1, inner_w-S(28))
        txt(f"Clear time  {fmt_dur_short(max(0, sess-det_s))}",
            pad+S(14), sy+S(75), .37, MUTED, 1, inner_w-S(28))
        charts_y = sy+sess_h+gap
    else:
        left_w = int((inner_w-gap)*.59)
        right_w = inner_w-gap-left_w
        panel(pad, after_kpis, left_w, summary_h, "PRESENCE RISK · NEXT 60 SECONDS",
              RISK_BGR.get(risk, AMBER))
        donut(pad+S(75), after_kpis+summary_h//2, S(39), p,
              RISK_BGR.get(risk, AMBER), f"{float(p)*100:.0f}%", "PROBABILITY")
        tx = pad+S(140)
        txt(f"Risk level: {risk}", tx, after_kpis+S(55), .48,
            RISK_BGR.get(risk, AMBER), 2, left_w-S(155))
        txt(f"Recent activity  {float(p_rec):.2f}", tx, after_kpis+S(85), .37, WHITE, 1, left_w-S(155))
        txt(f"Hourly baseline  {float(p_hour):.2f}", tx, after_kpis+S(112), .37, MUTED, 1, left_w-S(155))
        mg = getattr(predictor, "median_gap", None)
        gap_text = f"Typical detection gap  {mg:.0f}s" if isinstance(mg, (int, float)) else "Typical detection gap  —"
        txt(gap_text, tx, after_kpis+S(139), .34, MUTED, 1, left_w-S(155))
        rx = pad+left_w+gap
        panel(rx, after_kpis, right_w, summary_h, "SESSION BREAKDOWN", TEAL)
        donut(rx+right_w//2, after_kpis+summary_h//2-S(3), S(39), pct/100,
              TEAL, f"{pct:.0f}%", "DETECTED TIME")
        txt(f"Detected {fmt_dur_short(det_s)}  ·  Clear {fmt_dur_short(max(0, sess-det_s))}",
            rx+S(12), after_kpis+summary_h-S(14), .31, TEAL, 1, right_w-S(24))
        charts_y = after_kpis+summary_h+gap

    # Charts adapt to the available width and height: stacked when narrow,
    # side-by-side when wide. Plot areas never overlap the footer.
    footer_h = max(S(24), S(30))
    chart_space = H-charts_y-pad-footer_h-gap
    stack_charts = is_narrow or chart_space > 0 and (W < 900 or chart_space < S(210))
    if stack_charts:
        chart_gap = gap
        chart_h = max(S(88), (chart_space-chart_gap)//2)
        chart_w = inner_w
        panel(pad, charts_y, chart_w, chart_h, "DETECTIONS BY HOUR", CYAN)
        obs = [0.0]*24
        det = [0.0]*24
        for hr, val in analytics.history.get("hour_observed", {}).items():
            if 0 <= int(hr) < 24: obs[int(hr)] += float(val)
        for hr, val in analytics.history.get("hour_detected", {}).items():
            if 0 <= int(hr) < 24: det[int(hr)] += float(val)
        for hr, val in split_by_hour(analytics.t0, now): obs[hr] += val
        for a0, b0, *_ in analytics.detections:
            for hr, val in split_by_hour(a0, b0): det[hr] += val
        if getattr(analytics, "streak_start", None) is not None:
            for hr, val in split_by_hour(analytics.streak_start, now): det[hr] += val
        plot_y = charts_y+S(39)
        plot_h = max(8, chart_h-S(61))
        bar_chart(pad+S(16), plot_y, chart_w-S(32), plot_h, det, CYAN)
        chart2_y = charts_y+chart_h+chart_gap
        chart2_h = max(S(75), chart_space-chart_h-chart_gap)
        panel(pad, chart2_y, chart_w, chart2_h, "PRESENCE PROBABILITY · LAST 30 MIN", PURPLE)
        line_chart(pad+S(16), chart2_y+S(39), chart_w-S(32),
                   max(8, chart2_h-S(52)), getattr(analytics, "prob_history", []), PURPLE)
    else:
        chart_h = max(S(120), chart_space)
        half = (inner_w-gap)//2
        panel(pad, charts_y, half, chart_h, "DETECTIONS BY HOUR", CYAN)
        obs = [0.0]*24
        det = [0.0]*24
        for hr, val in analytics.history.get("hour_observed", {}).items():
            if 0 <= int(hr) < 24: obs[int(hr)] += float(val)
        for hr, val in analytics.history.get("hour_detected", {}).items():
            if 0 <= int(hr) < 24: det[int(hr)] += float(val)
        for hr, val in split_by_hour(analytics.t0, now): obs[hr] += val
        for a0, b0, *_ in analytics.detections:
            for hr, val in split_by_hour(a0, b0): det[hr] += val
        if getattr(analytics, "streak_start", None) is not None:
            for hr, val in split_by_hour(analytics.streak_start, now): det[hr] += val
        bar_chart(pad+S(16), charts_y+S(43), half-S(32),
                  max(8, chart_h-S(65)), det, CYAN)
        x2 = pad+half+gap
        panel(x2, charts_y, half, chart_h, "PRESENCE PROBABILITY · LAST 30 MIN", PURPLE)
        line_chart(x2+S(16), charts_y+S(43), half-S(32),
                   max(8, chart_h-S(65)), getattr(analytics, "prob_history", []), PURPLE)

    # Slim footer; avoids overlaying chart labels on smaller window sizes.
    fy = H-footer_h
    cv2.line(canvas, (pad, fy), (W-pad, fy), BORDER, 1, cv2.LINE_AA)
    hist_runs = analytics.history.get("n_runs", 0)
    footer = f"History: {hist_runs} previous runs  |  Updated {iso(now)}  |  Press Q to quit"
    txt(footer, pad, H-S(8), .29, MUTED, 1, inner_w)
    return canvas

def write_report(path, run_id, a, predictor, p, risk, p_rec, p_hour,

                 detected, now):

    sess = a.session_duration_s(now)

    det_s = a.session_detected_s(now)

    pct = (det_s / sess * 100) if sess > 0 else 0.0

    sld = a.since_last_detection(now)

    tot_obs = sum(predictor.history["hour_observed"].values())

    tot_det = sum(predictor.history["hour_detected"].values())

    n_runs  = predictor.history["n_runs"]

    comb_obs = tot_obs + sess

    comb_det = tot_det + det_s

    comb_rate = (comb_det / comb_obs * 100) if comb_obs > 0 else 0.0

    lines = [

        "================ LIVE ANALYTICS ================",

        f"Run ID      : {run_id}",

        f"Updated     : {iso(now)}",

        f"Session     : {hms(sess)}",

        "",

        "DETECTION",

        f"  Current state        : "

        f"{'ELEPHANT PRESENT (' + format(now - a.streak_start, '.1f') + ' s)' if detected else 'clear'}",

        f"  Detections this run  : {len(a.detections)}",

        f"  Time detected        : {hms(det_s)} ({pct:.1f} % of session)",

        f"  Last detection       : {'n/a' if sld is None else f'{sld:.1f} s ago'}",

        f"  Confidence           : mean {a.mean_conf():.2f} | peak {a.peak_conf():.2f}",

        f"  5-min activity rate  : {a.recent_rate * 100:.1f} %",

        "",

        "PREDICTION in next 60 s",

        f"  P(presence)          : {p:.2f}   [{risk}]",

        f"  Hour-of-day prior    : {p_hour:.2f}"

        + (f"   (peak hour {predictor.peak_hour:02d}:00)"

           if predictor.peak_hour is not None else ""),

        f"  Recency component    : {p_rec:.2f}   (tau = {DECAY_TAU_S:.0f} s)",

        f"  Historical median gap: "

        f"{'n/a' if predictor.median_gap is None else f'{predictor.median_gap:.0f} s'}",

        "",

        "HISTORY",

        f"  Number of Previous runs: {n_runs}",

        f"  Previous Observed Duration: {hms(tot_obs)}",

        f"  Previous Elephant detected Duration: {hms(tot_det)}"

        f"  ({(tot_det/tot_obs*100) if tot_obs > 0 else 0:.1f} %)",

        f"  Past + current runs: {n_runs + 1}",

        f"  Combined Observed Duration: {hms(comb_obs)}",

        f"  Combined Detected Duration: {hms(comb_det)}  ({comb_rate:.1f} %)",

        "================================================",

    ]

    with open(path, "w", encoding="utf-8") as f:

        f.write("\n".join(lines) + "\n")

def write_state_json(path, run_id, a, predictor, p, risk, detected, now):

    state = {

        "run_id": run_id,

        "updated": iso(now),

        "elephant_detected": bool(detected),

        "prediction": round(p, 3),

        "risk": risk,

        "recent_rate_5min": round(a.recent_rate, 3),

        "detections_this_run": len(a.detections),

        "session_seconds": round(a.session_duration_s(now), 1),

        "detected_seconds": round(a.session_detected_s(now), 1),

        "mean_confidence": round(a.mean_conf(), 3),

        "peak_confidence": round(a.peak_conf(), 3),

        "history": {

            "runs": predictor.history["n_runs"],

            "base_rate": round(predictor.base_rate, 4),

            "median_gap_s": (round(predictor.median_gap, 1)

                             if predictor.median_gap else None),

            "peak_hour": predictor.peak_hour,

            "hour_prior": {str(h): round(v, 4)

                           for h, v in predictor.hour_prior.items()},

        },

    }

    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:

        json.dump(state, f, indent=2)

    os.replace(tmp, path)

def save_session_plot(path, predictor, a, now):

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:

        return

    hours = list(range(24))

    priors = [predictor.hour_prior.get(h, predictor.base_rate) * 100 for h in hours]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(hours, priors, color="#c0392b")

    axes[0].set_title("Historical detection risk by hour of day")

    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("% time with elephant")

    axes[0].set_xticks(hours); axes[0].grid(axis="y", alpha=.3)

    dets = a.detections

    if dets:

        t0 = a.t0

        for s, e, pk, _ in dets:

            axes[1].barh(0, e - s, left=s - t0, height=0.5,

                         color="#c0392b", alpha=0.8)

        axes[1].set_ylim(-1, 1)

        axes[1].set_xlim(0, max(1, a.session_duration_s(now)))

        axes[1].set_yticks([])

    axes[1].set_title("This session — detection timeline")

    axes[1].set_xlabel("Seconds since session start")

    axes[1].grid(axis="x", alpha=.3)

    fig.tight_layout()

    fig.savefig(path, dpi=110)

    plt.close(fig)

    print(f"📊 Session chart saved: {path}")

# ============================================================*

#  MAIN*

# ============================================================*

def main():

    os.makedirs(LOG_DIR, exist_ok=True)

    try:
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    model = YOLO(MODEL_PATH)
    model_names = model.names

    history = load_history(CSV_LOG)

    logger = EventLogger(CSV_LOG, TEXT_LOG, RUN_ID, MODEL_PATH)

    analytics = LiveAnalytics(RUN_ID, history, logger)

    predictor = PresencePredictor(history)

    print(f"📚 History loaded: {history['n_runs']} previous run(s), "

          f"base rate {predictor.base_rate * 100:.1f} %")

    port = find_msp430_uart()

    if port:

        print(f"✅ Using {port} for MSP430 communication")

        ser = serial.Serial(port, SERIAL_BAUD, timeout=1)

    elif REQUIRE_MSP430:

        raise Exception("MSP430 Application UART not found")

    else:

        print("⚠️  MSP430 not found — running in dry-run mode")

        ser = NullSerial()

    signal = SignalController(ser)

    train_controller = BLETrainController().start() if TRAIN_BLE_ENABLE else None

    cap = open_camera()

    if cap is None:

        print(f"❌ Could not open any of camera indices {CAMERA_INDICES}.")

        print("   • Is your USB camera plugged in and recognized by Windows?")

        print("   • Is another app using it? (Teams / Zoom / Camera / browser)")

        if train_controller is not None:
            train_controller.stop()

        logger.close()

        sys.exit(1)

    camera = LatestFrame(cap).start()

    # Start with a clean OpenCV window set, then create exactly one camera and one analytics window.
    cv2.destroyAllWindows()
    # WINDOW_GUI_NORMAL suppresses OpenCV's extra toolbar area on supported backends.
    window_flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
    cv2.namedWindow(CAMERA_WINDOW, window_flags)
    cv2.namedWindow(ANALYTICS_WINDOW, window_flags)
    cv2.resizeWindow(CAMERA_WINDOW, *CAMERA_DEFAULT_SIZE)
    cv2.resizeWindow(ANALYTICS_WINDOW, *ANALYTICS_DEFAULT_SZ)
    cv2.waitKey(100)  # allow the native windows to be created before styling
    make_borderless_split_windows(CAMERA_WINDOW, ANALYTICS_WINDOW)

    print("✅ Borderless camera + analytics windows fitted edge-to-edge. Press q to quit.")


    last_report = 0.0
    last_prob_push = 0.0
    last_analytics_draw = 0.0
    analytics_canvas = None
    last_frame = None
    fps_window_start = time.time()
    fps_window_frames = 0
    fps_display = 0.0
    train_stop_latched = False
    train_clear_since = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                if last_frame is None:
                    time.sleep(0.01)
                    continue
                frame = last_frame
            else:
                last_frame = frame

            now = time.time()
            # COCO elephant class ID is 20. Small input + nano model reduces CPU work.
            results = model.predict(
                source=frame,
                imgsz=INFERENCE_SIZE,
                conf=CONF_THRESHOLD,
                classes=[20],
                max_det=MAX_DETECTIONS,
                device="cpu",
                verbose=False
            )

            cam_w, cam_h = get_window_size(CAMERA_WINDOW, CAMERA_DEFAULT_SIZE)
            ana_w, ana_h = get_window_size(ANALYTICS_WINDOW, ANALYTICS_DEFAULT_SZ)

            cam_canvas, detected, best_conf = prepare_camera_canvas(

                frame, results, model_names, cam_w, cam_h)

            analytics.update(detected, best_conf, now)

            fps_window_frames += 1
            fps_elapsed = now - fps_window_start
            if fps_elapsed >= 2.0:
                fps_display = fps_window_frames / max(fps_elapsed, 1e-6)
                fps_window_frames = 0
                fps_window_start = now
            cv2.putText(cam_canvas, f"Inference loop: {fps_display:.1f} FPS",
                        (12, max(24, cam_canvas.shape[0] - 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

            p, risk, p_rec, p_hour = predictor.predict(

                now, detected, analytics.last_det_end, analytics.recent_rate)

            if now - last_prob_push >= 1.0:

                analytics.prob_history.append(p)

                last_prob_push = now

            if detected:

                target = "ELEPHANT"

            elif SEND_PREDICTIVE_CAUTION and risk in ("HIGH", "CRITICAL"):

                target = "CAUTION"

            else:

                target = "NOELEPHANT"

            signal.set(target)

            # Stop immediately on detection; resume only after a continuous clear period.
            # This avoids brief RUN pulses if detection flickers for a frame or two.
            if train_controller is not None:
                if detected:
                    train_stop_latched = True
                    train_clear_since = None
                elif train_stop_latched:
                    if train_clear_since is None:
                        train_clear_since = now
                    elif now - train_clear_since >= TRAIN_CLEAR_BEFORE_RESUME_S:
                        train_stop_latched = False
                        train_clear_since = None
                train_controller.set_state("STOP" if train_stop_latched else "RUN")

            cv2.imshow(CAMERA_WINDOW, cam_canvas)

            # Dashboard redraw is deliberately throttled; it is not the live video path.
            if analytics_canvas is None or now - last_analytics_draw >= 1.0 / ANALYTICS_FPS:
                analytics_canvas = render_analytics_window(
                    analytics, predictor, p, risk, p_rec, p_hour,
                    detected, now, ana_w, ana_h)
                last_analytics_draw = now
            cv2.imshow(ANALYTICS_WINDOW, analytics_canvas)

            if now - last_report >= REPORT_INTERVAL_S:

                write_report(REPORT_TXT, RUN_ID, analytics, predictor,

                             p, risk, p_rec, p_hour, detected, now)

                write_state_json(STATE_JSON, RUN_ID, analytics, predictor,

                                 p, risk, detected, now)

                print(f"[{iso(now)}] P={p:.2f} [{risk}] | "

                      f"rate={analytics.recent_rate * 100:4.1f}% | "

                      f"dets={len(analytics.detections):3d} | "

                      f"{'ELEPHANT' if detected else 'clear'}")

                last_report = now

            if (cv2.waitKey(1) & 0xFF) == ord('q'):

                break

    finally:

        end = time.time()

        analytics.flush()

        if analytics.gap_start is not None and end > analytics.gap_start:

            analytics.gaps.append((analytics.gap_start, end))

        logger.run(analytics.t0, end)

        p, risk, p_rec, p_hour = predictor.predict(

            end, False, analytics.last_det_end, analytics.recent_rate)

        write_report(REPORT_TXT, RUN_ID, analytics, predictor,

                     p, risk, p_rec, p_hour, False, end)

        write_state_json(STATE_JSON, RUN_ID, analytics, predictor,

                         p, risk, False, end)

        save_session_plot(PLOT_PNG, predictor, analytics, end)

        if train_controller is not None:
            train_controller.stop()

        camera.stop()
        cap.release()
        cv2.destroyAllWindows()

        ser.close()

        logger.close()

        print("\n📂 Session summary")

        print(f"   Structured log : {CSV_LOG}")

        print(f"   Human log      : {TEXT_LOG}")

        print(f"   Live report    : {REPORT_TXT}")

        print(f"   JSON state     : {STATE_JSON}")

        print(f"   Runs logged    : {predictor.history['n_runs'] + 1}")

if __name__ == "__main__":

    main()
