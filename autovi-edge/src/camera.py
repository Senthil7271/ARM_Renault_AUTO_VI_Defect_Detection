"""AutoVI Edge: Video and Image Acquisition Module.

Supports:
1. V4L2 USB / MIPI Cameras (OpenCV VideoCapture)
2. Raspberry Pi Camera Module 3 / HQ Camera (Picamera2 backend)
3. Synthetic / Directory File Streamer (for offline testing & CI/CD benchmarks)
"""

import os
import threading
import time
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np


class CameraAcquisition:
    """Universal threaded frame acquisition engine for edge deployment."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        resolution: Tuple[int, int] = (640, 480),
        framerate: int = 30,
        loop_directory: bool = True,
    ):
        self.source = source
        self.resolution = resolution
        self.framerate = framerate
        self.loop_directory = loop_directory
        self.cap = None
        self.picam2 = None
        self.is_dir_source = False
        self.image_files = []
        self.dir_idx = 0
        self.is_synthetic = False

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stop_thread = False
        self._thread: Optional[threading.Thread] = None

        self._init_source()

        # Start non-blocking capture thread for hardware cameras
        if self.cap is not None or self.picam2 is not None:
            self._start_capture_thread()

    def _start_capture_thread(self):
        self._stop_thread = False
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

    def _capture_worker(self):
        """Dedicated background thread to continuously drain V4L2 buffers."""
        while not self._stop_thread:
            if self.picam2 is not None:
                try:
                    frame_rgb = self.picam2.capture_array()
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    with self._lock:
                        self._latest_frame = frame_bgr
                except Exception:
                    time.sleep(0.02)
                continue

            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def _init_source(self):
        # 1. Directory Source (Image dataset playback)
        if isinstance(self.source, str) and Path(self.source).is_dir():
            self.is_dir_source = True
            p = Path(self.source)
            self.image_files = sorted(
                list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
            )
            if not self.image_files:
                raise ValueError(f"No valid images found in directory: {self.source}")
            print(f"[CAMERA] Initialized directory streamer: {len(self.image_files)} images from {p.name}")
            return

        # 2. Picamera2 on Raspberry Pi (if available)
        if str(self.source).lower() == "picam2":
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                print(f"[CAMERA] Initialized Picamera2 ({self.resolution[0]}x{self.resolution[1]} @ {self.framerate} FPS)")
                return
            except Exception as e:
                print(f"[CAMERA WARNING] Picamera2 init failed ({e}), falling back to OpenCV V4L2.")

        # 3. OpenCV VideoCapture (Camera Index, Auto-scan, or Video File)
        if isinstance(self.source, int):
            # Try direct V4L2 backend on Linux first, then default backend
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                self.cap = cv2.VideoCapture(self.source, backend)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"[CAMERA] Initialized hardware camera index={self.source} backend={backend} shape={test_frame.shape}")
                        return
                    self.cap.release()
                    self.cap = None

            # Auto-scan indices 0 to 4
            for idx in [0, 1, 2, 4]:
                for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                    test_cap = cv2.VideoCapture(idx, backend)
                    if test_cap.isOpened():
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None:
                            self.cap = test_cap
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                            self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
                            print(f"[CAMERA] Auto-detected active camera at index={idx} backend={backend} shape={test_frame.shape}")
                            return
                        test_cap.release()

        elif isinstance(self.source, str) and Path(self.source).is_file():
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                print(f"[CAMERA] Initialized video file source={self.source}")
                return

        # 4. Fallback Synthetic Pattern Generator
        print(f"[CAMERA] Hardware camera not detected. Initializing synthetic test pattern generator.")
        self.is_synthetic = True

    def read(self) -> Tuple[bool, np.ndarray]:
        """Reads a single BGR frame. Returns (success, frame_bgr)."""
        if self.is_dir_source:
            if self.dir_idx >= len(self.image_files):
                if self.loop_directory:
                    self.dir_idx = 0
                else:
                    return False, np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            img_path = self.image_files[self.dir_idx]
            self.dir_idx += 1
            frame = cv2.imread(str(img_path))
            if frame is None:
                return False, np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            frame = cv2.resize(frame, self.resolution)
            return True, frame

        if self._latest_frame is not None:
            with self._lock:
                return True, self._latest_frame.copy()

        # Hot-plug detection for physical camera when synthetic
        now = time.time()
        if not hasattr(self, '_last_hotplug_check') or (now - self._last_hotplug_check > 3.0):
            self._last_hotplug_check = now
            self._init_source()
            if (self.cap is not None or self.picam2 is not None) and (self._thread is None or not self._thread.is_alive()):
                self._start_capture_thread()

        # Synthetic Test Pattern
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        t = time.time()
        cx = int((self.resolution[0] / 2) + 50 * np.sin(t * 2))
        cy = int((self.resolution[1] / 2) + 30 * np.cos(t * 2))
        cv2.circle(frame, (cx, cy), 40, (0, 165, 255), -1)
        cv2.putText(frame, f"TEST FEED {time.strftime('%H:%M:%S')}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
