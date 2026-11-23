"""AutoVI Edge: Video and Image Acquisition Module.

Supports:
1. V4L2 USB / MIPI Cameras (OpenCV VideoCapture)
2. Raspberry Pi Camera Module 3 / HQ Camera (Picamera2 backend)
3. Synthetic / Directory File Streamer (for offline testing & CI/CD benchmarks)
"""

import os
import time
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np


class CameraAcquisition:
    """Universal frame acquisition engine for edge deployment."""

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

        self._init_source()

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

        # 3. OpenCV VideoCapture (Camera Index or Video File)
        if isinstance(self.source, int) or (isinstance(self.source, str) and Path(self.source).is_file()):
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
                print(f"[CAMERA] Initialized OpenCV VideoCapture source={self.source}")
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

        if self.picam2 is not None:
            # Picamera2 returns RGB -> convert to BGR for consistency
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return True, frame
            if self.loop_directory and isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.cap.read()
            return False, np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

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
