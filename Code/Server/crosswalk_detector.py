import time

import cv2
import numpy as np


class CrosswalkDetector:
    def __init__(self, green_threshold=8000, stop_duration=6.3):
        self.green_threshold = green_threshold
        self.stop_duration = stop_duration

        # Start with a broad green HSV range and tune from runtime logs.
        self.lower_green = np.array([35, 60, 60], dtype=np.uint8)
        self.upper_green = np.array([95, 255, 255], dtype=np.uint8)

        self.frame_size = (640, 480)
        self._perspective_matrix = self._build_perspective_matrix(*self.frame_size)

        self.green_pixels = 0
        self.stop_until = 0.0
        self.triggered_once = False

    def _build_perspective_matrix(self, width, height):
        src_points = np.float32([
            [-100, 400],
            [205, 290],
            [width - 205, 290],
            [740, 400],
        ])
        dst_points = np.float32([
            [width // 4, 460],
            [width // 4, 0],
            [width // 4 * 3, 0],
            [width // 4 * 3, 460],
        ])
        return cv2.getPerspectiveTransform(src_points, dst_points)

    def process_jpeg_frame(self, frame_bytes):
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            return False

        frame_resized = cv2.resize(frame, self.frame_size)
        img_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(img_hsv, self.lower_green, self.upper_green)
        warped_img_green = cv2.warpPerspective(mask_green, self._perspective_matrix, self.frame_size)

        roi_start_y = int(self.frame_size[1] * 0.3)
        warped_green_roi = warped_img_green[roi_start_y:self.frame_size[1], :]
        self.green_pixels = int(np.count_nonzero(warped_green_roi))

        if not self.triggered_once and self.green_pixels > self.green_threshold:
            self.triggered_once = True
            self.stop_until = time.time() + self.stop_duration
            return True
        return False

    def should_stop(self):
        return time.time() < self.stop_until

    def get_green_pixels(self):
        return self.green_pixels

    def get_black_pixels(self):
        return self.green_pixels
