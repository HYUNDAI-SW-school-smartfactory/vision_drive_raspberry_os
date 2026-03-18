import time

import cv2
import numpy as np


class CrosswalkDetector:
    def __init__(self, white_threshold=15000, stop_duration=6.3):
        self.white_threshold = white_threshold
        self.stop_duration = stop_duration

        # Values copied from the working lane_detection_hsv.py tuning.
        self.lower_white = np.array([0, 0, 190], dtype=np.uint8)
        self.upper_white = np.array([85, 48, 250], dtype=np.uint8)

        self.frame_size = (640, 480)
        self._perspective_matrix = self._build_perspective_matrix(*self.frame_size)

        self.white_pixels = 0
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
        mask_white = cv2.inRange(img_hsv, self.lower_white, self.upper_white)
        warped_img_white = cv2.warpPerspective(mask_white, self._perspective_matrix, self.frame_size)

        roi_start_y = int(self.frame_size[1] * 0.3)
        warped_white_roi = warped_img_white[roi_start_y:self.frame_size[1], :]
        self.white_pixels = int(np.count_nonzero(warped_white_roi))

        if not self.triggered_once and self.white_pixels > self.white_threshold:
            self.triggered_once = True
            self.stop_until = time.time() + self.stop_duration
            return True
        return False

    def should_stop(self):
        return time.time() < self.stop_until

    def get_white_pixels(self):
        return self.white_pixels
