import time

import cv2
import numpy as np


class CrosswalkDetector:
    def __init__(self, red_cross_walk_threshold=8000, stop_duration=6.3):
        self.red_cross_walk_threshold = red_cross_walk_threshold
        self.stop_duration = stop_duration

        # Tuned from the live camera feed for the physical red crosswalk tape.
        self.lower_red_cross_walk = np.array([108, 147, 110], dtype=np.uint8)
        self.upper_red_cross_walk = np.array([137, 255, 182], dtype=np.uint8)

        self.frame_size = (640, 480)
        self._perspective_matrix = self._build_perspective_matrix(*self.frame_size)

        self.red_cross_walk_pixels = 0
        self.red_cross_walk_raw_pixels = 0
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
        mask_red_cross_walk = cv2.inRange(
            img_hsv, self.lower_red_cross_walk, self.upper_red_cross_walk
        )
        self.red_cross_walk_raw_pixels = int(np.count_nonzero(mask_red_cross_walk))
        warped_img_red_cross_walk = cv2.warpPerspective(
            mask_red_cross_walk, self._perspective_matrix, self.frame_size
        )

        roi_start_y = int(self.frame_size[1] * 0.3)
        warped_red_cross_walk_roi = warped_img_red_cross_walk[roi_start_y:self.frame_size[1], :]
        self.red_cross_walk_pixels = int(np.count_nonzero(warped_red_cross_walk_roi))

        if not self.triggered_once and self.red_cross_walk_pixels > self.red_cross_walk_threshold:
            self.triggered_once = True
            self.stop_until = time.time() + self.stop_duration
            return True
        return False

    def should_stop(self):
        return time.time() < self.stop_until

    def get_red_cross_walk_pixels(self):
        return self.red_cross_walk_pixels

    def get_red_cross_walk_raw_pixels(self):
        return self.red_cross_walk_raw_pixels

    def get_green_pixels(self):
        return self.red_cross_walk_pixels

    def get_black_pixels(self):
        return self.red_cross_walk_pixels
