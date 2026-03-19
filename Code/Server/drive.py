import argparse
import os
import time

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2

from motor import Ordinary_Car


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def reset(self):
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def control(self, cte: float) -> float:
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte
        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error


class AutoRaceLaneDrive:
    def __init__(self, args):
        self.args = args
        self.motor = Ordinary_Car()
        self.pid = PID(args.kp, args.ki, args.kd)

        self.camera = Picamera2()
        transform = Transform(hflip=1 if args.hflip else 0, vflip=1 if args.vflip else 0)
        config = self.camera.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            transform=transform,
        )
        self.camera.configure(config)
        self.camera.start()

        self.width = args.width
        self.height = args.height
        self.last_x_location = self.width // 2
        self.last_target_x = self.width // 2
        self.last_lane_width = int(self.width * 0.45)
        self.lost_count = 0

        self.has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.show_window = (not args.headless) and self.has_display

        self.warp_src = np.float32(
            [
                [args.warp_src_x1, args.warp_src_y1],
                [args.warp_src_x2, args.warp_src_y2],
                [args.warp_src_x3, args.warp_src_y3],
                [args.warp_src_x4, args.warp_src_y4],
            ]
        )
        self.warp_dst = np.float32(
            [
                [self.width * 0.30, self.height],
                [self.width * 0.30, 0],
                [self.width * 0.70, 0],
                [self.width * 0.70, self.height],
            ]
        )
        self.warp_mtx = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        print(f"[WARP SRC] {self.warp_src.tolist()}")

    def stop(self):
        self.motor.set_motor_model(0, 0, 0, 0)

    def close(self):
        try:
            self.stop()
        finally:
            self.camera.stop()
            self.camera.close()
            self.motor.close()
            if self.show_window:
                cv2.destroyAllWindows()

    def make_binary(self, frame_bgr: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.args.morph_kernel, self.args.morph_kernel), np.uint8)

        if self.args.binary_mode == "black":
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0], dtype=np.uint8)
            upper_black = np.array(
                [180, self.args.black_s_max, self.args.black_v_max], dtype=np.uint8
            )
            mask = cv2.inRange(hsv, lower_black, upper_black)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            return mask

        # Legacy Auto-Race style binarization.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.args.gaussian_ksize, self.args.gaussian_ksize), 0)
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.args.adaptive_block_size,
            self.args.adaptive_c,
        )
        edges = cv2.Canny(adaptive, self.args.canny_low, self.args.canny_high)
        binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def apply_roi(self, binary: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(binary)
        y_top = int(self.height * self.args.roi_top_ratio)
        pts = np.array(
            [[
                (0, self.height - 1),
                (int(self.width * 0.15), y_top),
                (int(self.width * 0.85), y_top),
                (self.width - 1, self.height - 1),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(binary, mask)

    def warp(self, binary_roi: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(binary_roi, self.warp_mtx, (self.width, self.height))

    def slide_window(self, binary_warped: np.ndarray):
        h, w = binary_warped.shape
        histogram = np.sum(binary_warped[h // 2 :, :], axis=0)
        midpoint = w // 2

        leftx_base = int(np.argmax(histogram[:midpoint]))
        rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        if histogram[leftx_base] < self.args.hist_min_peak:
            leftx_base = None
        if histogram[rightx_base] < self.args.hist_min_peak:
            rightx_base = None

        nwindows = self.args.nwindows
        window_h = h // nwindows
        margin = self.args.window_margin
        minpix = self.args.window_minpix

        nz_y, nz_x = binary_warped.nonzero()
        nz_y = np.array(nz_y)
        nz_x = np.array(nz_x)

        left_inds = []
        right_inds = []
        leftx_current = leftx_base
        rightx_current = rightx_base

        debug = np.dstack((binary_warped, binary_warped, binary_warped))

        for win in range(nwindows):
            y_low = h - (win + 1) * window_h
            y_high = h - win * window_h

            if leftx_current is not None:
                lx_low = leftx_current - margin
                lx_high = leftx_current + margin
                cv2.rectangle(debug, (lx_low, y_low), (lx_high, y_high), (0, 255, 0), 1)
                good_left = (
                    (nz_y >= y_low)
                    & (nz_y < y_high)
                    & (nz_x >= lx_low)
                    & (nz_x < lx_high)
                ).nonzero()[0]
                if len(good_left) > minpix:
                    leftx_current = int(np.mean(nz_x[good_left]))
                left_inds.append(good_left)

            if rightx_current is not None:
                rx_low = rightx_current - margin
                rx_high = rightx_current + margin
                cv2.rectangle(debug, (rx_low, y_low), (rx_high, y_high), (255, 0, 0), 1)
                good_right = (
                    (nz_y >= y_low)
                    & (nz_y < y_high)
                    & (nz_x >= rx_low)
                    & (nz_x < rx_high)
                ).nonzero()[0]
                if len(good_right) > minpix:
                    rightx_current = int(np.mean(nz_x[good_right]))
                right_inds.append(good_right)

        try:
            left_inds = np.concatenate(left_inds)
        except Exception:
            left_inds = np.array([], dtype=np.int32)
        try:
            right_inds = np.concatenate(right_inds)
        except Exception:
            right_inds = np.array([], dtype=np.int32)

        left_detected = len(left_inds) > self.args.min_lane_pixels
        right_detected = len(right_inds) > self.args.min_lane_pixels

        x_location = self.last_x_location
        target_x = self.last_target_x
        left_mean = None
        right_mean = None
        left_fit = None
        right_fit = None

        if left_detected:
            left_mean = int(np.mean(nz_x[left_inds]))
            if len(left_inds) >= self.args.min_fit_pixels:
                left_fit = np.polyfit(nz_y[left_inds], nz_x[left_inds], 2)
        if right_detected:
            right_mean = int(np.mean(nz_x[right_inds]))
            if len(right_inds) >= self.args.min_fit_pixels:
                right_fit = np.polyfit(nz_y[right_inds], nz_x[right_inds], 2)

        if left_detected and right_detected:
            lane_width = max(80, right_mean - left_mean)
            self.last_lane_width = lane_width
            x_location = (left_mean + right_mean) // 2
            # Lookahead center for early turning.
            y_near = int(h * self.args.lookahead_near_ratio)
            y_far = int(h * self.args.lookahead_far_ratio)
            if left_fit is not None and right_fit is not None:
                left_near = np.polyval(left_fit, y_near)
                right_near = np.polyval(right_fit, y_near)
                left_far = np.polyval(left_fit, y_far)
                right_far = np.polyval(right_fit, y_far)
                center_near = (left_near + right_near) / 2.0
                center_far = (left_far + right_far) / 2.0
                target_x = int(
                    (1.0 - self.args.lookahead_weight) * center_near
                    + self.args.lookahead_weight * center_far
                )
            else:
                target_x = x_location
        elif left_detected:
            offset = int(max(self.args.single_lane_offset, self.last_lane_width * 0.5))
            x_location = int(left_mean + offset)
            if left_fit is not None:
                y_far = int(h * self.args.lookahead_far_ratio)
                target_x = int(np.polyval(left_fit, y_far) + offset)
            else:
                target_x = x_location
        elif right_detected:
            offset = int(max(self.args.single_lane_offset, self.last_lane_width * 0.5))
            x_location = int(right_mean - offset)
            if right_fit is not None:
                y_far = int(h * self.args.lookahead_far_ratio)
                target_x = int(np.polyval(right_fit, y_far) - offset)
            else:
                target_x = x_location

        x_location = int(
            self.args.x_smooth_alpha * self.last_x_location
            + (1.0 - self.args.x_smooth_alpha) * x_location
        )
        x_location = int(clamp(x_location, 0, w - 1))
        self.last_x_location = x_location
        target_x = int(
            self.args.target_smooth_alpha * self.last_target_x
            + (1.0 - self.args.target_smooth_alpha) * target_x
        )
        target_x = int(clamp(target_x, 0, w - 1))
        self.last_target_x = target_x

        # Paint detected lane pixels for debugging.
        if left_detected:
            debug[nz_y[left_inds], nz_x[left_inds]] = [0, 255, 0]
        if right_detected:
            debug[nz_y[right_inds], nz_x[right_inds]] = [255, 0, 0]

        return debug, x_location, target_x, left_detected, right_detected

    def compute_wheel_speed(self, control_x: int):
        center_target = (self.width / 2.0) + self.args.center_bias
        cte = center_target - control_x
        steer = self.pid.control(cte) * self.args.steer_gain * self.args.steer_sign

        # Separate base speeds for straight and turn sections.
        is_turn = (abs(cte) >= self.args.turn_cte_threshold) or (abs(steer) >= self.args.turn_steer_threshold)
        base_target = self.args.turn_base_speed if is_turn else self.args.straight_base_speed
        if is_turn:
            steer *= self.args.turn_steer_mult
        slowdown = min(self.args.max_turn_slowdown, abs(steer) * self.args.turn_slowdown_gain)
        base = max(self.args.min_base_speed, base_target - slowdown)

        left = int(base - steer)
        right = int(base + steer)

        if is_turn and self.args.turn_diff_boost > 0:
            boost = int(self.args.turn_diff_boost)
            if steer > 0:
                left -= boost
                right += boost
            elif steer < 0:
                left += boost
                right -= boost

        left = int(clamp(left, -self.args.max_speed, self.args.max_speed))
        right = int(clamp(right, -self.args.max_speed, self.args.max_speed))
        return left, right, cte, steer, int(base), center_target, is_turn

    def drive(self):
        print("Auto-Race style lane drive started.")
        if self.show_window:
            print("Press 'q' to quit.")
        else:
            print("Headless mode.")

        frame_dt = 1.0 / max(1.0, float(self.args.fps))
        try:
            while True:
                t0 = time.time()
                frame_raw = self.camera.capture_array()
                if self.args.frame_order == "rgb":
                    frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_raw.copy()

                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
                binary = self.make_binary(frame_bgr)
                roi = self.apply_roi(binary)
                warped = self.warp(roi)
                if np.count_nonzero(warped) < self.args.min_warp_pixels:
                    warped = roi.copy()
                dbg_warp, x_location, target_x, left_ok, right_ok = self.slide_window(warped)

                center_target = (self.width / 2.0) + self.args.center_bias
                if not left_ok and not right_ok:
                    self.lost_count += 1
                    if self.lost_count >= self.args.lost_stop_frames:
                        self.stop()
                    msg = f"lane_lost={self.lost_count}"
                else:
                    self.lost_count = 0
                    left, right, cte, steer, base, center_target, is_turn = self.compute_wheel_speed(target_x)
                    # Match your board direction convention.
                    cmd_left, cmd_right = left, right
                    if self.args.swap_left_right:
                        cmd_left, cmd_right = cmd_right, cmd_left
                    self.motor.set_motor_model(-cmd_left, -cmd_left, -cmd_right, -cmd_right)
                    msg = (
                        f"x={x_location} tx={target_x} ctr={center_target:.1f} "
                        f"cte={cte:.1f} steer={steer:.1f} base={base} mode={'TURN' if is_turn else 'STRAIGHT'} "
                        f"L={left} R={right} cmdL={cmd_left} cmdR={cmd_right}"
                    )

                if self.show_window:
                    vis = frame_bgr.copy()
                    y = int(self.height * 0.8)
                    cv2.line(vis, (self.width // 2, int(self.height * self.args.roi_top_ratio)),
                             (self.width // 2, self.height), (255, 0, 0), 2)
                    bias_x = int(clamp(center_target, 0, self.width - 1))
                    cv2.line(vis, (bias_x, int(self.height * self.args.roi_top_ratio)),
                             (bias_x, self.height), (0, 255, 255), 1)
                    cv2.circle(vis, (x_location, y), 8, (0, 0, 255), -1)
                    cv2.circle(vis, (target_x, y - 25), 7, (0, 165, 255), -1)
                    cv2.putText(vis, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    cv2.imshow("AutoRace Lane Drive", vis)
                    cv2.imshow("AutoRace Binary Warp", dbg_warp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                else:
                    print(f"[LANE] {msg}")

                elapsed = time.time() - t0
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
        finally:
            self.close()


def build_parser():
    p = argparse.ArgumentParser(description="Auto-Race style lane tracking for RC car")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--hflip", action="store_true")
    p.add_argument("--vflip", action="store_true")
    p.add_argument("--frame-order", choices=["bgr", "rgb"], default="bgr")
    p.add_argument("--headless", action="store_true")

    # Vision pipeline parameters.
    p.add_argument("--binary-mode", choices=["black", "adaptive"], default="black")
    p.add_argument("--black-s-max", type=int, default=70)
    p.add_argument("--black-v-max", type=int, default=70)
    p.add_argument("--gaussian-ksize", type=int, default=9)
    p.add_argument("--adaptive-block-size", type=int, default=11)
    p.add_argument("--adaptive-c", type=float, default=-2.0)
    p.add_argument("--canny-low", type=int, default=50)
    p.add_argument("--canny-high", type=int, default=200)
    p.add_argument("--morph-kernel", type=int, default=3)
    p.add_argument("--roi-top-ratio", type=float, default=0.45)

    # Perspective warp source points.
    # Tuned defaults for 640x480 front camera to include both black lanes.
    p.add_argument("--warp-src-x1", type=float, default=80)
    p.add_argument("--warp-src-y1", type=float, default=470)
    p.add_argument("--warp-src-x2", type=float, default=250)
    p.add_argument("--warp-src-y2", type=float, default=250)
    p.add_argument("--warp-src-x3", type=float, default=445)
    p.add_argument("--warp-src-y3", type=float, default=250)
    p.add_argument("--warp-src-x4", type=float, default=595)
    p.add_argument("--warp-src-y4", type=float, default=470)
    p.add_argument("--min-warp-pixels", type=int, default=300)

    # Sliding window parameters.
    p.add_argument("--nwindows", type=int, default=12)
    p.add_argument("--window-margin", type=int, default=60)
    p.add_argument("--window-minpix", type=int, default=40)
    p.add_argument("--hist-min-peak", type=int, default=50)
    p.add_argument("--min-lane-pixels", type=int, default=80)
    p.add_argument("--min-fit-pixels", type=int, default=120)
    p.add_argument("--single-lane-offset", type=int, default=100)
    p.add_argument("--x-smooth-alpha", type=float, default=0.50)
    p.add_argument("--target-smooth-alpha", type=float, default=0.35)
    p.add_argument("--lookahead-near-ratio", type=float, default=0.88)
    p.add_argument("--lookahead-far-ratio", type=float, default=0.25)
    p.add_argument("--lookahead-weight", type=float, default=0.85)

    # Drive parameters.
    p.add_argument("--base-speed", type=int, default=550)
    p.add_argument("--straight-base-speed", type=int, default=550)
    p.add_argument("--turn-base-speed", type=int, default=300)
    p.add_argument("--min-base-speed", type=int, default=300)
    p.add_argument("--max-speed", type=int, default=900)
    p.add_argument("--center-bias", type=float, default=-20.0)
    p.add_argument("--kp", type=float, default=1.8)
    p.add_argument("--ki", type=float, default=0.0004)
    p.add_argument("--kd", type=float, default=0.75)
    p.add_argument("--steer-gain", type=float, default=2.0)
    p.add_argument("--steer-sign", type=int, choices=[-1, 1], default=1)
    p.add_argument("--swap-left-right", action="store_true")
    p.add_argument("--turn-slowdown-gain", type=float, default=0.08)
    p.add_argument("--max-turn-slowdown", type=int, default=180)
    p.add_argument("--turn-cte-threshold", type=float, default=24.0)
    p.add_argument("--turn-steer-threshold", type=float, default=26.0)
    p.add_argument("--turn-steer-mult", type=float, default=1.6)
    p.add_argument("--turn-diff-boost", type=int, default=150)
    p.add_argument("--lost-stop-frames", type=int, default=6)
    return p


def main():
    args = build_parser().parse_args()
    driver = AutoRaceLaneDrive(args)
    driver.drive()


if __name__ == "__main__":
    main()
