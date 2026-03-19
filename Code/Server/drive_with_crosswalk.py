import time

import cv2
import numpy as np

from crosswalk_detector import CrosswalkDetector
from drive import AutoRaceLaneDrive, build_parser


class AutoRaceLaneDriveWithCrosswalk:
    def __init__(self, args):
        self.driver = AutoRaceLaneDrive(args)
        self.crosswalk_detector = CrosswalkDetector(
            red_cross_walk_threshold=args.red_cross_walk_threshold,
            stop_duration=args.crosswalk_stop_duration,
        )
        self.crosswalk_active = False
        self.last_crosswalk_log_time = 0.0

    def close(self):
        self.driver.close()

    def detect_crosswalk(self, frame_bgr: np.ndarray) -> bool:
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return False
        return self.crosswalk_detector.process_jpeg_frame(encoded.tobytes())

    def drive(self):
        print("Auto-Race lane drive with red crosswalk stop started.")
        if self.driver.show_window:
            print("Press 'q' to quit.")
        else:
            print("Headless mode.")

        frame_dt = 1.0 / max(1.0, float(self.driver.args.fps))

        try:
            while True:
                t0 = time.time()
                frame_raw = self.driver.camera.capture_array()
                if self.driver.args.frame_order == "rgb":
                    frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_raw.copy()

                frame_bgr = cv2.resize(frame_bgr, (self.driver.width, self.driver.height))

                detected_now = self.detect_crosswalk(frame_bgr)
                now = time.time()
                if now - self.last_crosswalk_log_time >= 0.2:
                    print(
                        "[CROSSWALK] raw_pixels={} warped_pixels={} threshold={}".format(
                            self.crosswalk_detector.get_red_cross_walk_raw_pixels(),
                            self.crosswalk_detector.get_red_cross_walk_pixels(),
                            self.crosswalk_detector.red_cross_walk_threshold,
                        )
                    )
                    self.last_crosswalk_log_time = now

                if detected_now:
                    print(
                        "[CROSSWALK] detected: red_cross_walk_pixels={}, stopping for {:.1f}s".format(
                            self.crosswalk_detector.get_red_cross_walk_pixels(),
                            self.crosswalk_detector.stop_duration,
                        )
                    )

                if self.crosswalk_detector.should_stop():
                    if not self.crosswalk_active:
                        remaining = max(0.0, self.crosswalk_detector.stop_until - time.time())
                        print(f"[CROSSWALK] stop active, remaining={remaining:.1f}s")
                        self.crosswalk_active = True
                    self.driver.pid.reset()
                    self.driver.stop()
                    if self.driver.show_window:
                        cv2.imshow("AutoRace Lane Drive", frame_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    elapsed = time.time() - t0
                    if elapsed < frame_dt:
                        time.sleep(frame_dt - elapsed)
                    continue

                if self.crosswalk_active:
                    self.crosswalk_active = False
                    print("[CROSSWALK] stop released, resume lane drive")

                result = self.driver.process_frame(frame_bgr)
                msg = result["msg"]
                x_location = result["x_location"]
                target_x = result["target_x"]
                center_target = result["center_target"]
                dbg_warp = result["debug_warp"]

                if result["stop_required"]:
                    self.driver.stop()
                elif result["lane_detected"]:
                    self.driver.apply_drive_command(result["cmd_left"], result["cmd_right"])

                if self.driver.show_window:
                    vis = result["frame_bgr"].copy()
                    y = int(self.driver.height * 0.8)
                    cv2.line(
                        vis,
                        (self.driver.width // 2, int(self.driver.height * self.driver.args.roi_top_ratio)),
                        (self.driver.width // 2, self.driver.height),
                        (255, 0, 0),
                        2,
                    )
                    bias_x = int(max(0, min(self.driver.width - 1, center_target)))
                    cv2.line(
                        vis,
                        (bias_x, int(self.driver.height * self.driver.args.roi_top_ratio)),
                        (bias_x, self.driver.height),
                        (0, 255, 255),
                        1,
                    )
                    cv2.circle(vis, (x_location, y), 8, (0, 0, 255), -1)
                    cv2.circle(vis, (target_x, y - 25), 7, (0, 165, 255), -1)
                    cv2.putText(
                        vis,
                        msg,
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )
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


def main():
    parser = build_parser()
    parser.description = "Auto-Race style lane tracking with red crosswalk stop"
    parser.add_argument("--red-cross-walk-threshold", type=int, default=8000)
    parser.add_argument("--crosswalk-stop-duration", type=float, default=5.0)
    args = parser.parse_args()
    args.frame_order = "rgb"
    driver = AutoRaceLaneDriveWithCrosswalk(args)
    driver.drive()


if __name__ == "__main__":
    main()
