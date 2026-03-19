import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2


WINDOW_NAME = "Green HSV Tuner"
MASK_NAME = "Green Mask"
FILTERED_NAME = "Green Filtered"


def nothing(_value):
    return


def create_trackbars():
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(MASK_NAME)
    cv2.namedWindow(FILTERED_NAME)

    # Start from the current green crosswalk defaults.
    cv2.createTrackbar("Low H", WINDOW_NAME, 25, 179, nothing)
    cv2.createTrackbar("High H", WINDOW_NAME, 95, 179, nothing)
    cv2.createTrackbar("Low S", WINDOW_NAME, 40, 255, nothing)
    cv2.createTrackbar("High S", WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar("Low V", WINDOW_NAME, 80, 255, nothing)
    cv2.createTrackbar("High V", WINDOW_NAME, 255, 255, nothing)


def read_trackbar_values():
    low_h = cv2.getTrackbarPos("Low H", WINDOW_NAME)
    high_h = cv2.getTrackbarPos("High H", WINDOW_NAME)
    low_s = cv2.getTrackbarPos("Low S", WINDOW_NAME)
    high_s = cv2.getTrackbarPos("High S", WINDOW_NAME)
    low_v = cv2.getTrackbarPos("Low V", WINDOW_NAME)
    high_v = cv2.getTrackbarPos("High V", WINDOW_NAME)

    lower = np.array(
        [min(low_h, high_h), min(low_s, high_s), min(low_v, high_v)],
        dtype=np.uint8,
    )
    upper = np.array(
        [max(low_h, high_h), max(low_s, high_s), max(low_v, high_v)],
        dtype=np.uint8,
    )
    return lower, upper


def main():
    camera = Picamera2()
    transform = Transform()
    config = camera.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        transform=transform,
    )
    camera.configure(config)
    camera.start()

    create_trackbars()
    print("Press 'q' to quit.")
    print("Press 'p' to print the current HSV values.")

    try:
        while True:
            frame_rgb = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            lower, upper = read_trackbar_values()
            mask = cv2.inRange(hsv, lower, upper)
            filtered = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

            overlay = frame_bgr.copy()
            text = (
                f"LOW=({lower[0]}, {lower[1]}, {lower[2]}) "
                f"HIGH=({upper[0]}, {upper[1]}, {upper[2]})"
            )
            cv2.putText(
                overlay,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, overlay)
            cv2.imshow(MASK_NAME, mask)
            cv2.imshow(FILTERED_NAME, filtered)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                print(
                    "lower_green = np.array([{}, {}, {}], dtype=np.uint8)".format(
                        lower[0], lower[1], lower[2]
                    )
                )
                print(
                    "upper_green = np.array([{}, {}, {}], dtype=np.uint8)".format(
                        upper[0], upper[1], upper[2]
                    )
                )
    finally:
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
