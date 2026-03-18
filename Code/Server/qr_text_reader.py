import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read QR code text values from the RC car camera."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera frame height",
    )
    parser.add_argument(
        "--hflip",
        action="store_true",
        help="Flip image horizontally",
    )
    parser.add_argument(
        "--vflip",
        action="store_true",
        help="Flip image vertically",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["A", "B", "C", "D"],
        help="Expected QR text values",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Seconds between current_position logs in terminal",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without OpenCV display window",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Print decoded QR text even when it is not in targets",
    )
    parser.add_argument(
        "--disable-url-resolve",
        action="store_true",
        help="Disable auto HTTP resolve for URL QR payloads",
    )
    parser.add_argument(
        "--url-timeout",
        type=float,
        default=3.5,
        help="Timeout seconds for URL resolve requests",
    )
    parser.add_argument(
        "--frame-order",
        choices=["bgr", "rgb"],
        default="bgr",
        help="Color channel order returned by camera.capture_array()",
    )
    return parser.parse_args()


def setup_camera(width: int, height: int, hflip: bool, vflip: bool) -> Picamera2:
    camera = Picamera2()
    transform = Transform(hflip=1 if hflip else 0, vflip=1 if vflip else 0)
    config = camera.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
        transform=transform,
    )
    camera.configure(config)
    camera.start()
    return camera


def decode_candidates_opencv(detector: cv2.QRCodeDetector, img: np.ndarray):
    results = []

    ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img)
    if ok and decoded_info:
        for idx, text in enumerate(decoded_info):
            text = (text or "").strip()
            if not text:
                continue
            poly = None
            if points is not None and idx < len(points):
                poly = points[idx]
            results.append((text, poly))

    # Fallback for environments where detectAndDecodeMulti is less reliable.
    text_single, points_single, _ = detector.detectAndDecode(img)
    text_single = (text_single or "").strip()
    if text_single:
        results.append((text_single, points_single))

    return results


def detect_qr(detector: cv2.QRCodeDetector, frame_rgb: np.ndarray):
    return decode_candidates_opencv(detector, frame_rgb)


def looks_like_url(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("http://", "https://")):
        return True
    return "." in stripped and " " not in stripped


def normalize_url(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("http://", "https://")):
        return stripped
    return "https://" + stripped


def extract_target_strict(candidate_text: str, targets: set) -> str:
    text = candidate_text.strip().upper()
    if text in targets:
        return text

    patterns = [
        r'(?i)\b(?:position|pos|location|label|target|value|text)\s*[:=]\s*["\']?([A-D])["\']?\b',
        r'(?i)"(?:position|pos|location|label|target|value|text)"\s*:\s*"([A-D])"',
        r'(?i)<(?:span|div|p|h1|h2|h3)[^>]*>\s*([A-D])\s*</(?:span|div|p|h1|h2|h3)>',
    ]
    for pattern in patterns:
        m = re.search(pattern, candidate_text)
        if m:
            candidate = m.group(1).upper()
            if candidate in targets:
                return candidate
    return ""


def resolve_url_to_target(raw_text: str, targets: set, timeout: float, cache: dict) -> str:
    if raw_text in cache:
        return cache[raw_text]

    url = normalize_url(raw_text)
    found_target = ""
    try:
        parsed = urllib.parse.urlparse(url)
        url_tokens = [p for p in parsed.path.split("/") if p]
        query = urllib.parse.parse_qs(parsed.query)
        url_tokens += query.get("pos", []) + query.get("position", []) + query.get("location", [])
        for token in url_tokens:
            t = token.strip().upper()
            if t in targets:
                found_target = t
                break

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (QRPositionTracker)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            data = resp.read(200000)
            body_text = data.decode("utf-8", errors="ignore")

            if not found_target:
                parsed_final = urllib.parse.urlparse(final_url)
                final_tokens = [p for p in parsed_final.path.split("/") if p]
                final_query = urllib.parse.parse_qs(parsed_final.query)
                final_tokens += final_query.get("pos", []) + final_query.get("position", []) + final_query.get("location", [])
                for token in final_tokens:
                    t = token.strip().upper()
                    if t in targets:
                        found_target = t
                        break

            if not found_target:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "application/json" in content_type:
                    try:
                        obj = json.loads(body_text)
                        for key in ("position", "pos", "location", "label", "target", "value", "text"):
                            v = obj.get(key)
                            if isinstance(v, str) and v.strip().upper() in targets:
                                found_target = v.strip().upper()
                                break
                    except Exception:
                        pass

            if not found_target:
                found_target = extract_target_strict(f"{final_url}\n{body_text}", targets)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"[URL RESOLVE FAIL] {url} ({e})")
        found_target = ""

    cache[raw_text] = found_target
    if found_target:
        print(f"[URL RESOLVE] {raw_text} -> {found_target}")
    return found_target


def main():
    args = parse_args()
    targets = {text.strip().upper() for text in args.targets if text.strip()}
    current_position = None
    last_update_time = None
    last_log_time = 0.0
    url_target_cache = {}

    detector = cv2.QRCodeDetector()
    camera = setup_camera(args.width, args.height, args.hflip, args.vflip)
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    show_window = (not args.headless) and has_display

    print("QR position tracker started.")
    if show_window:
        print("Press 'q' to quit, 'l' to print current_position now.")
    else:
        print("Headless mode: no OpenCV window.")
    if targets:
        print(f"Valid positions: {sorted(targets)}")

    try:
        while True:
            frame_raw = camera.capture_array()
            if args.frame_order == "rgb":
                frame_for_detect = frame_raw
                frame_for_show = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            else:
                frame_for_show = frame_raw.copy()
                frame_for_detect = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            decoded_items = detect_qr(detector, frame_for_detect)

            for raw_text, poly in decoded_items:
                if args.debug_raw:
                    print(f"[RAW DETECT] {raw_text!r}")
                text = raw_text.strip().upper()

                if targets and text not in targets:
                    if (not args.disable_url_resolve) and looks_like_url(raw_text):
                        resolved_target = resolve_url_to_target(
                            raw_text=raw_text.strip(),
                            targets=targets,
                            timeout=args.url_timeout,
                            cache=url_target_cache,
                        )
                        if resolved_target:
                            text = resolved_target
                        else:
                            continue
                    else:
                        continue

                if text != current_position:
                    current_position = text
                    last_update_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[POSITION UPDATE] current_position = {current_position} ({last_update_time})")

                if poly is not None:
                    poly_int = np.array(poly).astype(int)
                    if poly_int.ndim == 3:
                        poly_int = poly_int[0]
                    if poly_int.ndim == 2 and len(poly_int) >= 4:
                        cv2.polylines(frame_for_show, [poly_int], True, (0, 255, 0), 2)
                        x, y = poly_int[0][0], poly_int[0][1] - 8
                        cv2.putText(
                            frame_for_show,
                            text,
                            (x, max(20, y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

            status = f"Current Position: {current_position if current_position else 'Unknown'}"
            cv2.putText(
                frame_for_show,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            now = time.time()
            if args.log_interval > 0 and (now - last_log_time) >= args.log_interval:
                print(f"[LOG] current_position = {current_position if current_position else 'Unknown'}")
                last_log_time = now

            if show_window:
                cv2.imshow("RCcar QR Reader", frame_for_show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("l"):
                    print(f"[LOG] current_position = {current_position if current_position else 'Unknown'}")

    finally:
        camera.stop()
        camera.close()
        if show_window:
            cv2.destroyAllWindows()

    print("\n=== Result ===")
    if current_position:
        print(f"Final current_position: {current_position}")
        if last_update_time:
            print(f"Last update time: {last_update_time}")
    else:
        print("No valid position QR detected.")


if __name__ == "__main__":
    main()
