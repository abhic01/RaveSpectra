import argparse
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from playwright.sync_api import sync_playwright


OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "rave_vision_features.csv"

def classify_view(frame_bgr, beams, mask):
    h, w = frame_bgr.shape[:2]

    beam_count = len(beams)
    brightness = float(np.mean(mask))

    # Lasers usually span wide portions of the image
    long_beams = [b for b in beams if b[4] > w * 0.20]

    # Ignore frames with very few laser-like structures
    if beam_count < 3 and brightness < 3:
        return "non_stage", 0.2

    # Strong sign of stage/laser view
    if len(long_beams) >= 3:
        return "stage", 0.85

    # Could be DJ booth / partial stage
    if beam_count >= 5:
        return "partial_stage", 0.6

    return "unknown", 0.4

def detect_lasers(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Bright + saturated = likely laser/stage light pixels.
    lower = np.array([0, 80, 170])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=80,
        maxLineGap=20,
    )

    beams = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            beams.append((x1, y1, x2, y2, length, angle))

    return beams, mask


def screenshot_to_bgr(screenshot_bytes):
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    frame_rgb = np.array(img)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def analyze_url(url, seconds=60, fps=2, headed=True):
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = []
    interval = 1 / fps

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=[
                "--autoplay-policy=no-user-gesture-required",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        page.wait_for_timeout(5000)

        # Try to play the video.
        page.evaluate("""
            () => {
                const video = document.querySelector('video');
                if (video) {
                    video.muted = true;
                    video.play();
                }
            }
        """)

        # Prefer screenshotting the actual video element instead of the whole page.
        video = page.locator("video").first
        video.wait_for(timeout=20_000)

        start = time.time()
        frame_index = 0

        while time.time() - start < seconds:
            current_time = round(time.time() - start, 3)

            screenshot_bytes = video.screenshot()
            frame = screenshot_to_bgr(screenshot_bytes)

            beams, mask = detect_lasers(frame)

            view_type, stage_confidence = classify_view(frame, beams, mask)

            rows.append({
                "time": current_time,
                "frame_index": frame_index,
                "view_type": view_type,
                "stage_confidence": stage_confidence,
                "analyze_frame": view_type in ["stage", "partial_stage"],
                "beam_count": len(beams),
                "avg_beam_length": float(np.mean([b[4] for b in beams])) if beams else 0,
                "avg_beam_angle": float(np.mean([b[5] for b in beams])) if beams else 0,
                "laser_brightness": float(np.mean(mask)),
            })

            frame_index += 1
            time.sleep(interval)

        browser.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube/concert video URL")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()

    analyze_url(
        url=args.url,
        seconds=args.seconds,
        fps=args.fps,
        headed=not args.headless,
    )
    