#!/usr/bin/env python3
"""Check that all video_0.mp4 have dominant red channel (PPG finger video)."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

def is_red_dominant(video_path: str, sample_frame_ratio: float = 0.5) -> bool:
    """Sample one frame from video; return True if mean R > mean G and mean R > mean B."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return False
    idx = int((n - 1) * sample_frame_ratio)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    # OpenCV BGR: B=0, G=1, R=2
    mean_b, mean_g, mean_r = frame.mean(axis=(0, 1))
    return mean_r > mean_g and mean_r > mean_b

def main():
    samples_dir = Path(__file__).resolve().parent / "training_data" / "samples"
    if not samples_dir.exists():
        print("training_data/samples not found", file=sys.stderr)
        sys.exit(1)
    pairs = sorted(samples_dir.iterdir())
    not_red = []
    no_video = []
    for pair_dir in pairs:
        if not pair_dir.is_dir():
            continue
        v0 = pair_dir / "video_0.mp4"
        if not v0.exists():
            no_video.append(pair_dir.name)
            continue
        if not is_red_dominant(str(v0)):
            not_red.append(pair_dir.name)
    if no_video:
        print("No video_0.mp4:", no_video)
    if not_red:
        print("video_0 NOT red-dominant:", not_red)
        sys.exit(1)
    print("OK: all video_0 are red-dominant (R > G and R > B).")

if __name__ == "__main__":
    main()
