"""Quick GPU + face-alignment smoke test.

Usage:
  - With webcam: python tests/run_gpu_test.py
  - With image:  python tests/run_gpu_test.py --image C:\path\to\face.jpg

This script checks torch + CUDA availability, imports face_alignment, captures a frame (or reads image),
runs FaceAnalyzer.get_landmarks once and prints a short summary.
"""
import argparse
import cv2
import sys
import pathlib

# Ensure project root is importable when running the test from the tests/ folder
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Optional image path to test instead of webcam')
    args = parser.parse_args()

    try:
        import torch
        print('torch available:', torch.__version__, 'cuda:', torch.cuda.is_available())
    except Exception as e:
        print('torch import failed:', e)
        sys.exit(1)

    try:
        import face_alignment
        print('face_alignment available')
    except Exception as e:
        print('face_alignment import failed:', e)
        sys.exit(1)

    # Grab image
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print('Could not read image:', args.image)
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Cannot open webcam')
            sys.exit(1)
        ret, img = cap.read()
        cap.release()
        if not ret:
            print('Failed to capture frame')
            sys.exit(1)

    # Run detector
    try:
        from detectors import FaceAnalyzer
        import numpy as np
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = FaceAnalyzer(device=device)
        lms = fa.get_landmarks(img)
        if lms is None:
            print('No landmarks detected')
            sys.exit(2)
        print('Detected', len(lms), 'landmarks. Sample:', lms[:5])
    except Exception as e:
        print('Error running FaceAnalyzer:', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
