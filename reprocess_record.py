import argparse
import os
import cv2
import json
import pathlib
from datetime import datetime

# Import functions from main
import main
from detectors import FaceAnalyzer


def load_frames_from_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    ts = []
    start = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start is None:
            start = cv2.getTickCount()
        frames.append(frame)
        ts.append(idx)  # simple index timestamps
        idx += 1
    cap.release()
    return frames, ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-dir', required=True, help='Path to recorded outputs folder')
    parser.add_argument('--blur-threshold', type=float, default=30.0)
    parser.add_argument('--min-landmarks', type=int, default=40)
    parser.add_argument('--min-contrast', type=float, default=10.0)
    parser.add_argument('--min-brightness', type=float, default=30.0)
    parser.add_argument('--max-roll-deg', type=float, default=20.0)
    parser.add_argument('--save-bad-frames', action='store_true')
    args = parser.parse_args()

    outdir = args.record_dir
    front_path = os.path.join(outdir, 'front.mp4')
    side_path = os.path.join(outdir, 'side.mp4')
    if not os.path.exists(front_path) or not os.path.exists(side_path):
        raise SystemExit('front.mp4 or side.mp4 not found in record dir')

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
    except Exception:
        pass

    analyzer = FaceAnalyzer(device=device)

    quality_cfg = {
        'min_landmarks': args.min_landmarks,
        'blur_threshold': args.blur_threshold,
        'min_contrast': args.min_contrast,
        'min_brightness': args.min_brightness,
        'max_roll_deg': args.max_roll_deg
    }
    bad_save_dir = None
    if args.save_bad_frames:
        bad_save_dir = os.path.join(outdir, 'bad_frames_reprocess')
        pathlib.Path(bad_save_dir).mkdir(parents=True, exist_ok=True)

    print('Loading front video...')
    front_frames, front_ts = load_frames_from_video(front_path)
    print('Frames:', len(front_frames))
    print('Processing front frames with blur_threshold=', args.blur_threshold)
    front_per_frame, front_agg = main.process_frames(front_frames, front_ts, analyzer, sample_every=1, quality_cfg=quality_cfg, bad_save_dir=bad_save_dir)

    print('Loading side video...')
    side_frames, side_ts = load_frames_from_video(side_path)
    print('Frames:', len(side_frames))
    print('Processing side frames')
    side_per_frame, side_agg = main.process_frames(side_frames, side_ts, analyzer, sample_every=1, quality_cfg=quality_cfg, bad_save_dir=bad_save_dir)

    final_summary = {
        'front_summary': main.compute_summary(front_per_frame),
        'side_summary': main.compute_summary(side_per_frame)
    }
    final_path = os.path.join(outdir, 'final_reprocessed.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
    print('Saved final reprocessed summary to', final_path)

    # also save per-frame results
    out_json = {
        'front': {'per_frame': front_per_frame, 'aggregate': front_agg},
        'side': {'per_frame': side_per_frame, 'aggregate': side_agg}
    }
    out_json_path = os.path.join(outdir, 'results_reprocessed.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2)
    print('Saved reprocessed per-frame JSON to', out_json_path)
