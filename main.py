import argparse
import json
import cv2
import os
import time
import pathlib
import subprocess
import shutil
from datetime import datetime
from detectors import FaceAnalyzer
from utils import variance_of_laplacian, brightness_and_contrast, save_image_jpg, compute_basic_metrics

# Detect torch and CUDA availability (optional)
try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False


def capture_two_frames(analyzer: FaceAnalyzer = None, live_overlay=True, overlay_every=5, overlay_scale=0.5):
    """Quick capture: press SPACE to capture front, then SPACE for side.
    When an analyzer is provided and live_overlay is True, draw landmarks and quick
    image-quality metrics on the preview similar to recording mode.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam')

    print('Press SPACE to capture FRONT frame')
    front = None
    frame_idx = 0
    last_overlay_lm = None
    last_overlay_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        disp = frame.copy()

        # compute quick metrics
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_var = variance_of_laplacian(gray)
            mean_b, std_c = brightness_and_contrast(gray)
        except Exception:
            blur_var = 0.0
            mean_b, std_c = 0.0, 0.0

        status = 'OK'
        reasons = []
        if blur_var < 100.0:
            reasons.append('blur')
            status = 'BAD'
        if std_c < 10.0:
            reasons.append('low_contrast')
            status = 'BAD'
        if mean_b < 30.0:
            reasons.append('dark')
            status = 'BAD'

        cv2.putText(disp, f'Sharpness: {blur_var:.1f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(disp, f'Brightness: {mean_b:.1f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(disp, f'Contrast: {std_c:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(disp, f'Quality: {status}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == 'OK' else (0, 0, 255), 2)
        if reasons:
            cv2.putText(disp, 'Reasons: ' + ','.join(reasons), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # occasionally overlay landmarks when analyzer present
        do_overlay = live_overlay and analyzer is not None and (frame_idx % max(1, overlay_every) == 0)
        if do_overlay:
            try:
                if 0.0 < overlay_scale < 1.0:
                    small = cv2.resize(frame, (0, 0), fx=overlay_scale, fy=overlay_scale)
                else:
                    small = frame
                lm = analyzer.get_landmarks(small)
                if lm and len(lm) > 0:
                    if 0.0 < overlay_scale < 1.0:
                        scaled_lm = [(int(x / overlay_scale), int(y / overlay_scale)) for (x, y) in lm]
                    else:
                        scaled_lm = [(int(x), int(y)) for (x, y) in lm]
                    last_overlay_lm = scaled_lm
                    last_overlay_time = now
            except Exception:
                last_overlay_lm = None

        if last_overlay_lm is not None and (now - last_overlay_time) < 2.0:
            for (x, y) in last_overlay_lm:
                cv2.circle(disp, (int(x), int(y)), 1, (0, 255, 0), -1)

        cv2.putText(disp, 'Press SPACE to capture FRONT', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Capture FRONT', disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit('User cancelled')
        if k == 32:  # SPACE
            front = frame.copy()
            break
    cv2.destroyWindow('Capture FRONT')

    input('Turn to SIDE and press Enter to start SIDE capture...')
    print('Press SPACE to capture SIDE frame')
    side = None
    frame_idx = 0
    last_overlay_lm = None
    last_overlay_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        disp = frame.copy()

        # compute quick metrics
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_var = variance_of_laplacian(gray)
            mean_b, std_c = brightness_and_contrast(gray)
        except Exception:
            blur_var = 0.0
            mean_b, std_c = 0.0, 0.0

        status = 'OK'
        reasons = []
        if blur_var < 100.0:
            reasons.append('blur')
            status = 'BAD'
        if std_c < 10.0:
            reasons.append('low_contrast')
            status = 'BAD'
        if mean_b < 30.0:
            reasons.append('dark')
            status = 'BAD'

        cv2.putText(disp, f'Sharpness: {blur_var:.1f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(disp, f'Brightness: {mean_b:.1f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(disp, f'Contrast: {std_c:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(disp, f'Quality: {status}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == 'OK' else (0, 0, 255), 2)
        if reasons:
            cv2.putText(disp, 'Reasons: ' + ','.join(reasons), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # occasionally overlay landmarks when analyzer present
        do_overlay = live_overlay and analyzer is not None and (frame_idx % max(1, overlay_every) == 0)
        if do_overlay:
            try:
                if 0.0 < overlay_scale < 1.0:
                    small = cv2.resize(frame, (0, 0), fx=overlay_scale, fy=overlay_scale)
                else:
                    small = frame
                lm = analyzer.get_landmarks(small)
                if lm and len(lm) > 0:
                    if 0.0 < overlay_scale < 1.0:
                        scaled_lm = [(int(x / overlay_scale), int(y / overlay_scale)) for (x, y) in lm]
                    else:
                        scaled_lm = [(int(x), int(y)) for (x, y) in lm]
                    last_overlay_lm = scaled_lm
                    last_overlay_time = now
            except Exception:
                last_overlay_lm = None

        if last_overlay_lm is not None and (now - last_overlay_time) < 2.0:
            for (x, y) in last_overlay_lm:
                cv2.circle(disp, (int(x), int(y)), 1, (0, 255, 0), -1)

        cv2.putText(disp, 'Press SPACE to capture SIDE', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Capture SIDE', disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit('User cancelled')
        if k == 32:
            side = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()
    return front, side


def record_segment(duration_sec, fps: int = 15, preview_window_title='Recording', analyzer: FaceAnalyzer = None,
                   live_overlay=True, overlay_every=5, overlay_scale=0.5):
    """Record for duration_sec seconds, return frames list and timestamps and measured fps.
    If live_overlay and analyzer are provided, draw on-screen quality metrics and occasional landmark overlays.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam')
    frames = []
    timestamps = []
    start = time.time()
    frame_count = 0
    print(f'Starting recording ({duration_sec}s) - preview: {preview_window_title}')

    frame_idx = 0
    last_overlay_lm = None
    last_overlay_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        frames.append(frame.copy())
        timestamps.append(now - start)
        frame_count += 1

        overlay = frame.copy()
        # compute quick metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_var = variance_of_laplacian(gray)
        mean_b, std_c = brightness_and_contrast(gray)

        status = 'OK'
        reasons = []
        # simple visual cues based on original defaults
        if blur_var < 100.0:
            reasons.append('blur')
            status = 'BAD'
        if std_c < 10.0:
            reasons.append('low_contrast')
            status = 'BAD'
        if mean_b < 30.0:
            reasons.append('dark')
            status = 'BAD'

        # draw text metrics
        cv2.putText(overlay, f'Sharpness: {blur_var:.1f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(overlay, f'Brightness: {mean_b:.1f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(overlay, f'Contrast: {std_c:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(overlay, f'Quality: {status}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == 'OK' else (0, 0, 255), 2)
        if reasons:
            cv2.putText(overlay, 'Reasons: ' + ','.join(reasons), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # run lightweight landmark overlay occasionally
        do_overlay = live_overlay and analyzer is not None and (frame_idx % max(1, overlay_every) == 0)
        if do_overlay:
            try:
                # use a downscaled copy to speed up detection
                if 0.0 < overlay_scale < 1.0:
                    small = cv2.resize(frame, (0, 0), fx=overlay_scale, fy=overlay_scale)
                else:
                    small = frame
                lm = analyzer.get_landmarks(small)
                if lm and len(lm) > 0:
                    # if scaled, remap points
                    if 0.0 < overlay_scale < 1.0:
                        scaled_lm = [(int(x / overlay_scale), int(y / overlay_scale)) for (x, y) in lm]
                    else:
                        scaled_lm = [(int(x), int(y)) for (x, y) in lm]
                    last_overlay_lm = scaled_lm
                    last_overlay_time = now
            except Exception:
                last_overlay_lm = None

        # draw landmarks from last successful overlay (if recent)
        if last_overlay_lm is not None and (now - last_overlay_time) < 2.0:
            for (x, y) in last_overlay_lm:
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)

        cv2.imshow(preview_window_title, overlay)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if now - start >= duration_sec:
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start
    measured_fps = frame_count / elapsed if elapsed > 0 else fps
    return frames, timestamps, measured_fps


def save_video(frames, out_path, fps=15):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def ffmpeg_write_video(frames, out_path, fps=15, crf=18, preset='veryfast', use_nvenc=False):
    """Write frames (BGR numpy arrays) to out_path using ffmpeg via stdin streaming.
    Raises RuntimeError if ffmpeg is not found or if ffmpeg exits with non-zero code.
    """
    if not frames:
        return
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found in PATH')
    h, w = frames[0].shape[:2]
    size_str = f'{w}x{h}'
    # build command
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', size_str, '-r', str(fps), '-i', '-']
    if use_nvenc:
        # try NVENC encoder
        cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr_hq', '-cq', str(max(1, int(crf)))]
    else:
        cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf)]
    # ensure yuv420p for compatibility
    cmd += ['-pix_fmt', 'yuv420p', out_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f'ffmpeg exited with code {proc.returncode}')
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass


def write_video_conditional(frames, out_path, fps, use_ffmpeg=True, ffmpeg_opts=None):
    ffmpeg_opts = ffmpeg_opts or {}
    if use_ffmpeg and shutil.which('ffmpeg'):
        try:
            ffmpeg_write_video(frames, out_path, fps=fps, crf=ffmpeg_opts.get('crf', 18), preset=ffmpeg_opts.get('preset', 'veryfast'), use_nvenc=ffmpeg_opts.get('use_nvenc', False))
            return
        except Exception as e:
            print('ffmpeg writer failed, falling back to OpenCV writer:', e)
    # fallback
    save_video(frames, out_path, fps=fps)


def process_frames(frames, timestamps, analyzer: FaceAnalyzer, sample_every=1,
                   quality_cfg=None, bad_save_dir: str = None):
    """Process a list of frames using the analyzer with pre-filtering for image quality.
    quality_cfg: dict with keys: min_landmarks, blur_threshold, min_contrast, min_brightness, max_roll_deg
    bad_save_dir: when provided, save excluded frames there for debugging
    Returns (per_frame_list, aggregate_stats).
    """
    quality_cfg = quality_cfg or {}
    min_landmarks = int(quality_cfg.get('min_landmarks', 40))
    blur_threshold = float(quality_cfg.get('blur_threshold', 100.0))
    min_contrast = float(quality_cfg.get('min_contrast', 10.0))
    min_brightness = float(quality_cfg.get('min_brightness', 30.0))
    max_roll_deg = float(quality_cfg.get('max_roll_deg', 20.0))

    per_frame = []
    detect_count = 0
    for i in range(0, len(frames), sample_every):
        img = frames[i]
        entry = {'frame_index': i, 'timestamp': timestamps[i] if i < len(timestamps) else None, 'detected': False, 'quality_pass': True, 'quality_reasons': []}

        # quick image-level checks
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_var = variance_of_laplacian(gray)
        mean_b, std_c = brightness_and_contrast(gray)
        if blur_var < blur_threshold:
            entry['quality_pass'] = False
            entry['quality_reasons'].append('blur')
        if std_c < min_contrast:
            entry['quality_pass'] = False
            entry['quality_reasons'].append('low_contrast')
        if mean_b < min_brightness:
            entry['quality_pass'] = False
            entry['quality_reasons'].append('low_brightness')

        # If failed basic checks, optionally save and skip heavy analysis
        if not entry['quality_pass']:
            if bad_save_dir:
                save_image_jpg(os.path.join(bad_save_dir, f'bad_frame_{i}.jpg'), img)
            per_frame.append(entry)
            continue

        # run landmark detector
        try:
            lm = analyzer.get_landmarks(img)
            if lm is None or len(lm) == 0:
                entry['detected'] = False
                entry['quality_pass'] = False
                entry['quality_reasons'].append('no_landmarks')
                if bad_save_dir:
                    save_image_jpg(os.path.join(bad_save_dir, f'no_landmarks_{i}.jpg'), img)
                per_frame.append(entry)
                continue

            # too few landmarks
            if len(lm) < min_landmarks:
                entry['detected'] = False
                entry['quality_pass'] = False
                entry['quality_reasons'].append('few_landmarks')
                if bad_save_dir:
                    save_image_jpg(os.path.join(bad_save_dir, f'few_landmarks_{i}.jpg'), img)
                per_frame.append(entry)
                continue

            # compute roll angle from eye centers (works for 68-point)
            try:
                # indices for 68-landmarks (0-based): left eye 36-41, right eye 42-47
                left_pts = lm[36:42]
                right_pts = lm[42:48]
                left_c = (sum([p[0] for p in left_pts]) / len(left_pts), sum([p[1] for p in left_pts]) / len(left_pts))
                right_c = (sum([p[0] for p in right_pts]) / len(right_pts), sum([p[1] for p in right_pts]) / len(right_pts))
                dy = right_c[1] - left_c[1]
                dx = right_c[0] - left_c[0]
                import math
                roll_deg = math.degrees(math.atan2(dy, dx))
                if abs(roll_deg) > max_roll_deg:
                    entry['detected'] = False
                    entry['quality_pass'] = False
                    entry['quality_reasons'].append('roll')
                    if bad_save_dir:
                        save_image_jpg(os.path.join(bad_save_dir, f'roll_{i}.jpg'), img)
                    per_frame.append(entry)
                    continue
            except Exception:
                # non-fatal: if we can't compute roll, continue
                pass

            # passed quality checks and has landmarks — do full analysis
            entry['detected'] = True
            detect_count += 1
            entry['landmarks_count'] = len(lm)
            entry['face_shape'] = analyzer.classify_face_shape(lm)
            entry['eyes'] = analyzer.detect_eye_shape_and_color(img, lm)
            entry['hair'] = analyzer.detect_hair(img, lm)
            # existing analyzer metrics
            facial_metrics = {
                'canthal_tilt': analyzer.compute_canthal_tilt(lm),
                'proportions': analyzer.face_proportions(lm),
                'maxilla': analyzer.maxilla_metrics(lm),
                'lips': analyzer.lip_shape(lm, image_bgr=img)
            }
            try:
                # add basic landmark-derived metrics (jaw width, fWHR, etc.)
                basic = compute_basic_metrics(lm, image_shape=img.shape)
                if basic:
                    facial_metrics.update({'basic_metrics': basic})
            except Exception:
                # non-fatal: keep existing metrics if compute_basic_metrics fails
                pass
            entry['facial_metrics'] = facial_metrics
        except Exception as e:
            entry['analysis_error'] = str(e)
        per_frame.append(entry)

    aggregate = {
        'frames_captured': len(frames),
        'frames_sampled': len(per_frame),
        'frames_detected': detect_count,
        'detection_rate': float(detect_count) / (len(per_frame) + 1e-9)
    }
    return per_frame, aggregate


def analyze_pair(front_img, side_img, out_path='results.json', device='cpu'):
    analyzer = FaceAnalyzer(device=device)
    front_lm = analyzer.get_landmarks(front_img)
    side_lm = analyzer.get_landmarks(side_img)

    front_info = {
        'landmarks_count': len(front_lm) if front_lm is not None else 0,
        'face_shape': analyzer.classify_face_shape(front_lm) if front_lm is not None else None,
        'eyes': analyzer.detect_eye_shape_and_color(front_img, front_lm) if front_lm is not None else None,
        'hair': analyzer.detect_hair(front_img, front_lm) if front_lm is not None else None,
        'facial_metrics': None
    }

    side_info = {
        'landmarks_count': len(side_lm) if side_lm is not None else 0,
        'face_shape': analyzer.classify_face_shape(side_lm) if side_lm is not None else None,
        'eyes': analyzer.detect_eye_shape_and_color(side_img, side_lm) if side_lm is not None else None,
        'hair': analyzer.detect_hair(side_img, side_lm) if side_lm is not None else None,
        'facial_metrics': None
    }

    out = {'front': front_info, 'side': side_info}
    # populate facial_metrics with both analyzer and basic metrics when possible
    try:
        if front_lm is not None:
            fm = {
                'canthal_tilt': analyzer.compute_canthal_tilt(front_lm),
                'proportions': analyzer.face_proportions(front_lm),
                'maxilla': analyzer.maxilla_metrics(front_lm),
                'lips': analyzer.lip_shape(front_lm, image_bgr=front_img)
            }
            try:
                basic = compute_basic_metrics(front_lm, image_shape=front_img.shape)
                if basic:
                    fm['basic_metrics'] = basic
            except Exception:
                pass
            out['front']['facial_metrics'] = fm

        if side_lm is not None:
            fm2 = {
                'canthal_tilt': analyzer.compute_canthal_tilt(side_lm),
                'proportions': analyzer.face_proportions(side_lm),
                'maxilla': analyzer.maxilla_metrics(side_lm),
                'lips': analyzer.lip_shape(side_lm, image_bgr=side_img)
            }
            try:
                basic2 = compute_basic_metrics(side_lm, image_shape=side_img.shape)
                if basic2:
                    fm2['basic_metrics'] = basic2
            except Exception:
                pass
            out['side']['facial_metrics'] = fm2
    except Exception:
        # non-fatal: leave facial_metrics as None if any unexpected error occurs
        pass
    from utils import make_json_serializable
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable(out), f, indent=2)
    print('Saved analysis to', out_path)
    return out


def compute_summary(per_frame_list):
    """Aggregate per-frame results into numeric averages and majority categorical values."""
    import collections
    num_frames = 0
    sums = {}
    counts = {}
    votes = collections.Counter()
    eye_ear_sums = {'left': 0.0, 'right': 0.0}
    eye_ear_counts = {'left': 0, 'right': 0}
    eye_colors = collections.Counter()
    hair_colors = collections.Counter()

    for ent in per_frame_list:
        if not ent or not ent.get('detected'):
            continue
        num_frames += 1
        # face shape ratio
        fs = ent.get('face_shape')
        if fs:
            ratio = fs.get('ratio')
            shape = fs.get('shape')
            if ratio is not None:
                sums.setdefault('face_ratio', 0.0)
                counts.setdefault('face_ratio', 0)
                sums['face_ratio'] += float(ratio)
                counts['face_ratio'] += 1
            if shape:
                votes[('face_shape', shape)] += 1

        # hair mask ratio + color
        hair = ent.get('hair')
        if hair:
            hr = hair.get('hair_mask_ratio')
            if hr is not None:
                sums.setdefault('hair_mask_ratio', 0.0)
                counts.setdefault('hair_mask_ratio', 0)
                sums['hair_mask_ratio'] += float(hr)
                counts['hair_mask_ratio'] += 1
            hc = hair.get('color')
            if hc:
                hair_colors[hc] += 1

        # eyes
        eyes = ent.get('eyes')
        if eyes:
            left = eyes.get('left')
            right = eyes.get('right')
            if left:
                ear = left.get('eye_aspect_ratio')
                if ear is not None:
                    eye_ear_sums['left'] += float(ear)
                    eye_ear_counts['left'] += 1
                col = left.get('color')
                if col:
                    eye_colors[col] += 1
            if right:
                ear = right.get('eye_aspect_ratio')
                if ear is not None:
                    eye_ear_sums['right'] += float(ear)
                    eye_ear_counts['right'] += 1
                col = right.get('color')
                if col:
                    eye_colors[col] += 1

        # facial_metrics numeric fields
        fm = ent.get('facial_metrics')
        if fm:
            # canthal tilt
            ct = fm.get('canthal_tilt')
            if ct:
                ang = ct.get('angle_degrees')
                if ang is not None:
                    sums.setdefault('canthal_tilt_deg', 0.0)
                    counts.setdefault('canthal_tilt_deg', 0)
                    sums['canthal_tilt_deg'] += float(ang)
                    counts['canthal_tilt_deg'] += 1
            # proportions width/height
            prop = fm.get('proportions')
            if prop:
                wh = prop.get('width_height_ratio')
                if wh is not None:
                    sums.setdefault('width_height_ratio', 0.0)
                    counts.setdefault('width_height_ratio', 0)
                    sums['width_height_ratio'] += float(wh)
                    counts['width_height_ratio'] += 1
            # maxilla
            mx = fm.get('maxilla')
            if mx:
                mr = mx.get('mouth_nose_ratio')
                if mr is not None:
                    sums.setdefault('mouth_nose_ratio', 0.0)
                    counts.setdefault('mouth_nose_ratio', 0)
                    sums['mouth_nose_ratio'] += float(mr)
                    counts['mouth_nose_ratio'] += 1
                phil = mx.get('philtrum_length')
                if phil is not None:
                    sums.setdefault('philtrum_length', 0.0)
                    counts.setdefault('philtrum_length', 0)
                    sums['philtrum_length'] += float(phil)
                    counts['philtrum_length'] += 1
            # lips
            lips = fm.get('lips')
            if lips:
                th = lips.get('total_lip_height')
                mo = lips.get('mouth_opening')
                if th is not None:
                    sums.setdefault('total_lip_height', 0.0)
                    counts.setdefault('total_lip_height', 0)
                    sums['total_lip_height'] += float(th)
                    counts['total_lip_height'] += 1
                if mo is not None:
                    sums.setdefault('mouth_opening', 0.0)
                    counts.setdefault('mouth_opening', 0)
                    sums['mouth_opening'] += float(mo)
                    counts['mouth_opening'] += 1

    # Build summary
    summary = {'num_frames': num_frames}
    for k, v in sums.items():
        c = counts.get(k, 0)
        summary[f'avg_{k}'] = (v / c) if c > 0 else None

    # eye EAR averages
    for side in ('left', 'right'):
        c = eye_ear_counts[side]
        summary[f'avg_eye_ear_{side}'] = (eye_ear_sums[side] / c) if c > 0 else None

    # majority votes
    if votes:
        fs_votes = {k[1]: cnt for k, cnt in votes.items() if k[0] == 'face_shape'}
        if fs_votes:
            summary['most_common_face_shape'] = max(fs_votes.items(), key=lambda x: x[1])[0]
    # dominant colors
    summary['dominant_eye_color'] = eye_colors.most_common(1)[0][0] if eye_colors else None
    summary['dominant_hair_color'] = hair_colors.most_common(1)[0][0] if hair_colors else None

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true', help='Capture front and side via webcam')
    parser.add_argument('--record', action='store_true', help='Record longer front and side segments for dataset collection')
    parser.add_argument('--duration', type=float, default=8.0, help='Duration in seconds for each recording segment (front and side)')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS when recording')
    parser.add_argument('--record-dir', help='Directory to save recorded videos and outputs (defaults to outputs/record_TIMESTAMP)')
    parser.add_argument('--sample-every', type=int, default=1, help='Process every Nth captured frame (1 = every frame)')
    parser.add_argument('--min-landmarks', type=int, default=40, help='Minimum landmarks required to accept a frame')
    parser.add_argument('--blur-threshold', type=float, default=100.0, help='Minimum Laplacian variance to consider frame sharp')
    parser.add_argument('--min-contrast', type=float, default=10.0, help='Minimum grayscale stddev to consider contrast OK')
    parser.add_argument('--min-brightness', type=float, default=30.0, help='Minimum grayscale mean brightness')
    parser.add_argument('--max-roll-deg', type=float, default=20.0, help='Maximum roll (tilt) degrees allowed for a frame')
    parser.add_argument('--save-bad-frames', action='store_true', help='Save excluded/bad frames into output folder for debugging')
    # auto-relax is enabled by default; provide a flag to disable it
    parser.add_argument('--auto-relax', dest='auto_relax', action='store_true', help='Automatically relax thresholds and reprocess if detection rate is too low')
    parser.add_argument('--no-auto-relax', dest='auto_relax', action='store_false', help='Disable automatic relaxation/reprocessing')
    parser.set_defaults(auto_relax=True)
    parser.add_argument('--relax-retries', type=int, default=2, help='Number of progressive relax retries to attempt')
    parser.add_argument('--min-session-detection-rate', type=float, default=0.05, help='Minimum detection rate (per-segment) considered acceptable; below this triggers auto-relax')
    parser.add_argument('--front', help='Path to front image')
    parser.add_argument('--side', help='Path to side image')
    parser.add_argument('--output', default='results.json', help='Output JSON file')
    parser.add_argument('--no-ffmpeg', dest='use_ffmpeg', action='store_false', help='Disable ffmpeg-based writer and use OpenCV writer')
    parser.add_argument('--ffmpeg-crf', type=int, default=18, help='CRF for libx264 when using ffmpeg (lower => better quality)')
    parser.add_argument('--ffmpeg-preset', default='veryfast', help='x264 preset for ffmpeg writer')
    parser.add_argument('--use-nvenc', action='store_true', help='Prefer NVENC encoder when available')
    args = parser.parse_args()
    device = 'cpu'
    if _TORCH_AVAILABLE and _CUDA_AVAILABLE:
        device = 'cuda'
    analyzer = FaceAnalyzer(device=device)
    if args.record:
        # prepare output dir
        outdir = args.record_dir
        if not outdir:
            tstamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            outdir = os.path.join('outputs', f'record_{tstamp}')
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

        

        # Configure quality filtering
        quality_cfg = {
            'min_landmarks': args.min_landmarks,
            'blur_threshold': args.blur_threshold,
            'min_contrast': args.min_contrast,
            'min_brightness': args.min_brightness,
            'max_roll_deg': args.max_roll_deg
        }
        bad_save_dir = None
        if args.save_bad_frames:
            bad_save_dir = os.path.join(outdir, 'bad_frames')
            pathlib.Path(bad_save_dir).mkdir(parents=True, exist_ok=True)

        # FRONT recording
        print('Prepare for FRONT recording. Please face the camera.')
        time.sleep(1.0)
        front_frames, front_ts, front_fps = record_segment(args.duration, fps=args.fps, preview_window_title='Recording FRONT')
        front_video_path = os.path.join(outdir, 'front.mp4')
        write_video_conditional(front_frames, front_video_path, fps=front_fps if front_fps>0 else args.fps, use_ffmpeg=args.use_ffmpeg, ffmpeg_opts={'crf': args.ffmpeg_crf, 'preset': args.ffmpeg_preset, 'use_nvenc': args.use_nvenc})
        print('Saved front video to', front_video_path)

        # Prompt user to turn for side capture immediately (processing deferred)
        input('Turn your head to the SIDE and press Enter to start side recording...')
        time.sleep(0.5)
        side_frames, side_ts, side_fps = record_segment(args.duration, fps=args.fps, preview_window_title='Recording SIDE')
        side_video_path = os.path.join(outdir, 'side.mp4')
        write_video_conditional(side_frames, side_video_path, fps=side_fps if side_fps>0 else args.fps, use_ffmpeg=args.use_ffmpeg, ffmpeg_opts={'crf': args.ffmpeg_crf, 'preset': args.ffmpeg_preset, 'use_nvenc': args.use_nvenc})
        print('Saved side video to', side_video_path)

        # Now process both segments (this may take time) — capture is complete so user interaction isn't blocked
        print('Processing captured segments (this may take some time)...')
        front_per_frame, front_agg = process_frames(front_frames, front_ts, analyzer, sample_every=args.sample_every, quality_cfg=quality_cfg, bad_save_dir=bad_save_dir)
        side_per_frame, side_agg = process_frames(side_frames, side_ts, analyzer, sample_every=args.sample_every, quality_cfg=quality_cfg, bad_save_dir=bad_save_dir)

        # Auto-relax logic: if detection rate is too low, progressively relax thresholds and reprocess
        def detection_ok(agg):
            return float(agg.get('detection_rate', 0.0)) >= float(args.min_session_detection_rate)

        if args.auto_relax and (not detection_ok(front_agg) or not detection_ok(side_agg)):
            print('Detection rate below threshold; attempting automatic relaxation and reprocessing...')
            relaxed_cfg = dict(quality_cfg)
            # store best seen
            best_front_pf, best_front_ag = front_per_frame, front_agg
            best_side_pf, best_side_ag = side_per_frame, side_agg
            for retry in range(max(1, args.relax_retries)):
                # relax rules progressively
                relaxed_cfg['blur_threshold'] = max(0.0, relaxed_cfg.get('blur_threshold', 100.0) * 0.5)
                relaxed_cfg['min_contrast'] = max(0.0, relaxed_cfg.get('min_contrast', 10.0) * 0.5)
                relaxed_cfg['min_brightness'] = max(0.0, relaxed_cfg.get('min_brightness', 30.0) - 10)
                relaxed_cfg['min_landmarks'] = max(6, int(relaxed_cfg.get('min_landmarks', 40)) - 15)

                bad_save_dir_rel = None
                if args.save_bad_frames:
                    bad_save_dir_rel = os.path.join(outdir, f'bad_frames_relaxed_{retry}')
                    pathlib.Path(bad_save_dir_rel).mkdir(parents=True, exist_ok=True)

                print(f'Auto-relax attempt {retry+1}: {relaxed_cfg}')
                front_pf, front_ag = process_frames(front_frames, front_ts, analyzer, sample_every=args.sample_every, quality_cfg=relaxed_cfg, bad_save_dir=bad_save_dir_rel)
                side_pf, side_ag = process_frames(side_frames, side_ts, analyzer, sample_every=args.sample_every, quality_cfg=relaxed_cfg, bad_save_dir=bad_save_dir_rel)

                # if both segments meet threshold, accept and stop
                if detection_ok(front_ag) and detection_ok(side_ag):
                    print('Auto-relax succeeded: detection rate now acceptable')
                    best_front_pf, best_front_ag = front_pf, front_ag
                    best_side_pf, best_side_ag = side_pf, side_ag
                    front_per_frame, front_agg = best_front_pf, best_front_ag
                    side_per_frame, side_agg = best_side_pf, best_side_ag
                    break

                # otherwise keep the best improvement
                improved = False
                if front_ag.get('detection_rate', 0.0) > best_front_ag.get('detection_rate', 0.0):
                    best_front_pf, best_front_ag = front_pf, front_ag
                    improved = True
                if side_ag.get('detection_rate', 0.0) > best_side_ag.get('detection_rate', 0.0):
                    best_side_pf, best_side_ag = side_pf, side_ag
                    improved = True

                if improved:
                    print('Auto-relax: found improved detection rate; keeping best so far and continuing')
                    front_per_frame, front_agg = best_front_pf, best_front_ag
                    side_per_frame, side_agg = best_side_pf, best_side_ag
                else:
                    print('Auto-relax: no improvement this attempt')

            # after attempts, save reprocessed results if they differ
            re_out_json = {
                'front': {'per_frame': best_front_pf, 'aggregate': best_front_ag},
                'side': {'per_frame': best_side_pf, 'aggregate': best_side_ag}
            }
            re_out_path = os.path.join(outdir, 'results_reprocessed.json')
            from utils import make_json_serializable
            with open(re_out_path, 'w', encoding='utf-8') as f:
                json.dump(make_json_serializable(re_out_json), f, indent=2)
            re_final = {
                'front_summary': compute_summary(best_front_pf),
                'side_summary': compute_summary(best_side_pf)
            }
            re_final_path = os.path.join(outdir, 'final_reprocessed.json')
            with open(re_final_path, 'w', encoding='utf-8') as f:
                json.dump(make_json_serializable(re_final), f, indent=2)
            print('Saved reprocessed results to', re_out_path, 'and', re_final_path)

        # pick representative frames (first detected) and run detailed analysis per segment
        def representative_analysis(frames, per_frame_list):
            for ent in per_frame_list:
                if ent.get('detected'):
                    idx = ent['frame_index']
                    img = frames[idx]
                    try:
                        lm = analyzer.get_landmarks(img)
                        return {
                            'frame_index': idx,
                            'landmarks_count': len(lm) if lm is not None else 0,
                            'face_shape': analyzer.classify_face_shape(lm) if lm is not None else None,
                            'eyes': analyzer.detect_eye_shape_and_color(img, lm) if lm is not None else None,
                            'hair': analyzer.detect_hair(img, lm) if lm is not None else None,
                            'facial_metrics': (lambda: (
                                (lambda fm: (fm.update({'basic_metrics': compute_basic_metrics(lm, image_shape=img.shape)}) or fm))({
                                    'canthal_tilt': analyzer.compute_canthal_tilt(lm),
                                    'proportions': analyzer.face_proportions(lm),
                                    'maxilla': analyzer.maxilla_metrics(lm),
                                    'lips': analyzer.lip_shape(lm, image_bgr=img)
                                })
                            )()) if lm is not None else None
                        }
                    except Exception as e:
                        return {'error': str(e)}
            return None

        front_repr = representative_analysis(front_frames, front_per_frame)
        side_repr = representative_analysis(side_frames, side_per_frame)

        out_json = {
            'front': {
                'video': str(front_video_path),
                'per_frame': front_per_frame,
                'aggregate': front_agg,
                'representative': front_repr
            },
            'side': {
                'video': str(side_video_path),
                'per_frame': side_per_frame,
                'aggregate': side_agg,
                'representative': side_repr
            }
        }
        out_json_path = os.path.join(outdir, args.output)
        from utils import make_json_serializable
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(out_json), f, indent=2)
        print('Saved recording JSON to', out_json_path)

        # Save aggregated final summary (averages + majority values)
        final_summary = {
            'front_summary': compute_summary(front_per_frame),
            'side_summary': compute_summary(side_per_frame)
        }
        final_path = os.path.join(outdir, 'final.json')
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(final_summary), f, indent=2)
        print('Saved final summary to', final_path)

        print('All done. Recorded data placed in', outdir)
        return
    elif args.capture:
        front, side = capture_two_frames(analyzer)
    else:
        if not args.front or not args.side:
            parser.error('Either --capture or both --front and --side must be provided')
        front = cv2.imread(args.front)
        side = cv2.imread(args.side)
        if front is None or side is None:
            raise RuntimeError('Could not read input images')

    analyze_pair(front, side, args.output, device=('cuda' if (_TORCH_AVAILABLE and _CUDA_AVAILABLE) else 'cpu'))


if __name__ == '__main__':
    main()
