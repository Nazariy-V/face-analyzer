import os
import cv2
import numpy as np

def dominant_color_bgr(image, mask=None, k=3):
    """Return dominant color in BGR as (b,g,r) using k-means.
    image: BGR image
    mask: optional binary mask where to consider pixels
    """
    if mask is not None:
        pixels = image[mask > 0]
    else:
        pixels = image.reshape(-1, 3)
    if len(pixels) == 0:
        return (0, 0, 0)
    pixels = np.float32(pixels)
    # criteria, attempts
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = centers[np.argmax(counts)]
    return tuple(int(c) for c in dominant)

def bgr_to_hex(bgr):
    b, g, r = bgr
    return '#%02x%02x%02x' % (r, g, b)

def resize_max(image, max_dim=1024):
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    return image, scale

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def variance_of_laplacian(gray):
    """Return the variance of the Laplacian (measure of sharpness)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def brightness_and_contrast(gray):
    """Return (mean_brightness, std_contrast) from a grayscale image."""
    return float(np.mean(gray)), float(np.std(gray))


def save_image_jpg(path, image):
    """Save image as JPEG (create parent dirs)."""
    os_path = os.path.dirname(path)
    if os_path and not os.path.exists(os_path):
        os.makedirs(os_path, exist_ok=True)
    cv2.imwrite(path, image)


def _pt(landmarks, idx):
    """Return landmark point as (x,y). Expects landmarks as list/array of (x,y)."""
    try:
        p = landmarks[idx]
        return (float(p[0]), float(p[1]))
    except Exception:
        return (0.0, 0.0)


def compute_basic_metrics(landmarks, image_shape=None):
    """Compute low-effort facial metrics from 68-point landmarks.

    Returns a dict with normalized distances/ratios. Landmarks are expected
    to follow the common 68-point mapping (dlib/face-alignment).

    image_shape: optional (h,w) if you want metrics normalized by image size.
    """
    import math
    lm = landmarks
    if not lm or len(lm) < 68:
        return {}

    def dist(i, j):
        a = _pt(lm, i)
        b = _pt(lm, j)
        return euclidean(a, b)

    # canonical points (0-based dlib/68):
    # chin = 8, jaw left/right approximations: 4 & 12, zygion approximations: 1 & 15
    # nasion/top nose: 27, nose tip: 30, subnasale/base ~33
    # eyes: left 36-41, right 42-47; brows inner: 21 & 22
    # mouth corners: 48 & 54

    metrics = {}

    # basic scales
    interocular = (dist(36, 45) if len(lm) > 45 else 1.0)
    face_height = (dist(27, 8) if len(lm) > 8 else 1.0)
    face_width = (dist(1, 15) if len(lm) > 15 else 1.0)

    metrics['interocular_distance'] = interocular
    metrics['face_height'] = face_height
    metrics['face_width'] = face_width

    # jaw / chin widths
    metrics['jaw_width'] = dist(4, 12)
    metrics['chin_width'] = dist(6, 10)

    # cheekbone / bizygomatic width
    metrics['bizygomatic_width'] = dist(1, 15)

    # facial width-to-height ratio (fWHR)
    metrics['fwh_ratio'] = (metrics['bizygomatic_width'] / face_height) if face_height > 0 else None

    # midface length (nasion to base of nose)
    metrics['midface_length'] = dist(27, 33)

    # nose measures
    metrics['nose_length'] = dist(27, 30)
    metrics['nose_width'] = dist(31, 35)

    # forehead height estimate: distance from top of landmarks bbox to nasion (27)
    ys = [float(p[1]) for p in lm]
    top_y = min(ys)
    nasion_y = _pt(lm, 27)[1]
    metrics['forehead_height_est'] = max(0.0, nasion_y - top_y)

    # eyes: height approx as vertical bbox of eye landmarks
    def eye_height(indices):
        ys = [_pt(lm, i)[1] for i in indices]
        return max(ys) - min(ys)

    metrics['left_eye_height'] = eye_height(range(36, 42))
    metrics['right_eye_height'] = eye_height(range(42, 48))
    # eye centers and spacing
    def center(indices):
        xs = [_pt(lm, i)[0] for i in indices]
        ys = [_pt(lm, i)[1] for i in indices]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    left_c = center(range(36, 42))
    right_c = center(range(42, 48))
    metrics['eye_spacing'] = euclidean(left_c, right_c)

    # eyebrow spacing (inner brows 21 & 22)
    metrics['eyebrow_spacing'] = dist(21, 22)

    # mouth width
    metrics['mouth_width'] = dist(48, 54)

    # normalization: provide some normalized variants if face_width or interocular available
    if face_width > 0:
        metrics['jaw_width_norm'] = metrics['jaw_width'] / face_width
        metrics['chin_width_norm'] = metrics['chin_width'] / face_width
        metrics['mouth_width_norm'] = metrics['mouth_width'] / face_width
        metrics['bizygomatic_width_norm'] = metrics['bizygomatic_width'] / face_width
    if interocular > 0:
        metrics['left_eye_height_norm'] = metrics['left_eye_height'] / interocular
        metrics['right_eye_height_norm'] = metrics['right_eye_height'] / interocular
        metrics['eye_spacing_norm'] = metrics['eye_spacing'] / interocular

    # add image-normalized variants if image_shape supplied
    if image_shape is not None:
        h, w = image_shape[:2]
        metrics['jaw_width_px'] = metrics['jaw_width']
        metrics['jaw_width_rel_img'] = metrics['jaw_width'] / float(w) if w>0 else None

    return metrics


def make_json_serializable(obj):
    """Recursively convert Python objects to JSON-serializable forms.

    - numpy types/arrays -> native Python types/lists
    - callables/functions -> their repr string
    - unknown objects -> str(obj)
    """
    import numpy as _np

    def _convert(o):
        # primitives
        if o is None or isinstance(o, (str, bool, int, float)):
            return o
        # numpy scalars
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        # dict
        if isinstance(o, dict):
            return {str(k): _convert(v) for k, v in o.items()}
        # list/tuple/set
        if isinstance(o, (list, tuple, set)):
            return [_convert(x) for x in o]
        # callable -> repr
        if callable(o):
            try:
                return repr(o)
            except Exception:
                return f'<callable {type(o).__name__}>'
        # otherwise try to stringify
        try:
            return str(o)
        except Exception:
            return f'<unserializable {type(o).__name__}>'

    return _convert(obj)
