import math
import sys, types
# Ensure tests can run even if cv2 is not installed in this environment by
# inserting a minimal dummy module into sys.modules before importing utils.
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.SimpleNamespace()

from utils import compute_basic_metrics


def make_synthetic_landmarks(cx=200, cy=200, scale=1.0):
    """Create a simple 68-point synthetic landmark set roughly centered.
    This is not anatomically perfect but sufficient to exercise functions.
    """
    lm = []
    # jaw 0-16: semicircle below
    for i in range(17):
        angle = math.pi * (i / 16.0) + math.pi
        x = cx + scale * 80.0 * math.cos(angle)
        y = cy + scale * 100.0 * math.sin(angle)
        lm.append((x, y))
    # brows 17-26: small arcs
    for i in range(10):
        angle = -0.3 + 0.06 * i
        x = cx + scale * (30.0 * (i/9.0 - 0.5))
        y = cy - scale * (70.0 + 5.0 * math.sin(angle))
        lm.append((x, y))
    # nose 27-35 (9 points)
    for i in range(9):
        x = cx + scale * 0.0
        y = cy - scale * (40.0 - i*5.0)
        lm.append((x, y))
    # left eye 36-41
    for i in range(6):
        angle = i / 5.0 * math.pi
        x = cx - scale * 30 + scale * 6.0 * math.cos(angle)
        y = cy - scale * 20 + scale * 3.0 * math.sin(angle)
        lm.append((x, y))
    # right eye 42-47
    for i in range(6):
        angle = i / 5.0 * math.pi
        x = cx + scale * 30 + scale * 6.0 * math.cos(angle)
        y = cy - scale * 20 + scale * 3.0 * math.sin(angle)
        lm.append((x, y))
    # mouth 48-59 (12 points)
    for i in range(12):
        angle = i / 11.0 * 2.0 * math.pi
        x = cx + scale * 20.0 * math.cos(angle)
        y = cy + scale * 30.0 * math.sin(angle)
        lm.append((x, y))
    # inner mouth 60-67 (8 points)
    for i in range(8):
        angle = i / 7.0 * 2.0 * math.pi
        x = cx + scale * 10.0 * math.cos(angle)
        y = cy + scale * 15.0 * math.sin(angle)
        lm.append((x, y))
    return lm


def test_compute_basic_metrics_minimal():
    lm = make_synthetic_landmarks()
    metrics = compute_basic_metrics(lm, image_shape=(480, 640, 3))
    # check some expected keys
    for k in ['jaw_width', 'chin_width', 'bizygomatic_width', 'fwh_ratio', 'left_eye_height', 'right_eye_height', 'mouth_width']:
        assert k in metrics
        assert metrics[k] is not None
    # normalized variants
    assert 'jaw_width_norm' in metrics
    assert metrics['jaw_width_norm'] > 0


if __name__ == '__main__':
    test_compute_basic_metrics_minimal()
    print('ok')
