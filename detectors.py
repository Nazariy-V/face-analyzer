import cv2
import numpy as np
from utils import dominant_color_bgr, bgr_to_hex, resize_max, euclidean

# Try to import PyTorch-based face_alignment first (preferred). Fall back to
# MediaPipe if available, otherwise use landmark-free heuristics.
try:
    import face_alignment
    _HAS_FA = True
except Exception:
    _HAS_FA = False

try:
    import mediapipe as mp
    _HAS_MP = True
    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
except Exception:
    _HAS_MP = False

try:
    import torch
    _HAS_TORCH = True
    # enable cudnn benchmark for potentially faster kernels (when input sizes stable)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    # disable gradient computations globally (we don't train)
    torch.set_grad_enabled(False)
except Exception:
    _HAS_TORCH = False


class FaceAnalyzer:
    def __init__(self, device='cpu'):
        self.device = device
        self.fa = None
        if _HAS_FA:
            # Choose a 2D landmarks enum value robustly (face_alignment versions differ)
            lm_type = None
            try:
                LT = face_alignment.LandmarksType
                # try common attribute names
                for attr in ('_2D', '_2d', 'LANDMARKS_2D', 'LANDMARKS_2d'):
                    if hasattr(LT, attr):
                        lm_type = getattr(LT, attr)
                        break
                # if not found, take the first enum member
                if lm_type is None:
                    try:
                        lm_type = list(LT)[0]
                    except Exception:
                        lm_type = None
            except Exception:
                lm_type = None

            try:
                if lm_type is not None:
                    self.fa = face_alignment.FaceAlignment(lm_type, device=device)
                else:
                    # try default construction (some versions accept only device)
                    self.fa = face_alignment.FaceAlignment(device=device)
            except Exception:
                # If initialization fails, leave fa as None and fall back to other backends
                self.fa = None
        # helper to run face_alignment safely under torch.no_grad and autocast when available
        if _HAS_FA:
            def _fa_predict(img):
                # img: RGB numpy image
                try:
                    if _HAS_TORCH and torch.cuda.is_available():
                        with torch.no_grad():
                            # use the unified amp API; pass device_type explicitly for CUDA
                            with torch.amp.autocast(device_type='cuda'):
                                return self.fa.get_landmarks_from_image(img)
                    else:
                        with torch.no_grad():
                            return self.fa.get_landmarks_from_image(img)
                except Exception:
                    return None
            self._fa_predict = _fa_predict
        else:
            self._fa_predict = None
        # Haar cascade fallback for face detection (used to crop for face_alignment)
        try:
            self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.haar_cascade = None

        self.face_mesh = None
        self.seg = None
        if _HAS_MP:
            self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                                  refine_landmarks=True,
                                                  max_num_faces=1,
                                                  min_detection_confidence=0.5)
            self.seg = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def get_landmarks(self, image_bgr):
        """Return landmarks as a list of (x,y) in image pixel coords.
        Tries face_alignment (68 points) first, then MediaPipe (468 points).
        """
        h, w = image_bgr.shape[:2]
        # Try face_alignment
        if self.fa is not None:
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # primary attempt
                if self._fa_predict is not None:
                    preds = self._fa_predict(image_rgb)
                else:
                    preds = None
                if preds is not None and len(preds) > 0:
                    lm = preds[0]
                    pts = [(int(float(x)), int(float(y))) for (x, y) in lm]
                    return pts

                # try a few fallback strategies to improve detection robustness
                # 1) upscale the image and retry
                    try:
                        scale = 1.5
                        up = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                        preds = self._fa_predict(up) if self._fa_predict is not None else None
                        if preds is not None and len(preds) > 0:
                            lm = preds[0]
                            pts = [(int(float(x)/scale), int(float(y)/scale)) for (x, y) in lm]
                            return pts
                    except Exception:
                        pass

                # 2) apply CLAHE/equalization on grayscale and retry
                    try:
                        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                        ge = clahe.apply(gray)
                        ge_color = cv2.cvtColor(ge, cv2.COLOR_GRAY2RGB)
                        preds = self._fa_predict(ge_color) if self._fa_predict is not None else None
                        if preds is not None and len(preds) > 0:
                            lm = preds[0]
                            pts = [(int(float(x)), int(float(y))) for (x, y) in lm]
                            return pts
                    except Exception:
                        pass

                # 3) use Haar cascade to find face bbox and run face_alignment on the cropped region
                try:
                    if self.haar_cascade is not None:
                        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                        rects = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
                        for (x, y, ww, hh) in rects:
                            crop = image_rgb[y:y+hh, x:x+ww]
                            preds = self._fa_predict(crop) if self._fa_predict is not None else None
                            if preds is not None and len(preds) > 0:
                                lm = preds[0]
                                pts = [(int(float(px) + x), int(float(py) + y)) for (px, py) in lm]
                                return pts
                except Exception:
                    pass
            except Exception:
                pass

        # Try MediaPipe
        if self.face_mesh is not None:
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)
                if not results.multi_face_landmarks:
                    return None
                lm = results.multi_face_landmarks[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                return pts
            except Exception:
                pass

        return None

    def get_face_bbox(self, landmarks):
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (x1, y1, x2, y2)

    def classify_face_shape(self, landmarks):
        # Support both 68-point (face_alignment) and 468-point (MediaPipe) landmarks
        if len(landmarks) == 68:
            left = landmarks[0]
            right = landmarks[16]
            chin = landmarks[8]
            # approximate forehead by moving point 27 upwards by a fraction of eye distance
            eye_left = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
            eye_right = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
            eye_dist = euclidean(eye_left, eye_right)
            p27 = landmarks[27]
            forehead = (int(p27[0]), int(p27[1] - eye_dist * 0.9))
        else:
            # assume MediaPipe 468
            left = landmarks[234]
            right = landmarks[454]
            chin = landmarks[152]
            forehead = landmarks[10]

        width = euclidean(left, right)
        height = euclidean(forehead, chin)
        ratio = height / (width + 1e-6)

        # compute jaw vs cheek widths where possible
        if len(landmarks) == 68:
            jaw_width = euclidean(landmarks[0], landmarks[16])
            cheek_width = euclidean(landmarks[2], landmarks[14])
        else:
            jaw_width = euclidean(landmarks[234], landmarks[454])
            cheek_width = euclidean(landmarks[127], landmarks[356])

        if ratio < 0.9:
            shape = 'round'
        elif ratio > 1.6:
            shape = 'oblong'
        else:
            if jaw_width > cheek_width * 1.06:
                shape = 'square'
            else:
                shape = 'oval'

        return {'shape': shape, 'ratio': float(ratio)}

    def compute_canthal_tilt(self, landmarks):
        """Compute canthal tilt (angle between inner canthi) in degrees and classify.
        Uses 68-point indices 39 (left inner) and 42 (right inner) when available,
        otherwise attempts to use MediaPipe approximate indices 133/362 inner corners.
        """
        try:
            if len(landmarks) == 68:
                left_inner = landmarks[39]
                right_inner = landmarks[42]
            else:
                # approximate medial canthi in MediaPipe
                left_inner = landmarks[133]
                right_inner = landmarks[362]
        except Exception:
            return {'angle_degrees': None, 'classification': 'unknown'}

        dx = right_inner[0] - left_inner[0]
        dy = right_inner[1] - left_inner[1]
        import math
        angle = math.degrees(math.atan2(dy, dx))
        # classification: small thresholds for practical purposes
        if angle > 2.0:
            cls = 'positive'  # right inner is higher -> positive tilt
        elif angle < -2.0:
            cls = 'negative'
        else:
            cls = 'neutral'
        return {'angle_degrees': float(angle), 'classification': cls}

    def face_proportions(self, landmarks):
        """Compute vertical thirds and width/height ratio. Returns values and simple labels."""
        import numpy as _np
        try:
            if len(landmarks) == 68:
                glabella = landmarks[27]
                nose_base = landmarks[33]
                chin = landmarks[8]
                # approximate forehead point by moving pt27 up by eye distance
                eye_left = _np.mean([landmarks[i] for i in range(36, 42)], axis=0)
                eye_right = _np.mean([landmarks[i] for i in range(42, 48)], axis=0)
                eye_dist = euclidean(eye_left, eye_right)
                forehead = (int(glabella[0]), int(glabella[1] - eye_dist * 0.9))
                # bizygomatic width approx using points 1 and 15
                try:
                    zyg_left = landmarks[1]
                    zyg_right = landmarks[15]
                except Exception:
                    zyg_left = landmarks[2]
                    zyg_right = landmarks[14]
            else:
                forehead = landmarks[10]
                glabella = landmarks[27]
                nose_base = landmarks[2] if len(landmarks) > 33 else landmarks[33]
                chin = landmarks[152]
                zyg_left = landmarks[127]
                zyg_right = landmarks[356]
        except Exception:
            return {'upper': None, 'middle': None, 'lower': None, 'width_height_ratio': None}

        upper = euclidean(forehead, glabella)
        middle = euclidean(glabella, nose_base)
        lower = euclidean(nose_base, chin)
        face_h = upper + middle + lower
        width = euclidean(zyg_left, zyg_right)
        wh_ratio = width / (face_h + 1e-6)

        # classify thirds roughly
        thirds = _np.array([upper, middle, lower]) / (face_h + 1e-6)
        thirds_labels = ['upper', 'middle', 'lower']
        largest = thirds_labels[int(_np.argmax(thirds))]

        return {
            'upper_mm': float(upper),
            'middle_mm': float(middle),
            'lower_mm': float(lower),
            'thirds_ratio': [float(x) for x in thirds.tolist()],
            'largest_third': largest,
            'width_height_ratio': float(wh_ratio)
        }

    def maxilla_metrics(self, landmarks):
        """Compute simple maxilla-related metrics: mouth-to-nose width ratio and philtrum length."""
        try:
            if len(landmarks) == 68:
                mouth_left = landmarks[48]
                mouth_right = landmarks[54]
                nose_left = landmarks[31]
                nose_right = landmarks[35]
                subnasale = landmarks[33]
                upper_lip_top = landmarks[51]
            else:
                # MediaPipe approximations
                mouth_left = landmarks[61]
                mouth_right = landmarks[291]
                nose_left = landmarks[93]
                nose_right = landmarks[323]
                subnasale = landmarks[2]
                upper_lip_top = landmarks[13]
        except Exception:
            return {'mouth_nose_ratio': None, 'philtrum_length': None}

        mouth_w = euclidean(mouth_left, mouth_right)
        nose_w = euclidean(nose_left, nose_right)
        philtrum = euclidean(subnasale, upper_lip_top)
        ratio = mouth_w / (nose_w + 1e-6)
        return {'mouth_nose_ratio': float(ratio), 'philtrum_length': float(philtrum)}

    def lip_shape(self, landmarks, image_bgr=None):
        """Estimate lip fullness and shape using 68 (preferred) or MediaPipe landmarks.
        Returns measures and a simple classification: thin/average/full.
        """
        try:
            if len(landmarks) == 68:
                top_outer = landmarks[51]
                bottom_outer = landmarks[57]
                inner_top = landmarks[62]
                inner_bottom = landmarks[66]
            else:
                top_outer = landmarks[13]
                bottom_outer = landmarks[14]
                inner_top = landmarks[0]
                inner_bottom = landmarks[17]
        except Exception:
            return {'total_lip_height': None, 'mouth_opening': None, 'classification': 'unknown'}

        total_height = euclidean(top_outer, bottom_outer)
        mouth_opening = euclidean(inner_top, inner_bottom)
        # Use total_height relative to face height (approx using chin and glabella if available)
        face_h = None
        try:
            if len(landmarks) == 68:
                glabella = landmarks[27]
                chin = landmarks[8]
            else:
                glabella = landmarks[10]
                chin = landmarks[152]
            face_h = euclidean(glabella, chin)
        except Exception:
            face_h = None

        fullness = None
        if face_h is not None:
            rel = total_height / (face_h + 1e-6)
            if rel < 0.03:
                fullness = 'thin'
            elif rel < 0.06:
                fullness = 'average'
            else:
                fullness = 'full'
        else:
            fullness = 'unknown'

        return {'total_lip_height': float(total_height), 'mouth_opening': float(mouth_opening), 'classification': fullness}

    def detect_eye_shape_and_color(self, image_bgr, landmarks):
        # Support both landmark sets
        def eye_info(eye_idxs, iris_box=None):
            pts = [landmarks[i] for i in eye_idxs]
            left = np.array(pts[0])
            right = np.array(pts[3])
            horiz = euclidean(left, right)
            top = np.array(pts[1])
            bottom = np.array(pts[2])
            vert = euclidean(top, bottom)
            ear = vert / (horiz + 1e-6)
            if ear > 0.35:
                shape = 'round'
            elif ear > 0.23:
                shape = 'almond'
            else:
                shape = 'hooded'

            color_hex = None
            if iris_box is not None:
                x1, y1, x2, y2 = iris_box
                roi = image_bgr[y1:y2, x1:x2]
                if roi.size > 0:
                    dom = dominant_color_bgr(roi, k=2)
                    color_hex = bgr_to_hex(dom)

            return {'shape': shape, 'eye_aspect_ratio': float(ear), 'color': color_hex}

        if len(landmarks) == 68:
            left_eye_idxs = [36, 37, 41, 39]
            right_eye_idxs = [42, 43, 47, 45]
            # iris not available from 68-point model; approximate iris box from eye bbox
            def iris_box_for_eye(idxs):
                pts = [landmarks[i] for i in idxs]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, x2 = max(min(xs)-3,0), min(max(xs)+3, image_bgr.shape[1]-1)
                y1, y2 = max(min(ys)-3,0), min(max(ys)+3, image_bgr.shape[0]-1)
                return (x1, y1, x2, y2)

            left = eye_info(left_eye_idxs, iris_box_for_eye(left_eye_idxs))
            right = eye_info(right_eye_idxs, iris_box_for_eye(right_eye_idxs))
        else:
            # MediaPipe indices (approx)
            left_eye_idxs = [33, 159, 145, 133]
            right_eye_idxs = [362, 386, 374, 263]
            left_iris_idxs = [468, 469, 470, 471]
            right_iris_idxs = [473, 474, 475, 476]

            def iris_box_for_idxs(idxs):
                irpts = [landmarks[i] for i in idxs]
                xs = [p[0] for p in irpts]
                ys = [p[1] for p in irpts]
                x1, x2 = max(min(xs)-2,0), min(max(xs)+2, image_bgr.shape[1]-1)
                y1, y2 = max(min(ys)-2,0), min(max(ys)+2, image_bgr.shape[0]-1)
                return (x1, y1, x2, y2)

            left = eye_info(left_eye_idxs, iris_box_for_idxs(left_iris_idxs))
            right = eye_info(right_eye_idxs, iris_box_for_idxs(right_iris_idxs))

        return {'left': left, 'right': right}

    def detect_hair(self, image_bgr, landmarks):
        # Try to create a hair mask using segmentation if available; otherwise use GrabCut
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = self.get_face_bbox(landmarks)
        pad_x = int((x2-x1) * 0.25)
        pad_y_top = int((y2-y1) * 0.8)
        xa = max(0, x1 - pad_x)
        xb = min(w-1, x2 + pad_x)
        ya = max(0, y1 - pad_y_top)
        yb = min(h-1, y2 + int((y2-y1)*0.2))
        head_roi = image_bgr[ya:yb, xa:xb]

        mask = None
        if self.seg is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            seg_res = self.seg.process(rgb)
            if seg_res and seg_res.segmentation_mask is not None:
                seg_mask = (seg_res.segmentation_mask * 255).astype(np.uint8)
                mask = seg_mask[ya:yb, xa:xb]

        if mask is None:
            # Initialize GrabCut mask: mark face bbox as probable foreground, edges as probable background
            roi = head_roi.copy()
            mask_gc = np.zeros(roi.shape[:2], np.uint8)
            # rectangle relative to roi where foreground likely exists (shrink face bbox)
            h_roi, w_roi = roi.shape[:2]
            rect = (int(w_roi*0.1), int(h_roi*0.05), int(w_roi*0.8), int(h_roi*0.9))
            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)
            try:
                cv2.grabCut(roi, mask_gc, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                mask_gc2 = np.where((mask_gc==2)|(mask_gc==0), 0, 1).astype('uint8')*255
                combined = mask_gc2
            except Exception:
                # Fallback to HSV skin removal similar to prior heuristic
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                skin_mask = cv2.inRange(hsv, (0, 30, 50), (50, 200, 255))
                combined = cv2.bitwise_not(skin_mask)
        else:
            combined = mask

        kernel = np.ones((5,5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        dom = dominant_color_bgr(head_roi, mask=combined, k=2)
        color_hex = bgr_to_hex(dom)

        hair_pixels = np.count_nonzero(combined)
        total = combined.size
        hair_ratio = hair_pixels / (total + 1e-6)
        if hair_ratio < 0.02:
            length = 'very short'
        elif hair_ratio < 0.08:
            length = 'short'
        elif hair_ratio < 0.18:
            length = 'medium'
        else:
            length = 'long'

        return {'color': color_hex, 'length': length, 'hair_mask_ratio': float(hair_ratio)}
