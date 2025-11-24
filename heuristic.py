import cv2
import numpy as np


def add_auto_padding(image, ratio=0.05):
    h, w = image.shape[:2]
    pad = int(min(h, w) * ratio)
    return add_padding(image, pad)


def add_padding(image, padding=1000, color=(44, 44, 44)):
    """
    Add padding around the entire image.
    padding: number of pixels on each side
    color: RGB background color (default = white)
    """
    return cv2.copyMakeBorder(
        image,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=color
    )

def crop_to_document_edges(image_np, padding_ratio=0.009, extra_padding=40):
    """
    1. Try detect edges on original.
    2. If fail → add padding and retry.
    3. Crop using detected box.
    """

    # ----- Try detection on original image -----
    box = detect_document_edges(image_np)

    if box is None:
        # Add padding ONLY when needed
        padded = add_auto_padding(image_np, ratio=0.04)
        box = detect_document_edges(padded)

        # If still nothing detected → give up
        if box is None:
            return padded, None

        # detection succeeded on padded image
        working_image = padded

    else:
        # detection succeeded on original image
        working_image = image_np

    # ----- Final crop -----
    x, y, w, h = box
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x1 = max(x - pad_w, 0)
    y1 = max(y - pad_h, 0)
    x2 = min(x + w + pad_w, working_image.shape[1])
    y2 = min(y + h + pad_h, working_image.shape[0])

    cropped = working_image[y1:y2, x1:x2]
    return cropped, (x1, y1, x2 - x1, y2 - y1)

def detect_document_edges(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = image_np.shape[:2]
    img_area = img_h * img_w

    best_box = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < img_area * 0.05:  # ignore small noise
            continue

        aspect = w / h if h != 0 else 0

        # ---- Strict horizontal rectangle filter ----
        if w <= h:          # must be wider than tall
            continue
        if not (1.2 <= aspect <= 2.2):   # adjust if needed
            continue

        # pick the largest valid horizontal box
        if area > best_area:
            best_area = area
            best_box = (x, y, w, h)

    return best_box


def detect_blur_fft(image, size=30, thresh=10):
    """
    Detect blur using FFT (Fast Fourier Transform).

    Args:
        image: input image (RGB or BGR)
        size: size of the low-frequency cutoff square
        thresh: threshold to classify blur (lower = more blurry)

    Returns:
        is_blurry (True/False)
        blur_score (float)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # FFT transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Remove low frequencies (center region)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    fshift[cy - size:cy + size, cx - size:cx + size] = 0

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Blur score = mean of high-frequency response
    blur_score = np.mean(img_back)

    # Blurry if score is below threshold
    is_blurry = blur_score < thresh

    return is_blurry, blur_score



def detect_brightness(image):
    """Return average brightness (0–255)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2].mean()
    return brightness

def detect_rotation_angle(image):
    """
    Detect dominant rotation angle using Hough Transform.
    Returns angle in degrees in range [-45, 45].
    """
    # convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0  # fallback

    angles = []
    for line in lines:
        rho, theta = line[0]

        angle = (theta * 180 / np.pi)

        # convert Hough angle to rotation angle
        angle = angle - 90  # vertical edges → horizontal reference

        # normalize to [-45, 45]
        if angle > 45:
            angle -= 90
        if angle < -45:
            angle += 90

        angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))



def detect_face(image):
    """Return (face_detected: bool, face_boxes: list of (x, y, w, h))."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0, faces


def detect_mrz_region(image):
    """
    Detects MRZ-like text lines near the bottom and returns debug image.

    Returns:
        mrz_present (bool): True if 2+ text-like lines detected
        debug_image (np.array): Original image annotated with MRZ region and contours
    """
    debug_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # --- Step 1: Determine probable document type ---
    aspect_ratio = w / h  # landscape ID ≈ 1.5+ | passport ≈ 1.3
    is_id_card = aspect_ratio > 1.5     # Wide → ID card expected
    is_passport = not is_id_card

    # --- Step 2: Select MRZ zone automatically ---
    if is_passport:
        y_start = int(h * 0.75)   # passports bottom ~30%
    else:
        y_start = int(h * 0.62)   # ID cards bottom ~40%

    mrz_area = gray[y_start:, :]

    # Threshold
    _, thresh = cv2.threshold(mrz_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology to join text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_like_lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100:
            text_like_lines.append(cnt)
            # Draw bounding box on debug image (shift y by y_start)
            cv2.rectangle(debug_image, (x, y + y_start), (x + w, y + h + y_start), (255, 255, 0), 1)

    # Draw MRZ area rectangle
    cv2.rectangle(debug_image, (0, y_start), (image.shape[1], image.shape[0]), (0, 0, 255), 1)

    mrz_present = len(text_like_lines) >= 2
    return mrz_present, debug_image


def is_likely_document(image):
    """Detects if a document-like rotated rectangle exists in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.05:
            continue  # ignore small shapes

        # --- use rotated rectangle instead of axis-aligned ---
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]

        if w == 0 or h == 0:
            continue

        # Ensure aspect ratio is always > 1
        aspect = max(w, h) / min(w, h)

        # Typical document ratios (ID card 1.5–1.7, passport ~1.42, A4 ~1.41)
        if 1.2 < aspect < 2.2:
            return True

    return False



# ---- Quality Assessment Functions ----
def calculate_face_blur(face_crop, width, height):
    """Multi-scale FFT analysis for maximum accuracy"""
    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    h, w = face_gray.shape

    # Compute FFT
    fft = np.fft.fft2(face_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.log(np.abs(fft_shift) + 1)  # Log scale for better distribution

    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]

    # Analyze multiple frequency bands
    frequency_bands = [
        (0.1, 0.3),  # Low-mid frequencies
        (0.3, 0.5),  # Mid frequencies
        (0.5, 0.7),  # High-mid frequencies
        (0.7, 1.0)  # High frequencies
    ]

    band_energies = []
    max_radius = min(h, w) / 2

    for low_frac, high_frac in frequency_bands:
        low_radius = max_radius * low_frac
        high_radius = max_radius * high_frac

        band_mask = (((x - center_x) ** 2 + (y - center_y) ** 2) > low_radius ** 2) & \
                    (((x - center_x) ** 2 + (y - center_y) ** 2) <= high_radius ** 2)

        band_energy = np.sum(magnitude_spectrum * band_mask)
        band_energies.append(band_energy)

    # Weight higher frequencies more (they indicate sharpness)
    weights = [0.1, 0.2, 0.3, 0.4]  # Higher weights for higher frequencies
    weighted_score = sum(e * w for e, w in zip(band_energies, weights))

    # Normalize
    normalized_score = weighted_score / 10  # Adjust based on your testing

    return normalized_score


def detect_face_glare(face_crop):
    """Detect percentage of face area with glare"""
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_RGB2HSV)
    value_channel = hsv[:, :, 2]
    glare_threshold = 220
    glare_areas = value_channel > glare_threshold
    glare_percentage = np.sum(glare_areas) / (face_crop.shape[0] * face_crop.shape[1]) * 100
    return glare_percentage



# ---- Evaluation Functions ----
def evaluate_blur_simple(blur_score):
    """Returns True if blur is acceptable (lower score = sharper)."""
    return blur_score <= 20


def evaluate_glare_simple(glare_percentage):
    """Returns True if glare is acceptable."""
    return glare_percentage < 5


# ---- Evaluation Functions ----
def evaluate_blur(blur_score):
    """Evaluation for accurate FFT-based blur detection"""
    # These thresholds will need calibration with your test images
    if blur_score > 80:  # Excellent - very sharp
        return 10, "✅ Face extremely clear (10/10)"
    elif blur_score > 60:  # Very good
        return 8, "✅ Face very clear (18/20)"
    elif blur_score > 40:  # Good
        return 6, "✅ Face clear (15/20)"
    elif blur_score > 20:  # Acceptable
        return 4, "⚠️ Face slightly blurry (10/20)"
    elif blur_score > 10:  # Poor
        return 2, "⚠️ Face blurry (5/20)"
    else:  # Very poor
        return 0, "❌ Face too blurry (0/20)"


def evaluate_glare(glare_percentage):
    """Evaluate glare with penalty"""
    if glare_percentage < 2:
        return 10, "✅ No significant glare (+5)"
    elif glare_percentage < 5:
        return 5, "⚠️ Some glare detected (−5)"
    else:
        return 0, "❌ Significant glare detected (−10)"


def evaluate_quality(image_np):
    """
    Full evaluation of passport/ID image.
    Returns:
        score (int): Final numeric score 0-100
        friendly_reasons (list): Human-readable grouped feedback for UI
        debug (dict): Internal metrics (always complete)
    """

    # -----------------------------
    # 1. DETECTIONS
    # -----------------------------
    brightness = detect_brightness(image_np)
    angle = detect_rotation_angle(image_np)
    face_present, face_data = detect_face(image_np)
    mrz_present, debug_image = detect_mrz_region(image_np)
    document_like = is_likely_document(image_np)
    is_blurry, fft_score = detect_blur_fft(image_np)
    image_glare_percentage = detect_face_glare(image_np)

    # Initialize debug dictionary with all keys
    debug = {
        "face_present": face_present,
        "face_box": tuple(face_data[0]) if face_present and len(face_data) > 0 else None,
        "face_blur_score": None,
        "face_glare_percentage": None,
        "brightness": brightness,
        "blur_score": fft_score,
        "angle": angle,
        "mrz_present": mrz_present,
        "document_like": document_like,
        "image_glare_percentage": image_glare_percentage,
        "final_score": 0
    }


    # -----------------------------
    # 2. EARLY STOPS (critical fails)
    # -----------------------------
    friendly_reasons = []
    if abs(angle) > 5:
        friendly_reasons.append(f"❌ Image not horizontally aligned ({angle:.1f}°). Hold document straight.")
        return 0, friendly_reasons, debug, image_np
    if fft_score <= 15:
        friendly_reasons.append("❌ Image is too blurry to process")
        return 0, friendly_reasons, debug, image_np
    if image_glare_percentage >= 40:
        friendly_reasons.append("❌ Strong glare detected")
        return 0, friendly_reasons, debug, image_np
    # if not face_present:
    #     friendly_reasons.append("❌ No face detected")
    #     return 0, friendly_reasons, debug, image_np

    # -----------------------------
    # 3. QUALITY SCORE SUMMARY
    # -----------------------------
    summary = {
        "Face": {"Detected": 0, "Not Blurry": 0, "No Glare": 0, "Total": 0},
        "Document": {"Detected": 0, "Brightness OK": 0, "Not Blurry": 0, "No Glare": 0, "Total": 0},
        "MRZ": {"Detected": 0, "Total": 0},
        "Final Score": 0
    }

    # ---- Face ---- (max 30)
    face_score = 0
    if face_present and len(face_data) > 0:
        summary["Face"]["Detected"] = 10
        face_score += 10

        x, y, w, h = face_data[0]
        face_crop = image_np[y:y + h, x:x + w]

        blur_ok = evaluate_blur(calculate_face_blur(face_crop, w, h))
        glare_ok = evaluate_glare(detect_face_glare(face_crop))

        if blur_ok:
            summary["Face"]["Not Blurry"] = 10
            face_score += 10
        if glare_ok:
            summary["Face"]["No Glare"] = 10
            face_score += 10

        debug["face_blur_score"] = calculate_face_blur(face_crop, w, h)
        debug["face_glare_percentage"] = detect_face_glare(face_crop)

    summary["Face"]["Total"] = face_score

    # ---- Document ---- (max 40)
    document_score = 0
    if document_like:
        summary["Document"]["Detected"] = 10
        document_score += 10
    if 80 <= brightness <= 220:
        summary["Document"]["Brightness OK"] = 10
        document_score += 10
    if fft_score > 15:
        summary["Document"]["Not Blurry"] = 10
        document_score += 10
    if image_glare_percentage < 40:
        summary["Document"]["No Glare"] = 10
        document_score += 10

    summary["Document"]["Total"] = document_score

    # ---- MRZ ---- (max 30)
    if mrz_present:
        mrz_score = 30
    else:
        mrz_score = 0
        debug_image = image_np.copy()
    summary["MRZ"]["Detected"] = mrz_score
    summary["MRZ"]["Total"] = mrz_score

    # ---- Final Score ----
    final_score = min(face_score + document_score + mrz_score, 100)
    summary["Final Score"] = final_score
    debug["final_score"] = final_score

    # -----------------------------
    # 4. Friendly Reasons (grouped)
    # -----------------------------
    friendly_reasons = []

    # ---- Face ----
    friendly_reasons.append(f"### Face {summary['Face']['Total']}/30")
    friendly_reasons.append(f"- Detected: {'Yes ✅' if summary['Face']['Detected'] else 'No ❌'}")
    friendly_reasons.append(f"- Blur: {'OK ✅' if summary['Face']['Not Blurry'] else 'Too blurry ❌'}")
    friendly_reasons.append(f"- Glare: {'OK ✅' if summary['Face']['No Glare'] else 'Too much ❌'}")

    # ---- Document ----
    friendly_reasons.append(f"### Document {summary['Document']['Total']}/40")
    friendly_reasons.append(f"- Layout: {'Detected ✅' if summary['Document']['Detected'] else 'Not detected ❌'}")
    friendly_reasons.append(f"- Brightness: {'Good ✅' if summary['Document']['Brightness OK'] else 'Poor ❌'}")
    friendly_reasons.append(f"- Blur: {'OK ✅' if summary['Document']['Not Blurry'] else 'Too blurry ❌'}")
    friendly_reasons.append(f"- Glare: {'OK ✅' if summary['Document']['No Glare'] else 'Too much ❌'}")

    # ---- MRZ ----
    friendly_reasons.append(f"### MRZ {summary['MRZ']['Total']}/30")
    friendly_reasons.append(f"- Detected: {'Yes ✅' if summary['MRZ']['Detected'] else 'No ❌'}")

    # ---- Final ----
    # friendly_reasons.append(f"### Final Score: {final_score}/100")

    return final_score, friendly_reasons, debug, debug_image


def draw_overlays(image_np, debug):
    """Draw detected face, MRZ, and document boxes."""
    annotated = image_np.copy()

    if debug.get("face_present") and debug.get("face_box"):
        x, y, w, h = debug["face_box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    doc_box = detect_document_edges(image_np)

    if doc_box:
        x, y, w, h = doc_box

        # ----- DRAW DOCUMENT EDGE -----
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            annotated, "Document Edge",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    return annotated

def convert_debug_to_summary(debug):
    """Convert internal debug dict to clean summary JSON."""
    return {
        "Face": {
            "Detected": debug.get("face_present", False),
            "Blur Score": debug.get("face_blur_score"),
            "Glare": debug.get("face_glare_percentage"),
            "Box": debug.get("face_box")
        },
        "Image": {
            "Brightness": debug.get("brightness"),
            "Blur": debug.get("blur_score"),
            "Rotation": debug.get("angle"),
            "Glare": debug.get("image_glare_percentage")
        },
        "Document": {
            "MRZ Detected": debug.get("mrz_present"),
            "Layout": debug.get("document_like")
        },
        "Final Score": debug.get("final_score", 0)
    }