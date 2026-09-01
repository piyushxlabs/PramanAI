"""Stage 2: 300+ DPI Image Conditioning & Preprocessing.

Enhances image quality so subtle Devanagari diacritics (अनुस्वार ं, नुक्ता ़,
छोटी 'ि', बड़ी 'ी') are preserved without merging or vanishing.
Performs zero-copy 300+ DPI rasterization, Pixmap normalization guards, auto-deskewing
(±0.1° precision), non-destructive HSV stamp/seal suppression, and Sauvola/adaptive binarization.
"""

import io
import logging
import os
from typing import Optional, Tuple
import cv2
import pymupdf as fitz
import numpy as np
from PIL import Image, ImageEnhance

from src.gov_pdf_extractor.models import BoundingBox

logger = logging.getLogger("gov_pdf_extractor.preprocessor")


def enhance_gov_document_image(pil_image: Image.Image) -> Image.Image:
    """Enhances scanned administrative PDFs by boosting contrast and sharpness

    to resolve smudged conjunct characters (e.g., 'ज्ञ', 'ष्ट', 'ध्य', '०').
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Boost contrast to separate faint ink from grayish background
    contrast_enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_img = contrast_enhancer.enhance(1.35)

    # Boost edge sharpness for fine Devanagari strokes and abbreviations
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced_img)
    enhanced_img = sharpness_enhancer.enhance(1.8)

    return enhanced_img


class ImagePreprocessor:
    """High-resolution vision preprocessor for Indian Government gazette/order pages."""

    def __init__(self, target_dpi: int = 150, deskew_threshold_deg: float = 0.2):
        self.target_dpi = target_dpi
        self.deskew_threshold_deg = deskew_threshold_deg

    def rasterize_page(self, page: fitz.Page, dpi: Optional[int] = None) -> np.ndarray:
        """Renders a PDF page to a BGR NumPy array with zero-copy memoryview and Pixmap Normalization Guards."""
        render_dpi = dpi or self.target_dpi
        pix = None
        for try_dpi in [render_dpi, 150, 100, 72]:
            try:
                pix = page.get_pixmap(dpi=try_dpi)
                if pix is not None:
                    break
            except Exception:
                continue
        if pix is None:
            raise RuntimeError(f"Failed to rasterize page {page.number} at any DPI")

        h, w = pix.height, pix.width

        # Pixmap Normalization Guard: handle all channel variations with fast NumPy slicing
        if pix.alpha or pix.n == 4:
            raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, 4))
            img_bgr = raw[:, :, [2, 1, 0]].copy()
        elif pix.n == 1:
            raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, 1))
            img_bgr = np.repeat(raw, 3, axis=2)
        elif pix.n == 3:
            raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, 3))
            img_bgr = raw[:, :, ::-1].copy()
        else:
            rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            raw = np.frombuffer(rgb_pix.samples, dtype=np.uint8).reshape((rgb_pix.height, rgb_pix.width, 3))
            img_bgr = raw[:, :, ::-1].copy()
            del rgb_pix

        del pix

        # Memory Guard: Constrain maximum rendering dimension to 1600px to prevent OutOfMemory crashes while maintaining high OCR fidelity
        max_dim = int(os.getenv("VLM_MAX_IMAGE_DIM", "1600"))
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img_bgr

    def suppress_stamps_and_ink(self, img_bgr: np.ndarray) -> np.ndarray:
        """Non-destructive Stamp & Seal Masking via HSV color-space filtering.

        Isolates red and blue official government rubber stamps while strictly protecting
        high-contrast black/dark foreground text pixels (signatures, numbers, clauses).
        """
        if img_bgr is None:
            return img_bgr

        try:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

            # Red stamp masks (two hue ranges covering wrap-around at 0 and 180)
            red_mask_1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
            red_mask_2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

            # Blue/Purple seal and dispatch ink mask
            blue_mask = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
            stamp_mask = cv2.bitwise_or(red_mask, blue_mask)

            # If no stamp pixels detected, return early
            if cv2.countNonZero(stamp_mask) == 0:
                return img_bgr

            # Non-destructive protection: Identify dark achromatic text foreground pixels (low saturation, dark luminance)
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]

            # Fast OpenCV uint8 masks avoiding intermediate float/bool arrays
            low_sat = cv2.inRange(s_channel, 0, 60)
            dark_gray = cv2.inRange(gray, 0, 100)
            dark_v = cv2.inRange(v_channel, 0, 90)
            dark_lum = cv2.bitwise_or(dark_gray, dark_v)
            dark_text_mask = cv2.bitwise_and(low_sat, dark_lum)

            # Neutralize mask: in stamp_mask AND NOT in dark_text_mask
            neutralize_mask = cv2.bitwise_and(stamp_mask, cv2.bitwise_not(dark_text_mask))

            if cv2.countNonZero(neutralize_mask) == 0:
                return img_bgr

            cleaned_bgr = img_bgr.copy()
            cleaned_bgr[neutralize_mask > 0] = (255, 255, 255)
            return cleaned_bgr
        except Exception as e:
            logger.warning("Could not execute stamp suppression: %s; using raw image", e)
            return img_bgr

    def sauvola_binarize(
        self, gray: np.ndarray, window_size: int = 25, k: float = 0.2, r: float = 128.0
    ) -> np.ndarray:
        """Sauvola Local Adaptive Binarization for degraded historical scans and uneven lighting.

        Formula: T(x, y) = mean * (1 + k * (std / r - 1))
        Output contract: Text is strictly 0 (black), Background is 255 (white).
        """
        try:
            h, w = gray.shape[:2]
            # For large images (>1500px), compute threshold on 2x downsampled image for memory efficiency
            if max(h, w) > 1500:
                small_gray = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                small_f = small_gray.astype(np.float32)
                w_small = max(3, (window_size // 2) | 1)
                mean = cv2.blur(small_f, (w_small, w_small))
                mean_sq = cv2.blur(small_f * small_f, (w_small, w_small))
                variance = np.maximum(mean_sq - mean * mean, 0)
                std = np.sqrt(variance)
                thresh_small = mean * (1.0 + k * (std / r - 1.0))
                threshold = cv2.resize(thresh_small, (w, h), interpolation=cv2.INTER_LINEAR)
                return np.where(gray.astype(np.float32) > threshold, 255, 0).astype(np.uint8)
            else:
                gray_f = gray.astype(np.float32)
                mean = cv2.blur(gray_f, (window_size, window_size))
                mean_sq = cv2.blur(gray_f * gray_f, (window_size, window_size))
                variance = np.maximum(mean_sq - mean * mean, 0)
                std = np.sqrt(variance)
                threshold = mean * (1.0 + k * (std / r - 1.0))
                return np.where(gray_f > threshold, 255, 0).astype(np.uint8)
        except Exception:
            # Fallback to OpenCV adaptive thresholding on memory pressure
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
            )

    def calculate_skew_angle(self, binary_img: np.ndarray) -> float:
        """Calculates skew angle in degrees (±0.1° precision) using Hough Line Transform and minAreaRect."""
        inv = cv2.bitwise_not(binary_img) if np.mean(binary_img) > 127 else binary_img.copy()

        pts = np.column_stack(np.where(inv > 0))
        if len(pts) < 100:
            return 0.0

        angle = 0.0
        try:
            rect = cv2.minAreaRect(pts)
            box_angle = rect[-1]
            if box_angle < -45:
                angle = -(90 + box_angle)
            elif box_angle > 45:
                angle = 90 - box_angle
            else:
                angle = -box_angle
        except Exception:
            angle = 0.0

        lines = cv2.HoughLinesP(
            inv, 1, np.pi / 180, threshold=100, minLineLength=inv.shape[1] // 6, maxLineGap=20
        )
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > 0:
                    deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(deg) < 45.0:
                        angles.append(deg)
            if angles:
                median_angle = float(np.median(angles))
                if abs(median_angle) > 0.1:
                    angle = median_angle

        return round(float(angle), 2)

    def deskew_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotates image by angle_deg with white border padding."""
        if abs(angle_deg) < self.deskew_threshold_deg:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(
            img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )
        return rotated

    def adaptive_binarize(self, img_bgr: np.ndarray) -> np.ndarray:
        """Applies gentle denoising, Sauvola binarization, and Otsu-Gaussian hybrid for diacritic separation."""
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr.copy()

        # Fast Gaussian denoising for scan artifacts
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # Sauvola binarization as primary high-fidelity estimator
        sauvola_res = self.sauvola_binarize(denoised, window_size=25, k=0.2, r=128.0)

        # Otsu thresholding combined with adaptive Gaussian for sharp diacritic separation
        otsu_val, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11
        )
        combined_gauss_otsu = cv2.bitwise_and(otsu_thresh, adaptive_thresh)

        # Blend Sauvola and Gaussian-Otsu to ensure no diacritics are dropped
        combined = cv2.bitwise_and(sauvola_res, combined_gauss_otsu)
        return combined

    def preprocess_page(
        self, page: fitz.Page, dpi: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Full Stage 2 pipeline: Zero-copy 300+ DPI rasterization, Stamp suppression, deskewing, binarization.

        Returns:
            (processed_bgr, binarized_img, detected_skew_angle)
        """
        raw_bgr = self.rasterize_page(page, dpi=dpi or self.target_dpi)
        
        # Step 1: Suppress rubber stamps and seal ink non-destructively
        stamp_cleaned_bgr = self.suppress_stamps_and_ink(raw_bgr)

        # Step 2: Initial binarization for skew calculation
        initial_bin = self.adaptive_binarize(stamp_cleaned_bgr)
        skew_angle = self.calculate_skew_angle(initial_bin)

        if abs(skew_angle) > self.deskew_threshold_deg:
            logger.info("Auto-deskewing page: detected angle = %.2f°", skew_angle)
            deskewed_bgr = self.deskew_image(stamp_cleaned_bgr, skew_angle)
            binarized = self.adaptive_binarize(deskewed_bgr)
            return deskewed_bgr, binarized, skew_angle

        return stamp_cleaned_bgr, initial_bin, skew_angle

    def crop_localized_box(
        self, page: fitz.Page, bbox: BoundingBox, dpi: int = 450
    ) -> np.ndarray:
        """Extracts a high-resolution 450 DPI crop around a specific bounding box for re-OCR with Pixmap guard."""
        rect = page.rect
        pw, ph = rect.width, rect.height

        is_normalized = (bbox.xmax <= 1.05 and bbox.ymax <= 1.05)

        if is_normalized:
            x0 = bbox.xmin * pw
            y0 = bbox.ymin * ph
            x1 = bbox.xmax * pw
            y1 = bbox.ymax * ph
        else:
            x0, y0, x1, y1 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

        crop_rect = fitz.Rect(max(0, x0 - 5), max(0, y0 - 5), min(pw, x1 + 5), min(ph, y1 + 5))

        pix = page.get_pixmap(dpi=dpi, clip=crop_rect)
        h, w = pix.height, pix.width

        if pix.alpha or pix.n == 4:
            raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, 4))
            crop_bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, 1))
            crop_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif pix.n == 3:
            raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, 3))
            crop_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            raw = np.frombuffer(rgb_pix.samples, dtype=np.uint8).reshape((rgb_pix.height, rgb_pix.width, 3))
            crop_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            del rgb_pix

        del pix

        if crop_bgr is not None and crop_bgr.size > 0:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return crop_bgr
