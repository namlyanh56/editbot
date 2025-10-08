from typing import Tuple
import cv2 as cv
import numpy as np
from PIL import Image
from .utils import (
    detect_light_mode,
    estimate_background_color,
    compute_font_size_for_height,
    render_text_supersampled,
    get_safe_font_path,
    sample_digit_color,
    adjust_font_width,
    variance_luminance
)

def patch_and_replace_number(
    img_pil: Image.Image,
    digit_bbox: Tuple[int,int,int,int],
    old_number: str,
    new_number: str,
    font_path: str
) -> Image.Image:
    x,y,w,h = digit_bbox
    pad = 2
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_pil.width, x + w + pad)
    y1 = min(img_pil.height, y + h + pad)

    img_np = np.array(img_pil)
    patch = img_np[y0:y1, x0:x1].copy()

    var_lum = variance_luminance(patch)
    bg_color = estimate_background_color(img_np, (x,y,w,h))

    if var_lum < 120:
        patch[:,:,:] = bg_color
    else:
        gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        thr = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY_INV,17,7)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        mask = cv.morphologyEx(thr, cv.MORPH_OPEN, kernel, iterations=1)
        inpainted = cv.inpaint(patch, mask, 3, cv.INPAINT_TELEA)
        patch = inpainted

    img_np[y0:y1, x0:x1] = patch

    light_mode = detect_light_mode(img_pil)
    try:
        text_color = sample_digit_color(img_pil, digit_bbox, light_mode)
    except:
        text_color = (25,25,25) if light_mode else (235,235,235)

    base_size = compute_font_size_for_height(font_path, h)
    # Render & width matching
    rendered = adjust_font_width(font_path, base_size, w, h, new_number, text_color)

    # Align baseline (center vertically)
    new_h = rendered.height
    paste_y = y + (h - new_h)//2
    out_pil = Image.fromarray(img_np)
    out_pil.paste(rendered, (x, paste_y), rendered)
    return out_pil
