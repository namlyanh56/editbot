from typing import Tuple
import cv2 as cv
import numpy as np
from PIL import Image
from .utils import (
    sample_text_color,
    detect_light_mode,
    estimate_background_color,
    compute_font_size_for_height,
    render_text_supersampled,
    variance_luminance,
    get_safe_font_path
)

def patch_and_replace_number(
    img_pil: Image.Image,
    digit_bbox: Tuple[int,int,int,int],
    old_number: str,
    new_number: str,
    font_path: str
) -> Image.Image:
    x,y,w,h = digit_bbox
    pad = 3
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_pil.width, x + w + pad)
    y1 = min(img_pil.height, y + h + pad)
    Wp = x1 - x0
    Hp = y1 - y0

    img_np = np.array(img_pil)
    patch = img_np[y0:y1, x0:x1].copy()

    var_lum = variance_luminance(patch)
    bg_color = estimate_background_color(img_np, (x0,y0,Wp,Hp))

    if var_lum < 80:
        patch[:,:,:] = bg_color
    else:
        gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        thr = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,15,8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        mask = cv.morphologyEx(thr, cv.MORPH_OPEN, kernel, iterations=1)
        inpainted = cv.inpaint(patch, mask, 4, cv.INPAINT_TELEA)
        patch = inpainted

    img_np[y0:y1, x0:x1] = patch

    light_mode = detect_light_mode(img_pil)
    try:
        text_color = sample_text_color(img_pil, digit_bbox)
    except:
        text_color = (17,17,17) if light_mode else (233,237,239)

    target_height = h
    try:
        font_size = compute_font_size_for_height(font_path, target_height)
    except FileNotFoundError:
        # fallback safe font
        font_size =  int(target_height * 0.9)

    rendered = render_text_supersampled(new_number, font_path, font_size, text_color, scale=3)

    if rendered.height > Hp:
        scale_factor = Hp / rendered.height
        new_w = max(1, int(rendered.width * scale_factor))
        rendered = rendered.resize((new_w, Hp), Image.Resampling.LANCZOS)

    out_pil = Image.fromarray(img_np)
    out_pil.paste(rendered, (x, y + (h - rendered.height)//2), rendered)
    return out_pil
