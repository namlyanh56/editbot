from typing import Tuple, Optional
import cv2 as cv
import numpy as np
from PIL import Image
from .utils import (
    sample_text_color,
    detect_light_mode,
    estimate_background_color,
    compute_font_size_for_height,
    render_text_supersampled,
    variance_luminance
)

def patch_and_replace_number(
    img_pil: Image.Image,
    digit_bbox: Tuple[int,int,int,int],
    old_number: str,
    new_number: str,
    font_path: str
) -> Image.Image:
    """
    Lakukan patch & substitusi angka.
    digit_bbox: (x,y,w,h) angka lama.
    """
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

    # Cek varian luminance untuk pilih metode
    var_lum = variance_luminance(patch)
    bg_color = estimate_background_color(img_np, (x0,y0,Wp,Hp))
    # Hilangkan angka lama
    if var_lum < 80:
        # Latar relatif homogen, cukup fill
        patch[:,:,:] = bg_color
    else:
        # Build mask digit
        gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        # adaptif threshold
        thr = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,15,8)
        # bersihkan sedikit
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        mask = cv.morphologyEx(thr, cv.MORPH_OPEN, kernel, iterations=1)
        inpainted = cv.inpaint(patch, mask, 3, cv.INPAINT_TELEA)
        patch = inpainted

    img_np[y0:y1, x0:x1] = patch

    # Warna teks baru: sampling di sekitar area, fallback mode
    light_mode = detect_light_mode(img_pil)
    # ambil warna referensi dari area lebih lebar (y - 2*h, dsb) kalau ingin; sederhana: gunakan sample_text_color pada digit bbox lama
    try:
        text_color = sample_text_color(img_pil, digit_bbox)
    except:
        text_color = (17,17,17) if light_mode else (233,237,239)

    # Tentukan font size agar tinggi cocok
    target_height = h
    font_size = compute_font_size_for_height(font_path, target_height)
    rendered = render_text_supersampled(new_number, font_path, font_size, text_color, scale=3)

    # Jika tinggi rendered > Hp, scale lagi
    if rendered.height > Hp:
        scale_factor = Hp / rendered.height
        new_w = max(1, int(rendered.width * scale_factor))
        rendered = rendered.resize((new_w, Hp), Image.Resampling.LANCZOS)

    # Tempel
    out_pil = Image.fromarray(img_np)
    out_pil.paste(rendered, (x, y + (h - rendered.height)//2), rendered)
    return out_pil
