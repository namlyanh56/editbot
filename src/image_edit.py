from typing import Tuple
import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from .utils import (
    detect_light_mode,
    estimate_background_color,
    compute_font_size_for_height,
    render_text_supersampled,
    get_safe_font_path,
    sample_digit_color,
    adjust_font_width,
    variance_luminance,
    sample_color_from_bbox,
    sample_neutral_text_color
)

def _rebuild_line_background(img_np, line_bbox):
    x,y,w,h = line_bbox
    H, W = img_np.shape[:2]
    y_top = max(0, y-4); y_bot = min(H-1, y+h+3)
    strip_top = img_np[y_top:y_top+3, x:x+w]
    strip_bot = img_np[y_bot-2:y_bot+1, x:x+w]
    if strip_top.size==0 or strip_bot.size==0:
        med = np.median(img_np[y:y+h, x:x+w].reshape(-1,3), axis=0)
        img_np[y:y+h, x:x+w] = med
        return
    col_top = np.median(strip_top.reshape(-1,3), axis=0)
    col_bot = np.median(strip_bot.reshape(-1,3), axis=0)
    # Jika hampir sama, isi rata
    if np.linalg.norm(col_top-col_bot) < 3.0:
        img_np[y:y+h, x:x+w] = col_top
        return
    # Gradien vertikal
    for i in range(h):
        t = i/max(1,h-1)
        col = (1-t)*col_top + t*col_bot
        img_np[y+i, x:x+w] = col

def replace_members_line_full(
    img_pil: Image.Image,
    line_bbox: Tuple[int,int,int,int],
    digit_bbox: Tuple[int,int,int,int],
    anchor_bbox: Tuple[int,int,int,int],
    new_number: str,
    font_path: str
) -> Image.Image:
    x,y,w,h = line_bbox
    img_np = np.array(img_pil)

    # 1) Bersihkan latar baris
    _rebuild_line_background(img_np, line_bbox)

    # 2) Warna
    light_mode = detect_light_mode(img_pil)
    accent = sample_color_from_bbox(img_pil, anchor_bbox, prefer_green=True, light_mode=light_mode)
    neutral = sample_neutral_text_color(img_pil, line_bbox, light_mode)

    # 3) Ukuran font dari tinggi baris
    base_size = compute_font_size_for_height(font_path, h-2)
    safe_font = get_safe_font_path(font_path)

    # 4) Render segmen teks
    #    a) "Grup · " (abu-abu)
    #    b) angka baru (accent)
    #    c) " anggota" (accent)
    segA = "Grup · "
    segC = " anggota"

    # Render dengan supersampling 3x, lalu blur halus supaya anti-alias mirip sistem
    def render_seg(text, color, size):
        img = render_text_supersampled(text, safe_font, size, color, scale=3)
        return img.filter(ImageFilter.GaussianBlur(radius=0.35))

    # Tentukan tinggi final dari baseline digit
    # Kita ingin angka baru setinggi digit_bbox.h
    dbx, dby, dbw, dbh = digit_bbox
    # coba sizing angka
    tmp = render_seg(new_number, accent, base_size)
    # sesuaikan lebar vs digit lama agar kerning terasa sama
    if tmp.height != dbh:
        scale_h = dbh / max(1, tmp.height)
        tmp = tmp.resize((max(1,int(tmp.width*scale_h)), dbh), Image.Resampling.LANCZOS)
    # paskan lagi lebar dengan sedikit penyesuaian
    target_w = dbw
    if abs(tmp.width - target_w) > target_w*0.12:
        ratio = target_w / max(1, tmp.width)
        tmp = tmp.resize((max(1,int(tmp.width*ratio)), dbh), Image.Resampling.LANCZOS)
    segB_img = tmp

    # Render segA & segC dengan ukuran relatif sama baseline
    seg_size = max(6, int(base_size * (dbh/(tmp.height+1e-6))))
    segA_img = render_seg(segA, neutral, seg_size)
    segC_img = render_seg(segC, accent, seg_size)

    # 5) Komposisi: letakkan segA + segB + segC di baseline vertikal tengah baris
    canvas = Image.fromarray(img_np)
    # Hitung start-x: geser agar angka baru menggantikan posisi digit lama
    start_x_B = dbx
    # Letakkan segA sehingga ujung kanannya menempel ke awal angka
    start_x_A = max(x, start_x_B - segA_img.width - 2)
    # Letakkan segC tepat setelah angka
    start_x_C = min(x+w-segC_img.width, start_x_B + segB_img.width + 2)
    base_y = y + (h - segA_img.height)//2
    canvas.paste(segA_img, (start_x_A, base_y), segA_img)
    canvas.paste(segB_img, (start_x_B, dby + (dbh - segB_img.height)//2), segB_img)
    canvas.paste(segC_img, (start_x_C, base_y), segC_img)
    return canvas

def patch_and_replace_number(
    img_pil: Image.Image,
    digit_bbox: Tuple[int,int,int,int],
    old_number: str,
    new_number: str,
    font_path: str
) -> Image.Image:
    # Versi lama: tetap dipakai untuk fallback berbasis angka saja
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
        thr = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,17,7)
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
    rendered = adjust_font_width(font_path, base_size, w, h, new_number, text_color)
    new_h = rendered.height
    paste_y = y + (h - new_h)//2
    out_pil = Image.fromarray(img_np)
    out_pil.paste(rendered, (x, paste_y), rendered)
    return out_pil
