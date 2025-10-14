import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2 as cv
from PIL import Image, ImageOps
from .utils import (
    detect_light_mode,
    estimate_background_color,
    variance_luminance,
    sample_color_from_bbox,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_ROOT = BASE_DIR / "templates"

MODES = ("android_light", "android_dark")

def _mode_from_image(img_pil: Image.Image) -> str:
    return "android_light" if detect_light_mode(img_pil) else "android_dark"

def _available_heights(mode_dir: Path) -> List[int]:
    heights = []
    if not mode_dir.exists():
        return heights
    for p in mode_dir.iterdir():
        if p.is_dir() and p.name.startswith("h"):
            try:
                heights.append(int(p.name[1:]))
            except:
                pass
    return sorted(heights)

def _pick_best_height(target_h: int, heights: List[int]) -> Optional[int]:
    if not heights:
        return None
    return sorted(heights, key=lambda h: abs(h - target_h))[0]

def _load_digit_set(dir_h: Path) -> Dict[str, Image.Image]:
    out = {}
    for d in "0123456789":
        fp = dir_h / f"{d}.png"
        if not fp.exists():
            continue
        try:
            img = Image.open(fp).convert("RGBA")
            out[d] = img
        except:
            pass
    return out

def _tint_rgba_white(img_rgba: Image.Image, color_rgb: Tuple[int,int,int]) -> Image.Image:
    """
    Recolor glyph putih transparan menjadi color_rgb, mempertahankan alpha.
    """
    arr = np.array(img_rgba)
    if arr.shape[2] == 4:
        R,G,B,A = arr[:,:,0],arr[:,:,1],arr[:,:,2],arr[:,:,3]
        # Map semua piksel RGB ke target color, simpan alpha apa adanya
        out = np.zeros_like(arr)
        out[:,:,0] = color_rgb[0]
        out[:,:,1] = color_rgb[1]
        out[:,:,2] = color_rgb[2]
        out[:,:,3] = A
        return Image.fromarray(out, mode="RGBA")
    else:
        return img_rgba

def _cleanup_number_background(img_np: np.ndarray, digit_bbox: Tuple[int,int,int,int]) -> None:
    """
    Bersihkan latar angka: adaptive inpaint jika ada tekstur, jika rata → isi dengan median strip.
    """
    x,y,w,h = digit_bbox
    pad = 2
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_np.shape[1], x + w + pad)
    y1 = min(img_np.shape[0], y + h + pad)

    patch = img_np[y0:y1, x0:x1].copy()
    var = variance_luminance(patch)
    bg = estimate_background_color(img_np, (x,y,w,h))
    if var < 120:
        patch[:,:,:] = bg
    else:
        gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        thr = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,17,7)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        mask = cv.morphologyEx(thr, cv.MORPH_OPEN, kernel, iterations=1)
        inpainted = cv.inpaint(patch, mask, 3, cv.INPAINT_TELEA)
        patch = inpainted
    img_np[y0:y1, x0:x1] = patch

def patch_with_templates(
    img_pil: Image.Image,
    digit_bbox: Tuple[int,int,int,int],
    anchor_bbox: Tuple[int,int,int,int],
    number: str,
    prefer_mode: Optional[str] = None,
    max_scale_delta: float = 0.08
) -> Image.Image:
    """
    Tempel angka baru memakai template digit.
    - digit_bbox: bbox angka lama (global)
    - anchor_bbox: bbox kata 'anggota' (sumber warna aksen)
    - number: string angka baru
    - prefer_mode: 'android_light' atau 'android_dark' (opsional). Jika None → auto dari gambar.
    - max_scale_delta: batas perubahan skala dari tinggi template agar tidak blur.
    """
    mode = prefer_mode or _mode_from_image(img_pil)
    if mode not in MODES:
        raise RuntimeError(f"Mode tidak didukung: {mode}")

    x,y,w,h = digit_bbox

    # 1) Pilih set template
    mode_dir = TEMPLATES_ROOT / mode
    heights = _available_heights(mode_dir)
    if not heights:
        raise FileNotFoundError(f"Tidak ada folder tinggi di {mode_dir}. Buat misal {mode_dir/'h28'} dan upload 0-9.png.")
    best_h = _pick_best_height(h, heights)
    dir_h = mode_dir / f"h{best_h}"
    digits = _load_digit_set(dir_h)
    if len(digits) < 10:
        raise FileNotFoundError(f"Digit 0-9 belum lengkap di {dir_h}. Ditemukan: {sorted(digits.keys())}")

    # 2) Siapkan warna target (dari anchor hijau “anggota”)
    light = detect_light_mode(img_pil)
    color_rgb = sample_color_from_bbox(img_pil, anchor_bbox, prefer_green=True, light_mode=light)

    # 3) Bersihkan latar belakang angka lama
    img_np = np.array(img_pil)
    _cleanup_number_background(img_np, digit_bbox)

    # 4) Siapkan glyph untuk setiap digit, dengan scaling terbatas
    target_h = h
    # Faktor skala dari template -> target
    scale = target_h / max(1, best_h)
    if abs(scale - 1.0) > max_scale_delta:
        # batasi agar tetap tajam
        scale = 1.0 + (max_scale_delta if scale > 1.0 else -max_scale_delta)
        target_h = int(best_h * scale)

    glyphs = []
    for ch in number:
        g = digits.get(ch)
        if g is None:
            raise ValueError(f"Template untuk digit '{ch}' tidak ada di {dir_h}.")
        # Pastikan warna
        g_col = _tint_rgba_white(g, color_rgb)
        if g_col.height != target_h:
            new_w = max(1, int(g_col.width * (target_h / g_col.height)))
            g_col = g_col.resize((new_w, target_h), Image.Resampling.LANCZOS)
        glyphs.append(g_col)

    # 5) Hitung total width & posisi paste (center ke bbox lama)
    total_w = sum(g.width for g in glyphs)
    start_x = x + max(0, (w - total_w)//2)
    # Baseline vertical center di digit_bbox
    start_y = y + (h - target_h)//2

    # 6) Paste
    out_pil = Image.fromarray(img_np)
    cursor_x = start_x
    for g in glyphs:
        out_pil.paste(g, (cursor_x, start_y), g)
        cursor_x += g.width

    return out_pil
