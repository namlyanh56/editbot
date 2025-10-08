from typing import Tuple
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2 as cv
import os

def get_safe_font_path(preferred_path: str) -> str:
    """
    Kembalikan path font valid; kalau preferred rusak atau tidak bisa dibuka,
    fallback ke font sistem.
    """
    candidates = [
        preferred_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                ImageFont.truetype(c, 16)
                return c
            except Exception:
                continue
    raise FileNotFoundError("Tidak menemukan font TTF valid (preferred & fallback gagal).")

def sample_text_color(img_pil, bbox):
    x,y,w,h = bbox
    region = img_pil.crop((x, y, x+w, y+h))
    g = region.convert("L")
    arr = np.array(g)
    flat = arr.flatten()
    perc_dark = np.percentile(flat, 30)
    perc_light = np.percentile(flat, 70)
    mean_all = np.mean(flat)
    rgb = np.array(region)
    light_mode = mean_all > 128
    if light_mode:
        mask = arr <= (perc_dark + 5)
        if mask.sum() < 10:
            return (17,17,17)
        cols = rgb[mask]
    else:
        mask = arr >= (perc_light - 5)
        if mask.sum() < 10:
            return (233,237,239)
        cols = rgb[mask]
    avg = cols.mean(axis=0)
    return tuple(int(c) for c in avg)

def detect_light_mode(img_pil):
    arr = np.array(img_pil.resize((32,32)).convert("L"))
    return arr.mean() > 128

def estimate_background_color(img_np, bbox, pad=4):
    h_img, w_img = img_np.shape[:2]
    x,y,w,h = bbox
    samples = []
    y_top = max(0, y - pad - 4)
    if y_top >= 0:
        samples.append(img_np[y_top:y_top+4, x:x+w])
    y_bottom = min(h_img-4, y + h + pad)
    if y_bottom+4 <= h_img:
        samples.append(img_np[y_bottom:y_bottom+4, x:x+w])
    if not samples:
        return (255,255,255)
    cat = np.concatenate(samples, axis=0)
    med = np.median(cat.reshape(-1,3), axis=0)
    return tuple(int(v) for v in med)

def compute_font_size_for_height(font_path, target_height, test_text="0123456789"):
    low, high = 5, 400
    best = low
    safe_path = get_safe_font_path(font_path)
    while low <= high:
        mid = (low + high)//2
        font = ImageFont.truetype(safe_path, mid)
        bbox = font.getbbox(test_text)
        height = bbox[3]-bbox[1]
        if height <= target_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best

def render_text_supersampled(text, font_path, font_size, color, scale=3):
    safe_path = get_safe_font_path(font_path)
    font = ImageFont.truetype(safe_path, font_size*scale)
    bbox = font.getbbox(text)
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=color)
    target_w = max(1, w//scale)
    target_h = max(1, h//scale)
    img_small = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img_small

def variance_luminance(patch):
    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    return gray.var()
