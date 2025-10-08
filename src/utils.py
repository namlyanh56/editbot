from typing import Tuple
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2 as cv
import os

def get_safe_font_path(preferred_path: str) -> str:
    candidates = [
        preferred_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                ImageFont.truetype(c, 20)
                return c
            except Exception:
                continue
    raise FileNotFoundError("Font valid tidak ditemukan.")

def detect_light_mode(img_pil):
    arr = np.array(img_pil.resize((48,48)).convert("L"))
    return arr.mean() > 128

def sample_digit_color(img_pil: Image.Image, digit_bbox: Tuple[int,int,int,int], light_mode: bool):
    x,y,w,h = digit_bbox
    region = img_pil.crop((x,y,x+w,y+h)).convert("RGB")
    g = region.convert("L")
    arr = np.array(g)
    rgb = np.array(region)
    flat = arr.flatten()
    if light_mode:
        thr = np.percentile(flat, 15)
        mask = arr <= thr
        if mask.sum() < 8:
            mask = arr <= np.percentile(flat, 30)
    else:
        thr = np.percentile(flat, 85)
        mask = arr >= thr
        if mask.sum() < 8:
            mask = arr >= np.percentile(flat, 70)
    cols = rgb[mask]
    med = np.median(cols, axis=0)
    return tuple(int(v) for v in med)

def estimate_background_color(img_np, bbox, pad=2):
    h_img, w_img = img_np.shape[:2]
    x,y,w,h = bbox
    strips=[]
    y1=max(0,y-pad-3)
    y2=min(h_img-4, y+h+pad)
    strips.append(img_np[y1:y1+3, x:x+w])
    strips.append(img_np[y2:y2+3, x:x+w])
    cat=np.concatenate(strips,axis=0)
    med=np.median(cat.reshape(-1,3),axis=0)
    return tuple(int(v) for v in med)

def compute_font_size_for_height(font_path, target_height):
    low, high = 4, 400
    best=low
    safe=get_safe_font_path(font_path)
    while low<=high:
        mid=(low+high)//2
        font=ImageFont.truetype(safe, mid)
        h = font.getbbox("0123456789")[3]
        if h <= target_height:
            best=mid
            low=mid+1
        else:
            high=mid-1
    return best

def render_text_supersampled(text, font_path, font_size, color, scale=3):
    safe=get_safe_font_path(font_path)
    font=ImageFont.truetype(safe, font_size*scale)
    bbox=font.getbbox(text)
    w=bbox[2]-bbox[0]; h=bbox[3]-bbox[1]
    img=Image.new("RGBA",(w,h),(0,0,0,0))
    d=ImageDraw.Draw(img)
    d.text((-bbox[0], -bbox[1]), text, font=font, fill=color)
    return img.resize((max(1,w//scale), max(1,h//scale)), Image.Resampling.LANCZOS)

def adjust_font_width(font_path, base_size, target_w, target_h, text, color):
    size=base_size
    rendered=render_text_supersampled(text, font_path, size, color, scale=3)
    for _ in range(12):
        diff = rendered.width - target_w
        if abs(diff) <= target_w*0.1:
            break
        ratio = target_w / max(1, rendered.width)
        size = max(4, int(size * (0.75 + 0.5*ratio)))
        rendered = render_text_supersampled(text, font_path, size, color, scale=3)
        if rendered.height > target_h*1.15:
            size = int(size*0.9)
            rendered = render_text_supersampled(text, font_path, size, color, scale=3)
    return rendered

def variance_luminance(patch):
    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    return gray.var()
