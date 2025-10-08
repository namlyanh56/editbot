"""
detector.py
Menyediakan dua fungsi yang diimpor bot.py:
 - ocr_find_member_number(img_pil, debug=False)
 - load_fallback_region(img_pil, regions_path="config/regions.json")

Strategi ocr_find_member_number:
 1. Preprocess (resize & histogram equalize).
 2. OCR full -> token data (pytesseract image_to_data).
 3. Gabungkan digit berurutan yang terpisah (1 2 3 -> 123).
 4. Cari anchor token 'anggota' / 'members' -> angka di kiri baris yang sama (tinggi hampir sama).
 5. Jika gagal, regex pada text gabungan "Grup ..." -> mapping kembali ke token angka.
 6. Jika tetap gagal, return None (bot akan fallback ke load_fallback_region).
"""

import json
import re
import os
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pytesseract
from pytesseract import Output
from .config import TESSERACT_CMD, OCR_LANG

# Konfigurasi path tesseract bila disediakan
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz·:"

# ---------- Preprocess ----------

def _preprocess(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = 2 if max(w, h) < 1400 else 1.5
    resized = img.resize((int(w*scale), int(h*scale)), Image.Resampling.BICUBIC)
    gray = ImageOps.grayscale(resized)
    arr = np.array(gray)
    arr_eq = _hist_eq(arr)
    return Image.fromarray(arr_eq)

def _hist_eq(arr: np.ndarray) -> np.ndarray:
    flat = arr.flatten()
    hist, _ = np.histogram(flat, 256, [0,256])
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
    cdf = cdf.astype(np.uint8)
    return cdf[flat].reshape(arr.shape)

# ---------- OCR helpers ----------

def _merge_digit_tokens(tokens: List[Tuple[int,str,int,int,int,int]]) -> List[Tuple[int,str,int,int,int,int]]:
    """
    Gabungkan token digit tunggal yang sebaris menjadi satu angka.
    tokens item: (idx, text, x, y, w, h)
    """
    merged = []
    buf = []
    baseline_y = None
    for idx,t,x,y,w,h in tokens:
        if re.fullmatch(r"\d", t):
            if baseline_y is None or abs(y - baseline_y) < h*0.6:
                buf.append((idx,t,x,y,w,h))
                baseline_y = y
                continue
        if buf:
            merged.append(_collapse(buf))
            buf = []
            baseline_y = None
        merged.append((idx,t,x,y,w,h))
    if buf:
        merged.append(_collapse(buf))
    return merged

def _collapse(buf):
    # buf: list of (idx,t,x,y,w,h) digit tunggal
    text = "".join(b[1] for b in buf)
    x0 = min(b[2] for b in buf)
    y0 = min(b[3] for b in buf)
    x1 = max(b[2]+b[4] for b in buf)
    hmax = max(b[5] for b in buf)
    return (buf[0][0], text, x0, y0, x1 - x0, hmax)

def _save_debug_image(pre, boxes, name="debug_ocr.png"):
    try:
        dbg = pre.convert("RGB").copy()
        dr = ImageDraw.Draw(dbg)
        for (x,y,w,h,t) in boxes:
            dr.rectangle([x,y,x+w,y+h], outline="red", width=2)
            dr.text((x, max(0,y-12)), t, fill="red")
        dbg.save(name)
    except Exception:
        pass

# ---------- Public OCR function ----------

def ocr_find_member_number(img_pil: Image.Image, debug: bool=False) -> Optional[Tuple[int,int,int,int,str]]:
    """
    Return (x,y,w,h,number) atau None.
    """
    pre = _preprocess(img_pil)

    cfg = f"--psm 6 -c tessedit_char_whitelist={WHITELIST}"
    data = pytesseract.image_to_data(pre, output_type=Output.DICT, lang=OCR_LANG, config=cfg)

    tokens_raw = []
    for i,txt in enumerate(data['text']):
        t = txt.strip()
        if not t: 
            continue
        tokens_raw.append((
            i,
            t,
            data['left'][i],
            data['top'][i],
            data['width'][i],
            data['height'][i]
        ))

    if not tokens_raw:
        return None

    tokens = _merge_digit_tokens(tokens_raw)

    # 1. Anchor method (anggota/members)
    anchors = [tok for tok in tokens if tok[1].lower() in ("anggota","members")]
    candidates = []
    for anchor in anchors:
        a_idx,a_text,a_x,a_y,a_w,a_h = anchor
        # Mundur beberapa token untuk cari angka
        for prev in reversed(tokens[:tokens.index(anchor)]):
            if re.fullmatch(r"\d{1,5}", prev[1]) and abs(prev[3]-a_y) < max(prev[5],a_h)*0.7:
                candidates.append(prev)
                break

    if candidates:
        # pilih angka dengan h terkecil (lebih konsisten) atau paling kiri
        best = sorted(candidates, key=lambda c: (c[3], c[2]))[0]
        _, num, x, y, w, h = best
        if debug:
            _save_debug_image(pre, [(x,y,w,h,num)], "debug_anchor.png")
        # scale back
        scale_x = img_pil.width / pre.width
        scale_y = img_pil.height / pre.height
        return (int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y), num)

    # 2. Regex fallback
    full_text = " ".join(t[1] for t in tokens)
    full_text_norm = full_text.replace("•","·")
    rgx_list = [
        r"Grup\s*[·\.\-]?\s*(\d{1,5})\s+anggota",
        r"Group\s*[·\.\-]?\s*(\d{1,5})\s+members"
    ]
    target_num = None
    for rg in rgx_list:
        m = re.search(rg, full_text_norm, flags=re.IGNORECASE)
        if m:
            target_num = m.group(1)
            break
    if target_num:
        # cari token angka sama
        for tok in tokens:
            if tok[1] == target_num:
                _, num, x,y,w,h = tok
                scale_x = img_pil.width / pre.width
                scale_y = img_pil.height / pre.height
                return (int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y), num)

    return None

# ---------- Fallback manual region ----------

def load_fallback_region(img_pil: Image.Image, regions_path="config/regions.json") -> Optional[Tuple[int,int,int,int]]:
    if not os.path.exists(regions_path):
        return None
    try:
        with open(regions_path,'r',encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return None
    w,h = img_pil.size
    profiles = cfg.get("profiles", [])
    # Prefer yang bukan generic (priority 1), generic (priority 2)
    candidates = []
    for p in profiles:
        if p["match_min_width"] <= w <= p["match_max_width"]:
            r = p["region"]
            x = int(r["x_pct"] * w)
            y = int(r["y_pct"] * h)
            ww = int(r["w_pct"] * w)
            hh = int(r["h_pct"] * h)
            priority = 2 if "generic" in p["name"] else 1
            candidates.append((priority,(x,y,ww,hh)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
