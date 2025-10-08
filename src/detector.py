import json, os, re
from typing import Optional, Tuple
import numpy as np
import cv2 as cv
from PIL import Image
import pytesseract
from pytesseract import Output
from .config import OCR_LANG

with open("config/modes.json","r",encoding="utf-8") as f:
    MODES_CFG = json.load(f)["modes"]

DIGIT_REGEX = r"\d{1,5}"
ANCHOR_WORDS = ("anggota","members")

def crop_line(img: Image.Image, mode_name: str):
    if mode_name not in MODES_CFG:
        raise ValueError("Mode tidak dikenal")
    w,h = img.size
    r = MODES_CFG[mode_name]["line_region"]
    x = int(r["x_pct"] * w)
    y = int(r["y_pct"] * h)
    ww = int(r["w_pct"] * w)
    hh = int(r["h_pct"] * h)
    return img.crop((x,y,x+ww,y+hh)), (x,y,ww,hh)

def ocr_line_number(line_img: Image.Image):
    # OCR khusus area kecil
    data = pytesseract.image_to_data(line_img, output_type=Output.DICT,
                                     lang=OCR_LANG,
                                     config="--psm 6")
    n = len(data['text'])
    tokens = []
    for i in range(n):
        t = data['text'][i].strip()
        if not t: continue
        tokens.append((t, data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
    # Cari anchor 'anggota/members'
    anchors = [tok for tok in tokens if tok[0].lower() in ANCHOR_WORDS]
    for a in anchors:
        # cari token sebelum anchor yang full digits
        ax,ay,aw,ah = a[1],a[2],a[3],a[4]
        # tokens sebelum anchor
        idx_a = tokens.index(a)
        for prev in reversed(tokens[:idx_a]):
            if re.fullmatch(DIGIT_REGEX, prev[0]):
                return {
                    "number": prev[0],
                    "bbox": (prev[1], prev[2], prev[3], prev[4])
                }
    # Regex fallback dari text gabungan
    joined = " ".join(t[0] for t in tokens)
    m = re.search(r"(?:Grup|Group)\s*[·\.\-]?\s*(\d{1,5})\s+(?:anggota|members)", joined, re.IGNORECASE)
    if m:
        # Ambil token yang cocok angka
        target = m.group(1)
        for t in tokens:
            if t[0] == target:
                return {"number": target, "bbox": (t[1],t[2],t[3],t[4])}
    return None

def contour_digit_fallback(line_img: Image.Image):
    """
    Jika OCR gagal: threshold + cari cluster digit (angka biasanya lebih tebal dari teks 'Grup').
    """
    img_np = np.array(line_img.convert("L"))
    blur = cv.GaussianBlur(img_np,(3,3),0)
    # adaptive threshold invert (putih digit)
    th = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,31,9)
    contours,_ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_line = img_np.shape[0]
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
        if h < h_line*0.3:  # terlalu kecil
            continue
        if h > h_line*1.1:
            continue
        if w > h*3.2:  # kemungkinan bukan digit tunggal (mungkin kata)
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        return None
    # Gabungkan box yang berdekatan secara horizontal (mewakili angka multi-digit)
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    cur = list(boxes[0])
    for b in boxes[1:]:
        if b[0] <= cur[0]+cur[2]+6:  # jarak kecil => sambung
            # expand
            x0 = min(cur[0], b[0])
            x1 = max(cur[0]+cur[2], b[0]+b[2])
            y0 = min(cur[1], b[1])
            y1 = max(cur[1]+cur[3], b[1]+b[3])
            cur = [x0,y0,x1-x0,y1-y0]
        else:
            merged.append(tuple(cur))
            cur = list(b)
    merged.append(tuple(cur))
    # Pilih cluster dengan rasio w/h <= 6 & posisi relatif di tengah vertikal
    best = sorted(merged, key=lambda m: (-m[2]/(m[3]+1e-6), m[0]))[0]
    return {"number": None, "bbox": best}  # number None (kita tidak tahu angka lama)

def detect_number_with_mode(full_img: Image.Image, mode_name: str):
    line_img, (lx,ly,lw,lh) = crop_line(full_img, mode_name)
    res = ocr_line_number(line_img)
    if res:
        bx,by,bw,bh = res["bbox"]
        # konversi ke koordinat global
        return {
            "old_number": res["number"],
            "bbox_global": (lx+bx, ly+by, bw, bh),
            "method": "ocr_mode"
        }
    # fallback contour
    res2 = contour_digit_fallback(line_img)
    if res2:
        bx,by,bw,bh = res2["bbox"]
        return {
            "old_number": "?",
            "bbox_global": (lx+bx, ly+by, bw, bh),
            "method": "contour_mode"
        }
    return None
