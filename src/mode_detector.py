import json, re
from typing import Optional, Tuple
import numpy as np
import cv2 as cv
from PIL import Image
import pytesseract
from pytesseract import Output
from .config import OCR_LANG

# Muat konfigurasi mode
with open("config/modes.json","r",encoding="utf-8") as f:
    MODES_CFG = json.load(f)["modes"]

ANCHOR_WORDS = ("anggota","members")
DIGIT_REGEX = r"\d{1,5}"

def crop_line(img: Image.Image, mode_name: str):
    if mode_name not in MODES_CFG:
        raise ValueError(f"Mode tidak dikenal: {mode_name}")
    w,h = img.size
    r = MODES_CFG[mode_name]["line_region"]
    x = int(r["x_pct"] * w)
    y = int(r["y_pct"] * h)
    ww = int(r["w_pct"] * w)
    hh = int(r["h_pct"] * h)
    return img.crop((x,y,x+ww,y+hh)), (x,y,ww,hh)

def ocr_line_number(line_img: Image.Image):
    data = pytesseract.image_to_data(
        line_img,
        output_type=Output.DICT,
        lang=OCR_LANG,
        config="--psm 6"
    )
    n = len(data['text'])
    tokens = []
    for i in range(n):
        t = data['text'][i].strip()
        if not t: continue
        tokens.append((
            t,
            data['left'][i],
            data['top'][i],
            data['width'][i],
            data['height'][i]
        ))
    # Anchor method
    anchors = [t for t in tokens if t[0].lower() in ANCHOR_WORDS]
    for a in anchors:
        idx_a = tokens.index(a)
        ax, ay, aw, ah = a[1], a[2], a[3], a[4]
        for prev in reversed(tokens[:idx_a]):
            if re.fullmatch(DIGIT_REGEX, prev[0]) and abs(prev[2]-ay) < max(prev[4], ah)*0.7:
                return {
                    "number": prev[0],
                    "bbox": (prev[1], prev[2], prev[3], prev[4])
                }
    # Regex fallback pada gabungan string
    joined = " ".join(t[0] for t in tokens)
    m = re.search(r"(?:Grup|Group)\s*[·\.\-]?\s*(\d{1,5})\s+(?:anggota|members)", joined, re.IGNORECASE)
    if m:
        target = m.group(1)
        for t in tokens:
            if t[0] == target:
                return {"number": target, "bbox": (t[1],t[2],t[3],t[4])}
    return None

def contour_digit_fallback(line_img: Image.Image):
    """
    Cari cluster digit dengan threshold jika OCR gagal.
    """
    g = line_img.convert("L")
    arr = np.array(g)
    blur = cv.GaussianBlur(arr,(3,3),0)
    th = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,31,9)
    contours,_ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    h_line = arr.shape[0]
    boxes = []
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
        if h < h_line*0.3 or h > h_line*1.2:  # filter noise / outlier tinggi
            continue
        if w > h*3.5:   # kemungkinan kata, bukan digit cluster
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        return None
    boxes = sorted(boxes, key=lambda b: b[0])
    # gabung berdekatan
    merged = []
    cur = list(boxes[0])
    for b in boxes[1:]:
        if b[0] <= cur[0]+cur[2]+6:
            x0 = min(cur[0], b[0])
            y0 = min(cur[1], b[1])
            x1 = max(cur[0]+cur[2], b[0]+b[2])
            y1 = max(cur[1]+cur[3], b[1]+b[3])
            cur = [x0,y0,x1-x0,y1-y0]
        else:
            merged.append(tuple(cur))
            cur = list(b)
    merged.append(tuple(cur))
    # pilih cluster dengan rasio w/h wajar
    best = sorted(merged, key=lambda m: (abs((m[2]/(m[3]+1e-6))-2.0), m[0]))[0]
    return {"number": None, "bbox": best}

def detect_number_with_mode(full_img: Image.Image, mode_name: str):
    line_img, (lx,ly,lw,lh) = crop_line(full_img, mode_name)
    res = ocr_line_number(line_img)
    if res:
        bx,by,bw,bh = res["bbox"]
        return {
            "old_number": res["number"],
            "bbox_global": (lx+bx, ly+by, bw, bh),
            "method": "ocr_mode"
        }
    res2 = contour_digit_fallback(line_img)
    if res2:
        bx,by,bw,bh = res2["bbox"]
        return {
            "old_number": "?",
            "bbox_global": (lx+bx, ly+by, bw, bh),
            "method": "contour_mode"
        }
    return None
```

````json name=config/modes.json
{
  "modes": {
    "android_light": {
      "line_region": { "x_pct": 0.22, "y_pct": 0.155, "w_pct": 0.56, "h_pct": 0.045 },
      "note": "Baris anggota Android Light"
    },
    "android_dark": {
      "line_region": { "x_pct": 0.22, "y_pct": 0.165, "w_pct": 0.56, "h_pct": 0.045 },
      "note": "Android Dark"
    },
    "ios_light": {
      "line_region": { "x_pct": 0.18, "y_pct": 0.205, "w_pct": 0.64, "h_pct": 0.055 },
      "note": "iOS Light"
    },
    "ios_dark": {
      "line_region": { "x_pct": 0.18, "y_pct": 0.205, "w_pct": 0.64, "h_pct": 0.055 },
      "note": "iOS Dark"
    }
  }
}
