import cv2 as cv
import numpy as np
from typing import Optional, Tuple, Dict, List
from PIL import Image
import pytesseract
from pytesseract import Output
from .config import OCR_LANG

# Persentase region baris Android (hasil dari contoh-contoh kamu)
REGION_ANDROID_LIGHT = dict(x_pct=0.22, y_pct=0.155, w_pct=0.56, h_pct=0.045)
REGION_ANDROID_DARK  = dict(x_pct=0.22, y_pct=0.165, w_pct=0.56, h_pct=0.045)

WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz·:"

def _is_light(img: Image.Image) -> bool:
    arr = np.array(img.resize((48,48)).convert("L"))
    return arr.mean() > 128

def _get_line_region(img: Image.Image) -> Tuple[int,int,int,int]:
    w,h = img.size
    if _is_light(img):
        r = REGION_ANDROID_LIGHT
    else:
        r = REGION_ANDROID_DARK
    x = int(r["x_pct"] * w)
    y = int(r["y_pct"] * h)
    ww = int(r["w_pct"] * w)
    hh = int(r["h_pct"] * h)
    # beri margin kecil agar aman
    x = max(0, x - int(0.01*w))
    y = max(0, y - int(0.005*h))
    ww = min(w - x, ww + int(0.02*w))
    hh = min(h - y, hh + int(0.01*h))
    return (x,y,ww,hh)

def _green_mask(line_img: Image.Image, light: bool) -> np.ndarray:
    """Ambil masker hijau aksen WA pada line_img."""
    bgr = cv.cvtColor(np.array(line_img), cv.COLOR_RGB2BGR)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    # H: 0..179, S/V:0..255 — WA accent: sekitar hijau kebiruan
    if light:
        lower = np.array([40, 80, 120])   # adjust videns
        upper = np.array([95, 255, 255])
    else:
        lower = np.array([40, 70, 80])
        upper = np.array([95, 255, 255])
    mask = cv.inRange(hsv, lower, upper)
    # bersihkan noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    return mask

def _ocr_tokens(img: Image.Image) -> Dict[str, List]:
    data = pytesseract.image_to_data(
        img, output_type=Output.DICT, lang=OCR_LANG, config=f"--psm 6 -c tessedit_char_whitelist={WHITELIST}"
    )
    return data

def _bbox_intersection_ratio(mask: np.ndarray, box: Tuple[int,int,int,int]) -> float:
    x,y,w,h = box
    H,W = mask.shape[:2]
    x2 = min(W, x+w); y2 = min(H, y+h)
    x1 = max(0, x); y1 = max(0, y)
    if x2<=x1 or y2<=y1: return 0.0
    roi = mask[y1:y2, x1:x2]
    return float((roi>0).sum()) / float(roi.size + 1e-6)

def detect_android_members_line(img: Image.Image) -> Optional[Dict]:
    """
    Deteksi khusus Android:
      - Temukan line region
      - Masker hijau
      - OCR pada line region → pilih token yang overlap dengan hijau
      - Ambil 'anggota/members' + angka sebelum-nya
    Return:
      { "digit_bbox": (x,y,w,h), "anchor_bbox": (x,y,w,h), "line_bbox": (x,y,w,h), "number": "40" }
    """
    x,y,w,h = _get_line_region(img)
    crop = img.crop((x,y,x+w,y+h))
    light = _is_light(img)
    mask = _green_mask(crop, light)

    data = _ocr_tokens(crop)
    n = len(data['text'])
    tokens = []
    for i in range(n):
        t = (data['text'][i] or "").strip()
        if not t: continue
        bx = data['left'][i]; by = data['top'][i]; bw = data['width'][i]; bh = data['height'][i]
        # hanya token yang benar-benar hijau
        if _bbox_intersection_ratio(mask, (bx,by,bw,bh)) < 0.25:
            continue
        tokens.append((t, bx,by,bw,bh))

    if not tokens:
        return None

    anchors = [tok for tok in tokens if tok[0].lower()=="anggota" or tok[0].lower()=="members"]
    if not anchors:
        # kadang 'anggota' terdeteksi pecah → ambil token panjang mendekati 'anggota'
        anchors = [tok for tok in tokens if len(tok[0])>=5 and tok[0][0].isalpha()]
    if not anchors:
        return None

    # pilih anchor paling kanan
    anchor = sorted(anchors, key=lambda t: t[1], reverse=True)[0]
    ax,ay,aw,ah = anchor[1],anchor[2],anchor[3],anchor[4]

    # kandidat angka = token di sisi kiri anchor, satu baris (y sebaris), terdiri dari digit
    candid = []
    for t,bx,by,bw,bh in tokens:
        if bx >= ax: continue
        if abs((by+bh/2) - (ay+ah/2)) > max(bh,ah)*0.7: continue
        if all(ch.isdigit() for ch in t):
            candid.append((t,bx,by,bw,bh))
    if not candid:
        # jika tess memecah digit, gabungkan token digit kecil yang saling berdampingan
        digits = sorted([tok for tok in tokens if all(ch.isdigit() for ch in tok[0])], key=lambda z:z[1])
        merged = []
        if digits:
            cur = list(digits[0])
            for tok in digits[1:]:
                _,bx,by,bw,bh = tok
                if bx <= cur[1]+cur[3]+6 and abs(by-cur[2])<max(bh,cur[4])*0.7:
                    x0 = min(cur[1], bx); y0 = min(cur[2], by)
                    x1 = max(cur[1]+cur[3], bx+bw); y1 = max(cur[2]+cur[4], by+bh)
                    cur = [cur[0]+tok[0], x0, y0, x1-x0, y1-y0]
                else:
                    merged.append(tuple(cur)); cur=list(tok)
            merged.append(tuple(cur))
        candid = merged

    if not candid:
        return None

    # pilih angka paling dekat ke anchor dari kiri
    best = sorted(candid, key=lambda z: (ax - (z[1]+z[3]), abs((z[2]+z[4]/2)-(ay+ah/2))))[0]
    num = ''.join(ch for ch in best[0] if ch.isdigit()) or "?"
    bx,by,bw,bh = best[1],best[2],best[3],best[4]

    # kembalikan ke koordinat global
    return {
        "digit_bbox": (x+bx, y+by, bw, bh),
        "anchor_bbox": (x+ax, y+ay, aw, ah),
        "line_bbox": (x, y, w, h),
        "number": num
    }
