import json, re, os
from typing import Optional, Tuple, List
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from .config import TESSERACT_CMD, OCR_LANG

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz·:"

def _preprocess(img: Image.Image) -> Image.Image:
    """
    Perbesar & tingkatkan kontras agar OCR digit kecil lebih stabil.
    """
    w, h = img.size
    scale = 2 if max(w, h) < 1400 else 1.5
    new_img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.BICUBIC)
    gray = ImageOps.grayscale(new_img)
    # light adaptive normalize
    arr = np.array(gray)
    arr_eq = cv_equalize(arr)
    return Image.fromarray(arr_eq)

def cv_equalize(arr):
    # histogram equalization sederhana (tanpa OpenCV untuk menghindari overhead)
    flat = arr.flatten()
    hist, bins = np.histogram(flat, 256, [0,256])
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
    cdf = cdf.astype(np.uint8)
    return cdf[flat].reshape(arr.shape)

def _raw_ocr_text(img_pil: Image.Image) -> str:
    cfg = f"--psm 6 -c tessedit_char_whitelist={WHITELIST}"
    return pytesseract.image_to_string(img_pil, lang=OCR_LANG, config=cfg)

def _save_debug(img_pil, boxes, fname):
    dbg = img_pil.convert("RGB").copy()
    dr = ImageDraw.Draw(dbg)
    for (x,y,w,h,t) in boxes:
        dr.rectangle([x,y,x+w,y+h], outline="red", width=2)
        dr.text((x,y-12), t, fill="red")
    dbg.save(fname)

def _merge_digit_tokens(tokens: List[Tuple[int,str,int,int,int,int]]) -> List[Tuple[int,str,int,int,int,int]]:
    """
    Gabungkan token berurutan yang masing2 hanya 1 digit menjadi satu angka (untuk kasus OCR memecah).
    tokens: list (index, text, x,y,w,h)
    """
    merged = []
    buffer = []
    last_y = None
    for idx,t,x,y,w,h in tokens:
        if re.fullmatch(r"\d", t):
            if last_y is None or abs(y - last_y) < h*0.6:
                buffer.append((idx,t,x,y,w,h))
                last_y = y
                continue
        # flush buffer
        if buffer:
            num_text = "".join(bt for _,bt,_,_,_,_ in buffer)
            x0 = min(bx for _,_,bx,_,_,_ in buffer)
            y0 = min(by for _,_,_,by,_,_ in buffer)
            x1 = max(bx+bw for _,_,bx,_,bw,_ in buffer)
            hmax = max(bh for _,_,_,_,_,bh in buffer)
            merged.append((buffer[0][0], num_text, x0, y0, x1-x0, hmax))
            buffer = []
            last_y = None
        merged.append((idx,t,x,y,w,h))
    if buffer:
        num_text = "".join(bt for _,bt,_,_,_,_ in buffer)
        x0 = min(bx for _,_,bx,_,_,_ in buffer)
        y0 = min(by for _,_,_,by,_,_ in buffer)
        x1 = max(bx+bw for _,_,bx,_,bw,_ in buffer)
        hmax = max(bh for _,_,_,_,_,bh in buffer)
        merged.append((buffer[0][0], num_text, x0, y0, x1-x0, hmax))
    return merged

def ocr_find_member_number(img_pil: Image.Image, debug=False) -> Optional[Tuple[int,int,int,int,str]]:
    """
    Strategi:
      1. Preprocess & OCR full.
      2. Regex 'Grup' / 'Group' + 'anggota' / 'members'.
      3. Token pass: cari 'anggota/members', cari digit di baris sama atau dekat (termasuk gabungan digit).
      4. Fallback: cari pola 'Grup' lalu angka lalu 'anggota'.
    """
    pre = _preprocess(img_pil)
    raw = _raw_ocr_text(pre)
    raw_norm = raw.replace("•","·")
    # Simpan raw untuk debugging cepat (opsional)
    if debug:
        with open("raw_ocr.txt","w",encoding="utf-8") as f:
            f.write(raw_norm)

    data = pytesseract.image_to_data(pre, output_type=Output.DICT,
                                     lang=OCR_LANG,
                                     config=f"--psm 6 -c tessedit_char_whitelist={WHITELIST}")
    tokens = []
    n = len(data['text'])
    for i in range(n):
        t = data['text'][i].strip()
        if not t:
            continue
        tokens.append((i, t, data['left'][i], data['top'][i], data['width'][i], data['height'][i]))

    # Merge digit tokens
    tokens_merged = _merge_digit_tokens(tokens)

    # Buat map index -> tuple
    # 1. Cari 'anggota' / 'members'
    anchors = [tok for tok in tokens_merged if tok[1].lower() in ("anggota","members")]
    cand_boxes = []
    for anchor in anchors:
        _,_,ax,ay,aw,ah = anchor
        # scan backward beberapa token
        for tok in reversed(tokens_merged[:tokens_merged.index(anchor)]):
            _,text,x,y,w,h = tok
            if re.fullmatch(r"\d{1,5}", text) and abs(y - ay) < max(h,ah)*0.7:
                cand_boxes.append((x,y,w,h,text))
                break

    if cand_boxes:
        # pilih yang posisinya paling kiri (umumnya benar)
        best = sorted(cand_boxes, key=lambda b: b[0])[0]
        if debug:
            _save_debug(pre, [(best[0],best[1],best[2],best[3],best[4])], "debug_ocr.png")
        # Skala balik koordinat ke img_pil asli
        scale_x = img_pil.width / pre.width
        scale_y = img_pil.height / pre.height
        x,y,w,h,text = best
        return (int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y), text)

    # Regex fallback dari raw (tanpa bbox)
    rgx_list = [
        r"Grup\s*[·\.\-]?\s*(\d{1,5})\s+anggota",
        r"Group\s*[·\.\-]?\s*(\d{1,5})\s+members"
    ]
    for rg in rgx_list:
        m = re.search(rg, raw_norm, re.IGNORECASE)
        if m:
            # Jika cuma dapat angka tanpa bbox: cari token angka yang sama di tokens_merged
            target = m.group(1)
            for tok in tokens_merged:
                if tok[1] == target:
                    _,_,x,y,w,h = tok
                    scale_x = img_pil.width / pre.width
                    scale_y = img_pil.height / pre.height
                    return (int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y), target)
    return None

def load_fallback_region(img_pil: Image.Image, regions_path="config/regions.json") -> Optional[Tuple[int,int,int,int]]:
    if not os.path.exists(regions_path):
        return None
    with open(regions_path,'r',encoding='utf-8') as f:
        cfg = json.load(f)
    w,h = img_pil.size
    # ambil semua kandidat, pakai yang spesifik dulu, terakhir generic
    candidates = []
    for profile in cfg.get("profiles", []):
        if profile["match_min_width"] <= w <= profile["match_max_width"]:
            r = profile["region"]
            x = int(r["x_pct"] * w)
            y = int(r["y_pct"] * h)
            ww = int(r["w_pct"] * w)
            hh = int(r["h_pct"] * h)
            priority = 2 if "generic" in profile["name"] else 1
            candidates.append((priority,(x,y,ww,hh)))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]
