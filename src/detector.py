import json, re, os
from typing import Optional, Tuple
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
from .config import TESSERACT_CMD, OCR_LANG

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz·:"

def _raw_ocr_text(img_pil: Image.Image) -> str:
    cfg = f"--psm 6 -c tessedit_char_whitelist={WHITELIST}"
    return pytesseract.image_to_string(img_pil, lang=OCR_LANG, config=cfg)

def ocr_find_member_number(img_pil: Image.Image, debug=False) -> Optional[Tuple[int,int,int,int,str]]:
    """
    Pola yang dicari:
      'Grup · 53 anggota'
      'Grup 53 anggota'
      'Group · 120 members'
    Langkah:
      1. Coba regex di raw text.
      2. Kalau dapat angka, cocokkan ke token image_to_data untuk bounding box-nya.
      3. Jika gagal, fallback ke token scanning: token 'anggota/members' -> mundur angka.
    """
    raw = _raw_ocr_text(img_pil)
    # Normalisasi bullet
    raw_norm = raw.replace("•", "·")
    regexes = [
        r"Grup\s*[·\.\-]?\s*(\d{1,5})\s+anggota",
        r"Group\s*[·\.\-]?\s*(\d{1,5})\s+members"
    ]
    candidate_number = None
    for rg in regexes:
        m = re.search(rg, raw_norm, re.IGNORECASE)
        if m:
            candidate_number = m.group(1)
            break

    data = pytesseract.image_to_data(img_pil, output_type=Output.DICT,
                                     lang=OCR_LANG,
                                     config=f"--psm 6 -c tessedit_char_whitelist={WHITELIST}")
    n = len(data['text'])

    def tokens():
        for i in range(n):
            yield i, data['text'][i], data['left'][i], data['top'][i], data['width'][i], data['height'][i]

    # Jika kita sudah tahu angkanya, cari token yang cocok
    if candidate_number:
        matches = []
        for i,t,x,y,w,h in tokens():
            if t.strip() == candidate_number:
                matches.append((x,y,w,h,t))
        # pilih yang ketinggiannya paling kecil (biasanya teks baris deskripsi)
        if matches:
            best = sorted(matches, key=lambda r: r[3])[0]
            if debug:
                _save_debug(img_pil, [best], "debug_ocr.png")
            return best

    # Fallback lama: cari 'anggota/members' lalu token sebelumnya berupa angka
    idx_targets = [i for i in range(n) if data['text'][i].lower() in ("anggota","members")]
    for idx in idx_targets:
        for j in range(idx-1, max(idx-6, -1), -1):
            token = data['text'][j].strip()
            if re.fullmatch(r"\d{1,5}", token):
                x = data['left'][j]; y = data['top'][j]; w = data['width'][j]; h = data['height'][j]
                if debug: _save_debug(img_pil, [(x,y,w,h,token)], "debug_ocr.png")
                return (x,y,w,h,token)
    return None

def _save_debug(img_pil, boxes, fname):
    dbg = img_pil.copy()
    dr = ImageDraw.Draw(dbg)
    for (x,y,w,h,t) in boxes:
        dr.rectangle([x,y,x+w,y+h], outline="red", width=2)
        dr.text((x,y-12), t, fill="red")
    dbg.save(fname)

def load_fallback_region(img_pil: Image.Image, regions_path="config/regions.json") -> Optional[Tuple[int,int,int,int]]:
    if not os.path.exists(regions_path):
        return None
    with open(regions_path,'r',encoding='utf-8') as f:
        cfg = json.load(f)
    w,h = img_pil.size
    # heuristik mode: bedakan light/dark via mean luminance
    from statistics import mean
    import numpy as np
    arr = np.array(img_pil.convert("L"))
    lum = arr.mean()
    for profile in cfg.get("profiles", []):
        if profile["match_min_width"] <= w <= profile["match_max_width"]:
            r = profile["region"]
            x = int(r["x_pct"] * w)
            y = int(r["y_pct"] * h)
            ww = int(r["w_pct"] * w)
            hh = int(r["h_pct"] * h)
            return (x,y,ww,hh)
    return None
