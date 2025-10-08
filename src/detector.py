import json
from typing import Optional, Tuple
import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np
import re
from .config import TESSERACT_CMD, OCR_LANG

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def ocr_find_member_number(img_pil: Image.Image) -> Optional[Tuple[int,int,int,int,str]]:
    """
    Cari angka anggota melalui OCR.
    Return: (x,y,w,h,number) atau None
    Heuristik:
    - cari token 'anggota' / 'members'
    - token sebelumnya yang hanya digit dianggap jumlah anggota
    """
    custom_config = "--psm 6"
    data = pytesseract.image_to_data(img_pil, output_type=Output.DICT, lang=OCR_LANG, config=custom_config)
    n = len(data['text'])
    idx_targets = [i for i,t in enumerate(data['text']) if t.lower() in ("anggota","members")]
    for idx in idx_targets:
        # mundur cari angka
        for j in range(idx-1, max(idx-6, -1), -1):
            token = data['text'][j].strip()
            if re.fullmatch(r"\d{1,5}", token):
                x = data['left'][j]
                y = data['top'][j]
                w = data['width'][j]
                h = data['height'][j]
                return (x,y,w,h,token)
    return None

def load_fallback_region(img_pil: Image.Image, regions_path="config/regions.json") -> Optional[Tuple[int,int,int,int]]:
    import os
    if not os.path.exists(regions_path):
        return None
    with open(regions_path,'r',encoding='utf-8') as f:
        cfg = json.load(f)
    w,h = img_pil.size
    for profile in cfg.get("profiles", []):
        if profile["match_min_width"] <= w <= profile["match_max_width"]:
            r = profile["region"]
            x = int(r["x_pct"] * w)
            y = int(r["y_pct"] * h)
            ww = int(r["w_pct"] * w)
            hh = int(r["h_pct"] * h)
            return (x,y,ww,hh)
    return None
