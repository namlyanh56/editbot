import json, re, os
from typing import Optional, Tuple, List, Dict
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pytesseract
from pytesseract import Output
from .config import TESSERACT_CMD, OCR_LANG

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz·:"

def _preprocess(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = 2 if max(w, h) < 1400 else 1.5
    resized = img.resize((int(w*scale), int(h*scale)), Image.Resampling.BICUBIC)
    gray = ImageOps.grayscale(resized)
    arr = np.array(gray)
    flat = arr.flatten()
    hist, _ = np.histogram(flat, 256, [0,256])
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
    cdf = cdf.astype(np.uint8)
    arr_eq = cdf[flat].reshape(arr.shape)
    return Image.fromarray(arr_eq)

def _merge_digit_tokens(tokens: List[Tuple[int,str,int,int,int,int]]) -> List[Tuple[int,str,int,int,int,int]]:
    merged, buf, baseline_y = [], [], None
    for idx,t,x,y,w,h in tokens:
        if re.fullmatch(r"\d", t) and (baseline_y is None or abs(y - baseline_y) < h*0.6):
            buf.append((idx,t,x,y,w,h)); baseline_y = y; continue
        if buf: merged.append(_collapse(buf)); buf=[]; baseline_y=None
        merged.append((idx,t,x,y,w,h))
    if buf: merged.append(_collapse(buf))
    return merged

def _collapse(buf):
    text = "".join(b[1] for b in buf)
    x0 = min(b[2] for b in buf); y0 = min(b[3] for b in buf)
    x1 = max(b[2]+b[4] for b in buf); hmax = max(b[5] for b in buf)
    return (buf[0][0], text, x0, y0, x1 - x0, hmax)

def _tokens(img_pil: Image.Image):
    pre = _preprocess(img_pil)
    cfg = f"--psm 6 -c tessedit_char_whitelist={WHITELIST}"
    data = pytesseract.image_to_data(pre, output_type=Output.DICT, lang=OCR_LANG, config=cfg)
    toks=[]
    for i,txt in enumerate(data['text']):
        t = txt.strip()
        if not t: continue
        toks.append((i,t,data['left'][i],data['top'][i],data['width'][i],data['height'][i]))
    return pre, toks

def detect_members_line(img_pil: Image.Image) -> Optional[Tuple[int,int,int,int,str]]:
    res = detect_members_line_detailed(img_pil)
    if not res: return None
    x,y,w,h = res["digit_bbox"]; return (x,y,w,h,res["number"])

def detect_members_line_detailed(img_pil: Image.Image) -> Optional[Dict]:
    """
    Return:
      {
        "digit_bbox": (x,y,w,h),
        "number": "56",
        "anchor_bbox": (ax,ay,aw,ah)  # bbox kata 'anggota'/'members'
      }
    """
    pre, toks = _tokens(img_pil)
    if not toks: return None
    merged = _merge_digit_tokens(toks)
    anchors = [t for t in merged if t[1].lower() in ("anggota","members")]
    candidates=[]
    for a in anchors:
        a_y, a_h = a[3], a[5]
        idxa = merged.index(a)
        for prev in reversed(merged[:idxa]):
            if re.fullmatch(r"\d{1,5}", prev[1]) and abs(prev[3]-a_y) < max(prev[5], a_h)*0.7:
                candidates.append((prev, a)); break
    if not candidates:
        line_text = " ".join(t[1] for t in merged).replace("•","·")
        m = re.search(r"(?:Grup|Group)\s*[·\.\-]?\s*(\d{1,5})\s+(?:anggota|members)", line_text, re.IGNORECASE)
        if m:
            target = m.group(1)
            a = next((t for t in merged if t[1].lower() in ("anggota","members")), None)
            p = next((t for t in merged if t[1]==target), None)
            if p and a: candidates.append((p,a))
    if not candidates: return None

    (p,a) = candidates[0]
    sx = img_pil.width / pre.width; sy = img_pil.height / pre.height
    _, num, x,y,w,h = p
    _, _, ax,ay,aw,ah = a
    return {
        "digit_bbox": (int(x*sx), int(y*sy), int(w*sx), int(h*sy)),
        "number": num,
        "anchor_bbox": (int(ax*sx), int(ay*sy), int(aw*sx), int(ah*sy))
    }

def detect_title_trailing_number(img_pil: Image.Image) -> Optional[Tuple[int,int,int,int,str]]:
    pre, toks = _tokens(img_pil)
    if not toks: return None
    lines=[]
    for tok in toks:
        _,text,x,y,w,h=tok
        placed=False
        for L in lines:
            if abs(L['y']-y)<h*0.6: L['tokens'].append(tok); placed=True; break
        if not placed: lines.append({'y':y,'tokens':[tok]})
    lines.sort(key=lambda L:L['y'])
    for ln in lines[:5]:
        ln['tokens'].sort(key=lambda t:t[2])
        joined=" ".join(t[1] for t in ln['tokens'])
        m=re.search(r"[A-Za-zÀ-ÿ0-9]+?\s+(\d{1,5})$", joined)
        if m:
            num=m.group(1)
            for t in reversed(ln['tokens']):
                if t[1]==num:
                    _,_,x,y,w,h=t
                    sx=img_pil.width/pre.width; sy=img_pil.height/pre.height
                    return (int(x*sx),int(y*sy),int(w*sx),int(h*sy),num)
    return None

def ocr_find_member_number(img_pil: Image.Image, debug=False):
    return detect_members_line(img_pil)

def load_fallback_region(img_pil: Image.Image, regions_path="config/regions.json"):
    if not os.path.exists(regions_path):
        return None
    with open(regions_path,'r',encoding='utf-8') as f:
        cfg = json.load(f)
    w,h = img_pil.size
    cands=[]
    for p in cfg.get("profiles", []):
        if p["match_min_width"]<=w<=p["match_max_width"]:
            r=p["region"]
            x=int(r["x_pct"]*w); y=int(r["y_pct"]*h)
            ww=int(r["w_pct"]*w); hh=int(r["h_pct"]*h)
            pr=2 if "generic" in p["name"] else 1
            cands.append((pr,(x,y,ww,hh)))
    if not cands: return None
    cands.sort(key=lambda z:z[0])
    return cands[0][1]
