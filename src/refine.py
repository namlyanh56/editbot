import cv2 as cv
import numpy as np
from typing import Tuple, Optional
from PIL import Image

def refine_region_digits(img_pil: Image.Image, region_bbox: Tuple[int,int,int,int]) -> Optional[Tuple[int,int,int,int]]:
    """
    Diberi region besar (x,y,w,h), coba temukan cluster digit di dalamnya.
    Return bbox digit atau None
    """
    x,y,w,h = region_bbox
    crop = img_pil.crop((x,y,x+w,y+h))
    gray = crop.convert("L")
    arr = np.array(gray)
    # Normalisasi ringan
    arr = cv.equalizeHist(arr)
    th = cv.adaptiveThreshold(arr,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY_INV,31,9)
    contours,_ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cand=[]
    H = arr.shape[0]
    for c in contours:
        bx,by,bw,bh = cv.boundingRect(c)
        if bh < H*0.25 or bh > H*0.95: continue
        ratio = bw / (bh+1e-6)
        if 0.25 <= ratio <= 1.2:
            cand.append((bx,by,bw,bh))
    if not cand:
        return None
    cand = sorted(cand, key=lambda b: b[0])
    # gabung neighbor untuk multi-digit
    merged=[]
    cur=list(cand[0])
    for b in cand[1:]:
        if b[0] <= cur[0]+cur[2]+(cur[3]*0.8):
            x0=min(cur[0],b[0]); y0=min(cur[1],b[1])
            x1=max(cur[0]+cur[2], b[0]+b[2])
            y1=max(cur[1]+cur[3], b[1]+b[3])
            cur=[x0,y0,x1-x0,y1-y0]
        else:
            merged.append(tuple(cur)); cur=list(b)
    merged.append(tuple(cur))
    # pilih cluster paling tinggi (digit judul biasanya besar) atau paling tengah
    merged.sort(key=lambda m: (-m[3], abs((m[0]+m[2]/2)-w/2)))
    bx,by,bw,bh=merged[0]
    return (x+bx, y+by, bw, bh)
