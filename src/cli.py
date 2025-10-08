
import argparse
from PIL import Image
from .detector import ocr_find_member_number, load_fallback_region
from .image_edit import patch_and_replace_number
import os, json, sys

FONT_PATH_DEFAULT = "fonts/Roboto-Regular.ttf"

def main():
    parser = argparse.ArgumentParser(description="Ganti angka jumlah anggota.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--number", required=True, help="Angka baru")
    parser.add_argument("--output", required=True)
    parser.add_argument("--font", default=FONT_PATH_DEFAULT)
    args = parser.parse_args()

    if not args.number.isdigit():
        print("Caption / arg --number harus berupa angka.")
        sys.exit(1)

    img = Image.open(args.input).convert("RGB")

    det = ocr_find_member_number(img)
    if det:
        x,y,w,h, old = det
        print(f"OCR sukses: old={old} bbox=({x},{y},{w},{h})")
    else:
        fb = load_fallback_region(img)
        if not fb:
            print("Gagal menemukan angka (OCR & fallback).")
            sys.exit(2)
        x,y,w,h = fb
        old = "?"
        print(f"Fallback region digunakan: bbox=({x},{y},{w},{h})")

    out = patch_and_replace_number(img, (x,y,w,h), old, args.number, args.font)
    out.save(args.output)
    meta = {
        "old_number": old,
        "new_number": args.number,
        "used": "ocr" if det else "fallback",
        "bbox": [x,y,w,h]
    }
    with open(args.output + ".json","w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)
    print("Selesai:", args.output)

if __name__ == "__main__":
    main()
