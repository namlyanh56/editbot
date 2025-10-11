import logging, os
from io import BytesIO
from pathlib import Path
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from .config import TELEGRAM_BOT_TOKEN
from .detector import (
    detect_members_line,
    detect_members_line_detailed,
    detect_title_trailing_number,
    load_fallback_region
)
from .refine import refine_region_digits
from .image_edit import patch_and_replace_number, replace_members_line_full
from .utils import get_safe_font_path

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_PATH = str(BASE_DIR / "fonts" / "Roboto-Regular.ttf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("member_bot")

HELP_TXT = (
"Format caption:\n"
"  123       -> ubah angka anggota (baris 'Grup · ... anggota')\n"
"  123#t     -> ubah angka trailing judul (misal 'Freelance 62')\n"
"Tips: kirim sebagai File (bukan Photo) supaya tidak terkompres."
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TXT)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    cap = (msg.caption or "").strip().lower()
    target = "members" if "#t" not in cap else "title"
    number = cap.split("#")[0].strip()
    if not number.isdigit():
        await msg.reply_text("Caption harus angka (opsional #t). Contoh: 123#t")
        return

    photo = msg.photo[-1]
    file = await photo.get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)
    try:
        img = Image.open(bio).convert("RGB")
    except Exception as e:
        await msg.reply_text(f"Gagal baca gambar: {e}")
        return

    det = None; method = ""
    if target == "members":
        res = detect_members_line_detailed(img)
        if res:
            # Bangun line_bbox dari union digit+anchor dengan margin
            dbx,dby,dbw,dbh = res["digit_bbox"]
            ax,ay,aw,ah = res["anchor_bbox"]
            x0 = max(0, min(dbx, ax) - 20)
            y0 = max(0, min(dby, ay) - 8)
            x1 = min(img.width, max(dbx+dbw, ax+aw) + 20)
            y1 = min(img.height, max(dby+dbh, ay+ah) + 8)
            line_bbox = (x0,y0,x1-x0,y1-y0)
            try:
                font_ok = get_safe_font_path(FONT_PATH)
                edited = replace_members_line_full(
                    img,
                    line_bbox,
                    res["digit_bbox"],
                    res["anchor_bbox"],
                    number,
                    font_ok
                )
                out = BytesIO(); edited.save(out, format="PNG"); out.seek(0)
                await msg.reply_photo(out, caption=f"Selesai (retypeset members) {res['number']} -> {number}")
                return
            except Exception as e:
                logger.warning(f"Retypeset gagal, fallback patch angka. Err: {e}")
                det = (*res["digit_bbox"], res["number"]); method = "ocr_members_patch"
        else:
            # fallback refine
            fb = load_fallback_region(img)
            if fb:
                rb = refine_region_digits(img, fb)
                if rb: det = (*rb, "?"); method = "refine_fallback"
    else:
        dtitle = detect_title_trailing_number(img)
        if dtitle:
            det = dtitle; method = "ocr_title"
        else:
            fb = load_fallback_region(img)
            if fb:
                rb = refine_region_digits(img, fb)
                if rb: det = (*rb, "?"); method = "refine_fallback"

    if not det:
        await msg.reply_text("Tidak bisa menemukan digit target. Kirim sebagai File, atau coba target lain (#t).")
        return

    x,y,w,h, old_candidate = det
    old = old_candidate if old_candidate and old_candidate != "?" else "?"
    try:
        font_ok = get_safe_font_path(FONT_PATH)
        edited = patch_and_replace_number(img, (x,y,w,h), old, number, font_ok)
    except Exception as e:
        logger.exception("Patch error")
        await msg.reply_text(f"Error patch: {e}")
        return

    out = BytesIO(); edited.save(out, format="PNG"); out.seek(0)
    await msg.reply_photo(out, caption=f"Selesai ({method}) {old} -> {number} bbox=({x},{y},{w},{h})")

def run():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum di-set.")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    run()
