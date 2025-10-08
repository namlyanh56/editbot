import logging, os
from io import BytesIO
from pathlib import Path
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from .config import TELEGRAM_BOT_TOKEN
from .detector import (
    detect_members_line,
    detect_title_trailing_number,
    load_fallback_region
)
from .refine import refine_region_digits
from .image_edit import patch_and_replace_number
from .utils import get_safe_font_path

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_PATH = str(BASE_DIR / "fonts" / "Roboto-Regular.ttf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("member_bot")

user_modes = {}  # masih bisa dipakai kalau mau, tapi optional di sini

HELP_TXT = (
"Format caption:\n"
"  123       -> ubah angka anggota (baris 'Grup · ... anggota')\n"
"  123#m     -> paksa anggota\n"
"  123#t     -> ubah angka trailing judul (misal 'Freelance 62')\n"
"Tambahan: kirim sebagai File (bukan photo compressed) untuk OCR lebih baik."
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TXT)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    cap = (msg.caption or "").strip().lower()

    target = "members"
    if "#t" in cap:
        target = "title"
    elif "#m" in cap:
        target = "members"
    number = cap.split("#")[0].strip()

    if not number.isdigit():
        await msg.reply_text("Caption harus angka (opsional #m atau #t). Contoh: 123#t")
        return

    # Ambil file
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

    det = None
    old = "?"
    if target == "members":
        det = detect_members_line(img)
    else:
        det = detect_title_trailing_number(img)

    method = f"ocr_{target}"
    if not det:
        # fallback region + refine
        fb = load_fallback_region(img)
        if fb:
            refined = refine_region_digits(img, fb)
            if refined:
                det = (*refined, old)
                method = "refine_fallback"
    if not det:
        await msg.reply_text("Tidak bisa menemukan digit target (OCR & refine gagal). Coba kirim sebagai file atau gunakan target lain (#t / #m).")
        return

    x,y,w,h, old_candidate = det
    if old_candidate and old_candidate != "?":
        old = old_candidate

    try:
        font_ok = get_safe_font_path(FONT_PATH)
        edited = patch_and_replace_number(img, (x,y,w,h), old, number, font_ok)
    except Exception as e:
        logger.exception("Patch error")
        await msg.reply_text(f"Error patch: {e}")
        return

    out = BytesIO()
    edited.save(out, format="PNG")
    out.seek(0)
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
