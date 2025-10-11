import logging, os
from io import BytesIO
from pathlib import Path
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from .config import TELEGRAM_BOT_TOKEN
from .detector import (
    detect_members_line_detailed,
    detect_title_trailing_number,
    load_fallback_region
)
from .android_detector import detect_android_members_line
from .refine import refine_region_digits
from .image_edit import patch_and_replace_number, replace_members_line_full
from .utils import get_safe_font_path

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_PATH = str(BASE_DIR / "fonts" / "Roboto-Regular.ttf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("member_bot")

HELP_TXT = (
"Format caption:\n"
"  123       -> ubah angka anggota (Android light/dark didukung)\n"
"  123#t     -> ubah angka di judul (misal 'Freelance 62')\n"
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

    try:
        font_ok = get_safe_font_path(FONT_PATH)
    except Exception as e:
        await msg.reply_text(f"Font tidak valid: {e}")
        return

    if target == "members":
        # 1) Coba detektor Android (color+OCR kecil)
        res = detect_android_members_line(img)
        if res:
            edited = replace_members_line_full(
                img,
                res["line_bbox"],
                res["digit_bbox"],
                res["anchor_bbox"],
                number,
                font_ok
            )
            out = BytesIO(); edited.save(out, format="PNG"); out.seek(0)
            await msg.reply_photo(out, caption=f"Selesai (android detector) {res['number']} -> {number}")
            return

        # 2) Fallback: OCR detailed generic
        gen = detect_members_line_detailed(img)
        if gen:
            edited = replace_members_line_full(
                img,
                # buat line bbox dari union digit+anchor dengan margin
                (
                    max(0, min(gen['digit_bbox'][0], gen['anchor_bbox'][0]) - 20),
                    max(0, min(gen['digit_bbox'][1], gen['anchor_bbox'][1]) - 8),
                    min(img.width, max(gen['digit_bbox'][0]+gen['digit_bbox'][2], gen['anchor_bbox'][0]+gen['anchor_bbox'][2]) + 20) - max(0, min(gen['digit_bbox'][0], gen['anchor_bbox'][0]) - 20),
                    min(img.height, max(gen['digit_bbox'][1]+gen['digit_bbox'][3], gen['anchor_bbox'][1]+gen['anchor_bbox'][3]) + 8) - max(0, min(gen['digit_bbox'][1], gen['anchor_bbox'][1]) - 8)
                ),
                gen["digit_bbox"],
                gen["anchor_bbox"],
                number,
                font_ok
            )
            out = BytesIO(); edited.save(out, format="PNG"); out.seek(0)
            await msg.reply_photo(out, caption=f"Selesai (generic OCR) {gen['number']} -> {number}")
            return

        # 3) Fallback terakhir: refine region statis
        fb = load_fallback_region(img)
        if fb:
            rb = refine_region_digits(img, fb)
            if rb:
                img2 = patch_and_replace_number(img, rb, "?", number, font_ok)
                out = BytesIO(); img2.save(out, format="PNG"); out.seek(0)
                await msg.reply_photo(out, caption=f"Selesai (refine fallback) ? -> {number}")
                return

        await msg.reply_text("Tidak bisa menemukan digit (Android). Kirim sebagai File atau coba #t.")
        return

    else:
        # target judul
        dtitle = detect_title_trailing_number(img)
        if not dtitle:
            await msg.reply_text("Tidak bisa menemukan angka di judul. Kirim sebagai File.")
            return
        x,y,w,h,old = dtitle
        img2 = patch_and_replace_number(img, (x,y,w,h), old or "?", number, font_ok)
        out = BytesIO(); img2.save(out, format="PNG"); out.seek(0)
        await msg.reply_photo(out, caption=f"Selesai (title) {old} -> {number}")

def run():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum di-set.")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    run()
