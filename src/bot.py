import logging
from io import BytesIO
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from .config import TELEGRAM_BOT_TOKEN
from .detector import ocr_find_member_number, load_fallback_region
from .image_edit import patch_and_replace_number

FONT_PATH = "fonts/Roboto-Regular.ttf"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("member_bot")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Kirim foto + caption angka baru (contoh: 123).")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    caption = (msg.caption or "").strip()

    if not caption.isdigit():
        await msg.reply_text("Caption harus angka saja. Contoh: 123")
        return

    new_number = caption
    photo = msg.photo[-1]
    file = await photo.get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)
    img = Image.open(bio).convert("RGB")

    det = ocr_find_member_number(img)
    if det:
        x,y,w,h, old = det
        method = "ocr"
    else:
        fb = load_fallback_region(img)
        if not fb:
            await msg.reply_text("Gagal mendeteksi angka anggota (OCR & fallback). Pastikan format standar.")
            return
        x,y,w,h = fb
        old = "?"
        method = "fallback"

    edited = patch_and_replace_number(img, (x,y,w,h), old, new_number, FONT_PATH)
    out_bio = BytesIO()
    edited.save(out_bio, format="PNG")
    out_bio.seek(0)

    await msg.reply_photo(out_bio, caption=f"Done ({method}). {old} -> {new_number}")

def run():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum di-set.")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    run()
