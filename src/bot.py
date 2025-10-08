import logging, os, json
from io import BytesIO
from pathlib import Path
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from .config import TELEGRAM_BOT_TOKEN
from .detector import ocr_find_member_number, load_fallback_region
from .image_edit import patch_and_replace_number
from .utils import get_safe_font_path
from .mode_detector import detect_number_with_mode, MODES_CFG

BASE_DIR = Path(__file__).resolve().parent.parent
PREFERRED_FONT = BASE_DIR / "fonts" / "Roboto-Regular.ttf"
FONT_PATH = str(PREFERRED_FONT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("member_bot")

# state sederhana (RAM)
user_modes = {}
debug_flags = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Kirim FOTO screenshot + caption ANGKA baru.\n"
        "Gunakan /modes untuk daftar mode & /setmode <nama_mode>.\n"
        "Contoh: /setmode android_dark"
    )

async def modes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["Daftar mode:"]
    for k,v in MODES_CFG.items():
        lines.append(f"- {k}: {v.get('note','')}")
    cur = user_modes.get(update.effective_chat.id, "(belum diset)")
    lines.append(f"Mode aktif chat ini: {cur}")
    await update.message.reply_text("\n".join(lines))

async def setmode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Gunakan: /setmode <nama_mode>")
        return
    mode = context.args[0].strip()
    if mode not in MODES_CFG:
        await update.message.reply_text("Mode tidak dikenal. /modes untuk melihat list.")
        return
    user_modes[update.effective_chat.id] = mode
    await update.message.reply_text(f"Mode diset ke: {mode}")

async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    flag = debug_flags.get(chat_id, False)
    debug_flags[chat_id] = not flag
    await update.message.reply_text(f"Debug sekarang: {debug_flags[chat_id]}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    caption = (msg.caption or "").strip()
    if not caption.isdigit():
        await msg.reply_text("Caption harus angka saja. Contoh: 123")
        return
    new_number = caption

    # Ambil file foto
    photo = msg.photo[-1]
    file = await photo.get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)
    try:
        img = Image.open(bio)
        img = img.convert("RGB")
    except Exception as e:
        await msg.reply_text(f"Gagal buka gambar: {e}")
        return

    chat_id = update.effective_chat.id
    active_mode = user_modes.get(chat_id)

    det_info = None
    if active_mode:
        try:
            det_info = detect_number_with_mode(img, active_mode)
        except Exception as e:
            logger.warning(f"Mode detection error: {e}")

    if det_info:
        x,y,w,h = det_info["bbox_global"]
        old = det_info["old_number"]
        method = det_info["method"]
    else:
        # fallback lama
        det = ocr_find_member_number(img, debug=debug_flags.get(chat_id, False))
        if det:
            x,y,w,h, old = det
            method = "ocr_global"
        else:
            fb = load_fallback_region(img)
            if not fb:
                await msg.reply_text("Gagal deteksi (mode+OCR+fallback). Set mode dengan /modes lalu /setmode dulu.")
                return
            x,y,w,h = fb
            old = "?"
            method = "region_fallback"

    try:
        font_valid = get_safe_font_path(FONT_PATH)
        edited = patch_and_replace_number(img, (x,y,w,h), old, new_number, font_valid)
    except Exception as e:
        logger.exception("Gagal patch.")
        await msg.reply_text(f"Error saat patch: {e}")
        return

    out_bio = BytesIO()
    edited.save(out_bio, format="PNG")
    out_bio.seek(0)
    await msg.reply_photo(out_bio, caption=f"Selesai ({method}). {old} -> {new_number} | bbox=({x},{y},{w},{h})")

def run():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN belum di-set.")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("modes", modes))
    app.add_handler(CommandHandler("setmode", setmode))
    app.add_handler(CommandHandler("debug", debug))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    run()
