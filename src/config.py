import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # optional explicit
OCR_LANG = os.getenv("OCR_LANG", "eng")  # bisa tambah "ind"
