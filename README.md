# Member Count Image Editor Bot

Fokus: Mengubah hanya angka jumlah anggota pada screenshot (WhatsApp/Telegram group info) secara mulus tanpa mengubah elemen lain.

## Fitur
- Deteksi angka lama via OCR (pytesseract).
- Fallback koordinat statis jika OCR gagal.
- Inpainting / patch adaptif.
- Rendering angka baru high-DPI (supersampling 3×).
- Sampling warna teks agar menyatu dengan tema (light/dark).
- CLI & Telegram bot.

## Arsitektur
Lihat penjelasan di `src/`:
- `detector.py`: OCR + fallback.
- `image_edit.py`: patch, inpaint, render angka baru.
- `utils.py`: fungsi sampling warna, penghitungan font size adaptif.
- `config/regions.json`: fallback region (persentase).
- `bot.py`: Telegram bot (caption harus hanya angka).
- `cli.py`: eksekusi offline.

## Dependensi
Lihat `requirements.txt`: Pillow, pytesseract, opencv-python, python-telegram-bot, numpy, python-dotenv.

Pastikan Tesseract terinstal di OS:
- Debian/Ubuntu: `sudo apt install tesseract-ocr`
- Mac (brew): `brew install tesseract`

## Penggunaan CLI
```
python src/cli.py --input input.png --number 150 --output edited.png
```

## Penggunaan Telegram
Kirim foto + caption angka (misal `123`).

## Catatan Kualitas
- Jika OCR salah, periksa resolusi atau tingkatkan kontras.
- Tambahkan font berbeda bila screenshot berasal dari iOS (SF Pro). Fallback cukup Roboto.

## Lisensi
Gunakan bebas; font lisensi mengikuti masing-masing file font.
