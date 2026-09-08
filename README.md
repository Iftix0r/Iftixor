# Iftixor

Iftixor — noldan (0 dan) o'zimiz yozgan va o'qitgan, GPT-uslubidagi kichik
transformer til modeli. Loyihaning maqsadi — tayyor API (OpenAI, Gemini,
Claude) chaqirmasdan, haqiqiy neyron tarmoqni o'zimiz qanday ishlashini
tushunib, noldan o'qitish va uni Telegram bot orqali sinash.

## Muhim eslatma

GPT/Gemini/Claude darajasidagi model milliardlab so'zlik ma'lumot va yuzlab
GPU bilan oylab o'qitiladi. Bu loyihadagi model ancha kichik (CPU'da ham
o'qitish mumkin bo'lishi uchun) va **ta'lim/tajriba maqsadida** yaratilgan.
U dastlab juda oddiy javoblar beradi. Ma'lumot va o'qitish vaqtini
oshirgan sari sifati asta-sekin yaxshilanadi.

## Loyihaning tuzilishi

```
iftixor/
  config.py     — model giperparametrlari (GPTConfig)
  tokenizer.py  — belgi darajasidagi (char-level) tokenizator
  model.py      — noldan yozilgan GPT arxitekturasi (attention, MLP, va h.k.)
  dataset.py    — o'qitish uchun batch tayyorlash
  train.py      — modelni o'qitish skripti
  generate.py   — o'qitilgan model bilan matn generatsiya qilish
  chat_cli.py   — terminalda tez sinov uchun suhbat rejimi
  bot.py        — Telegram bot handlerlari
data/
  corpus.txt    — namuna o'zbek matn korpusi (siz buni kengaytirasiz)
checkpoints/    — o'qitilgan model shu yerga saqlanadi (.gitignore'da)
deploy/
  iftixor.service — serverga systemd orqali deploy qilish uchun namuna
main.py         — Telegram botni ishga tushiruvchi asosiy fayl
```

## 1. O'rnatish

```bash
python3 -m venv .venv
source .venv/bin/activate

# MUHIM: torch'ni avval CPU-only versiyasi bilan o'rnating.
# Oddiy "pip install torch" PyPI'dan CUDA to'plamini (bir necha GB) ham
# tortib oladi va kichik VPS'larda diskni to'ldirib qo'yadi.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## 2. Ma'lumot (dataset) tayyorlash

`data/corpus.txt` faylida kichik namuna matn bor. Bu faqat pipeline
ishlashini tekshirish uchun. Modelni real yaxshilash uchun shu faylga
o'zingizning matnlaringizni (kitoblar, maqolalar, Telegram suhbatlaringiz,
va h.k.) qo'shib, faylni kengaytiring. Suhbat uslubini o'rgatish uchun
quyidagi formatga amal qiling:

```
Foydalanuvchi: <savol>
Iftixor: <javob>
```

## 3. Modelni noldan o'qitish

```bash
python -m iftixor.train --data data/corpus.txt --out checkpoints/iftixor.pt --steps 3000
```

Foydali parametrlar:

- `--steps` — o'qitish qadamlari soni (ko'proq dataset uchun ko'proq qadam kerak)
- `--batch-size`, `--block-size` — CPU xotirasiga qarab kamaytirish/oshirish mumkin
- `--n-layer`, `--n-head`, `--n-embd` — model hajmi (kattalashtirsangiz CPU'da sekinlashadi)

O'qitish tugagach, model `checkpoints/iftixor.pt` fayliga saqlanadi.
Checkpoint har `--eval-interval` qadamda ham saqlanib boriladi (nafaqat
oxirida) — shuning uchun server o'chib qolsa yoki jarayon uzilib qolsa,
hech narsa yo'qolmaydi.

Agar o'qitish biror sababdan (masalan, xotira yetishmasligi) uzilib qolsa,
uni oxirgi saqlangan joydan davom ettirish mumkin:

```bash
python -m iftixor.train --out checkpoints/iftixor.pt --steps 3000 --resume
```

## 4. Terminalda tez sinash

Telegram'ga ulashdan oldin, model qanday javob berayotganini terminalda
tekshiring:

```bash
python -m iftixor.chat_cli --checkpoint checkpoints/iftixor.pt
```

## 5. Telegram bot sozlash

1. Telegram'da [@BotFather](https://t.me/BotFather) orqali yangi bot yarating
   va tokenni oling.
2. `.env.example` faylidan nusxa oling:

   ```bash
   cp .env.example .env
   ```

3. `.env` faylini tahrirlab, o'z tokeningizni kiriting:

   ```
   TELEGRAM_BOT_TOKEN=xxxxxx:yyyyyyyyyyyyyyyyyyyyyyyyyyyy
   MODEL_CHECKPOINT_PATH=checkpoints/iftixor.pt
   ```

4. Botni ishga tushiring:

   ```bash
   python main.py
   ```

5. Telegram'da botingizga `/start` yozing va suhbatni boshlang.
   `/reset` — suhbat tarixini tozalaydi.

### O'qitish progressini Telegram orqali kuzatish (ixtiyoriy)

`python -m iftixor.train` ishga tushirilganda, agar `.env` faylida
`TELEGRAM_BOT_TOKEN` va `ADMIN_CHAT_ID` ikkalasi ham to'ldirilgan bo'lsa,
har `--eval-interval` qadamda (standart: 200) sizga qancha qadam bajarilgani,
train/val loss va taxminiy qolgan vaqt haqida Telegram xabari keladi.

`ADMIN_CHAT_ID` olish uchun: botingizga istalgan xabar yuboring, so'ng
brauzerda `https://api.telegram.org/bot<TOKEN>/getUpdates` manzilini oching
va javobdan `"chat":{"id": ...}` qiymatini `.env`ga yozing. Bu bot hali
o'qitilmagan (checkpoint yo'q) bo'lsa ham ishlaydi — chat_id olish uchun
botning o'zi ishlab turishi shart emas.

## 6. GitHub'ga yuklash

```bash
git add -A
git commit -m "Iftixor: noldan o'qitiladigan til modeli va Telegram bot"
git push origin main
```

`.env` va `checkpoints/*.pt` fayllari `.gitignore` orqali repo'ga
tushmaydi — token va og'ir model fayllarini GitHub'ga yubormang.

## 7. Serverga deploy qilish (systemd bilan)

```bash
# Serverda:
git clone git@github.com:Iftix0r/Iftixor.git /opt/iftixor
cd /opt/iftixor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env   # tokenni kiriting

python -m iftixor.train --steps 3000   # yoki tayyor checkpointni serverga yuklang
```

`deploy/iftixor.service` faylini namuna sifatida ishlating:

```bash
sudo cp deploy/iftixor.service /etc/systemd/system/iftixor.service
sudo systemctl daemon-reload
sudo systemctl enable --now iftixor
sudo journalctl -u iftixor -f   # loglarni kuzatish
```

## 8. Modelni keyinchalik yaxshilash yo'llari

- `data/corpus.txt` hajmini sezilarli kattalashtiring (minglab-o'n minglab qator).
- Ko'proq `--steps` bilan uzoqroq o'qiting.
- Agar keyinchalik GPU topsangiz, `--n-layer`, `--n-head`, `--n-embd` va
  `--block-size` qiymatlarini oshirib, kattaroq model o'qiting.
- Belgi darajasidan (char-level) so'z bo'laklariga (subword/BPE tokenizatsiya)
  o'tish sifatni sezilarli oshiradi.
- Instruction tuning (savol-javob juftliklari bilan maxsus o'qitish) modelni
  suhbat uslubiga yaqinlashtiradi.
