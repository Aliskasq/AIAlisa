# План реализации: Секторы монет в AIAlisa

## Новые файлы

| Файл | Назначение |
|---|---|
| `core/categories.py` | Вся логика секторов: загрузка, сохранение, поиск, маппинг |
| `data/scan_settings.json` | Настройки какие секторы сканировать |

`data/coin_categories.json` — **уже существует** (648 монет, 17 секторов).

## Структура данных

**`data/coin_categories.json`** (уже готов):
```json
{
  "_updated": "2026-07-04T...",
  "_sectors": ["🤖 AI", "🏦 DeFi", "🏗 L1", "🔗 L2", "💎 Meme", "🎮 Gaming", "🖼 NFT", "💰 RWA", "🔮 Oracle", "💾 Infra", "📊 Stocks", "🥇 Metals/Commodities", "👑 Blue Chip", "📦 Other", "⚽ Fan Token", "💳 Payments", "📈 ETFs"],
  "BTCUSDT": ["👑 Blue Chip"],
  "SIRENUSDT": ["🤖 AI", "💎 Meme"],
  "SONYUSDT": ["📊 Stocks"]
}
```

**`data/scan_settings.json`:**
```json
{
  "enabled_sectors": ["🤖 AI", "💎 Meme", "🏗 L1", "🔗 L2", "🏦 DeFi", "🎮 Gaming", "👑 Blue Chip", "🖼 NFT", "💰 RWA", "🔮 Oracle", "💾 Infra", "📦 Other", "⚽ Fan Token", "💳 Payments"],
  "disabled_sectors": ["📊 Stocks", "🥇 Metals/Commodities", "📈 ETFs"],
  "scan_unknown": true
}
```

`scan_unknown: true` — сканировать монеты без сектора (чтобы не пропустить новые листинги).

---

## Фильтрация сканера по секторам

Когда бот получает список symbols из `get_usdt_futures_symbols()`, перед анализом каждая монета проходит через `should_scan(symbol)`:

1. Монета есть в `coin_categories.json` → проверяем, есть ли хоть один её сектор в `enabled_sectors`
   - Да → сканируем
   - Нет (все секторы в `disabled_sectors`) → **пропускаем** (не грузим свечи, не рисуем линии)
2. Монеты нет в `coin_categories.json` → смотрим `scan_unknown`
   - `true` → сканируем + шлём уведомление в личный чат (см. ниже)
   - `false` → пропускаем

**Результат:** отключив 📊 Stocks, 🥇 Metals/Commodities, 📈 ETFs — бот не будет тратить API запросы на свечи акций/металлов/ETF.

---

## Уведомление о монетах без сектора

Когда пуш уходит в группу и у монеты нет сектора → **в личный чат (CHAT_ID)** отправляется сообщение:

```
⚠️ Монета без сектора: $NEWCOIN
Добавь сектор: /addsector NEWCOIN
```

Это позволяет:
- Не пропускать новые листинги (scan_unknown: true)
- Сразу видеть что нужно категоризировать
- Быстро добавить сектор через команду

---

## Telegram-команды

### 1. `/sector` — Просмотр секторов
Нажимаешь → появляются кнопки:

```
[🤖 AI (114)] [💎 Meme (63)] [🏗 L1 (92)]
[🔗 L2 (31)] [🏦 DeFi (126)] [🎮 Gaming (42)]
[👑 Blue Chip (3)] [📊 Stocks (91)] [🥇 M/C (9)]
[🖼 NFT (6)] [💰 RWA (18)] [🔮 Oracle (9)]
[💾 Infra (108)] [📦 Other (11)] [⚽ Fan (6)]
[💳 Payments (10)] [📈 ETFs (20)] [❓ Без сектора (X)]
```

Нажимаешь на сектор → список монет:

```
🤖 AI (114 монет):
RENDER, FET, TAO, WLD, IO, SIREN, XNY...
```

### 2. `/addsector SIREN` — Добавить монету в сектор
```
Добавить SIREN в сектор:

[🤖 AI] [💎 Meme] [🏗 L1]
[🔗 L2] [🏦 DeFi] [🎮 Gaming] ...
```

→ Нажимаешь 🤖 AI
→ `✅ SIREN добавлен в 🤖 AI`

```
[➕ Добавить ещё сектор] [✅ Готово]
```

→ Нажимаешь "Добавить ещё"
→ Снова кнопки секторов (без уже выбранных)
→ Нажимаешь 💎 Meme
→ `✅ SIREN: 🤖 AI, 💎 Meme`

### 3. `/movesector SIREN` — Переместить (заменить секторы)
```
SIREN сейчас в: 🤖 AI, 💎 Meme
Убрать из:

[🤖 AI ❌] [💎 Meme ❌] [Убрать все ❌]
```

→ Нажимаешь "Убрать все"
→ Секторы очищены. Выбери новые → тот же флоу как в `/addsector`

### 4. `/scanfilter` — Управление фильтром сканера
```
Секторы для сканирования трендлайнов:

✅ 🤖 AI        ✅ 💎 Meme
✅ 🏗 L1         ✅ 🔗 L2
✅ 🏦 DeFi       ✅ 🎮 Gaming
✅ 👑 Blue Chip  ❌ 📊 Stocks
❌ 🥇 M/C        ✅ 🖼 NFT
✅ 💰 RWA        ✅ 🔮 Oracle
✅ 💾 Infra      ✅ 📦 Other
✅ ⚽ Fan Token  ✅ 💳 Payments
❌ 📈 ETFs       ✅ ❓ Без сектора
```

Нажимаешь на любой → переключает ✅↔❌
Изменения сохраняются в `data/scan_settings.json`.

---

## Правки в существующих файлах

### `main.py` — Фильтр сканера (~5 строк)
```python
from core.categories import should_scan

for symbol in symbols:
    if not should_scan(symbol):
        continue
    # ... дальше как раньше
```

### `core/chart_drawer.py` — Сектор в пуше (~5 строк)
Сейчас:
```
$SIREN 🎯 TREND BREAKOUT
⏳ TF: 4H | 💰 Price: 0.0045
```
Станет:
```
$SIREN 🎯 TREND BREAKOUT
🏷 Sector: 🤖 AI / 💎 Meme
⏳ TF: 4H | 💰 Price: 0.0045
```

### `core/chart_drawer.py` — Уведомление в личку при отсутствии сектора (~10 строк)
Когда пуш уходит в группу и у монеты `sectors == []`:
```python
# После успешной отправки в GROUP_CHAT_ID:
if not sectors:
    await send_message(CHAT_ID, f"⚠️ Монета без сектора: ${symbol}\nДобавь: /addsector {symbol}")
```

### `core/tg_commands.py` — 4 новые команды (~200 строк)
- `/sector` — просмотр с кнопками
- `/addsector` — добавление
- `/movesector` — перемещение
- `/scanfilter` — фильтр сканера

### `core/tg_callbacks.py` — Обработка кнопок (~100 строк)
- `sector_view:{sector}` — показать монеты в секторе
- `sector_add:{symbol}:{sector}` — добавить
- `sector_more:{symbol}` — ещё сектор
- `sector_done:{symbol}` — готово
- `sector_remove:{symbol}:{sector}` — убрать
- `scanfilter_toggle:{sector}` — вкл/выкл

### `core/tg_reports.py` — Сектор в банке сигналов (~3 строки)
Рядом с каждой монетой — эмодзи сектора.

---

## Порядок работы

| Шаг | Что делаем | Файлы |
|-----|-----------|-------|
| 1 | `core/categories.py` — логика + `scan_settings.json` | новые |
| 2 | `/sector`, `/addsector`, `/movesector`, `/scanfilter` | `tg_commands.py` |
| 3 | Обработка кнопок | `tg_callbacks.py` |
| 4 | Сектор в пуше + уведомление в личку при ❓ | `chart_drawer.py` |
| 5 | Фильтр сканера по секторам | `main.py` |
| 6 | Сектор в банке сигналов | `tg_reports.py` |
