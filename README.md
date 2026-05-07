# Avito Room Type Classification
## Кейс 3 | МИФИ Практический курс 2025-2026

Классификация фотографий недвижимости по типу комнаты.  
Основная метрика: **Macro F1** (среднее по классам без весов).

Проект прошел полный цикл: baseline -> очистка/расширение данных -> усиление бэкбона -> LR sweep -> TTA -> 3-fold CV -> финальный ансамблевый инференс.

---

## TL;DR по результатам

- **Лучший single-model на val:** `0.7228` (`exp_convnext_base_class_focus_lr1e4`)
- **Лучший CV fold:** `0.7704` (`exp_convnext_base_clean_lr1e4_fold3`)
- **3-fold среднее:** `0.7559` (std `0.0135`)
- **ConvNeXt-only val ensemble (weight fit on val):** `0.7543` (оптимистично)
- **Финальный тестовый артефакт:** `outputs/ensemble_convnext_3fold_tta/submission.csv` (`48003` строк)

---

## Важные технические выводы

- Низкий стартовый скор (`~0.006`) был вызван неверными путями к данным в конфигах.
- Внешние данные в "сыром" виде ухудшали результат из-за domain shift.
- Очистка внешних данных (domain filter + class cap) дала прирост.
- `ConvNeXt` на GPU дал ключевой скачок качества.
- TTA и fold-ensemble увеличили устойчивость относительно одиночного чекпоинта.

---

## Хронология экспериментов

Все значения ниже — `Best val Macro F1`.

### 1) Этап EfficientNet

| Эксперимент | Конфиг | Macro F1 | Комментарий |
|---|---|---:|---|
| Baseline after path fix | `configs/baseline.yaml` | `0.5999` | Возврат к рабочему уровню |
| Ratio filter | `configs/exp2_ratio_filter.yaml` | `0.5706` | Сильная потеря данных |
| Weighted loss | `configs/exp3_weighted_loss.yaml` | `0.5963` | Около baseline |
| Full train (raw external) | `configs/exp_full_train.yaml` | `0.5756` | Просадка из-за domain shift |
| Full clean E1 | `configs/exp_full_train_clean_e1.yaml` | `0.6101` | Лучше baseline |
| Full clean E2 weighted | `configs/exp_full_train_clean_e2.yaml` | `0.6087` | Чуть хуже E1 |
| Full clean E1 160px | `configs/exp_full_train_clean_e1_160.yaml` | `0.5927` | Потеря деталей |

### 2) Ранний ансамбль (до ConvNeXt)

- `ensemble_b012_f017_e133_e238`: `0.6427` (валидационно)

### 3) ConvNeXt / EVA этап

| Эксперимент | Конфиг | Macro F1 | Комментарий |
|---|---|---:|---|
| ConvNeXt clean | `configs/exp_convnext_base_clean.yaml` | `0.6880` | Первый большой скачок |
| ConvNeXt original | `configs/exp_convnext_base_original.yaml` | `0.6847` | Чуть хуже clean |
| EVA02 small clean | `configs/exp_eva02_small_clean.yaml` | `0.5814` | Не взлетел в этой настройке |

### 4) LR sweep для ConvNeXt

| Конфиг | Macro F1 |
|---|---:|
| `configs/exp_convnext_base_clean_lr1e4.yaml` | `0.7176` |
| `configs/exp_convnext_base_clean_lr1p5e4.yaml` | `0.7147` |
| `configs/exp_convnext_base_clean_lr2e4.yaml` | `0.6893` |

Выбран оптимальный LR: **`1e-4`**.

### 5) TTA и ансамбли

- Для `exp_convnext_base_clean_lr1e4`:
  - single: `0.7176`
  - flip-TTA: `0.7215`
- ConvNeXt-only ensemble с подбором весов на val: `0.7543`

### 6) Class-focused clean

- `configs/exp_convnext_base_class_focus_lr1e4.yaml`: `0.7228`

### 7) 3-fold CV (последние эксперименты)

| Fold | Конфиг | Macro F1 |
|---|---|---:|
| 1 | `configs/exp_convnext_base_clean_lr1e4_fold1.yaml` | `0.7378` |
| 2 | `configs/exp_convnext_base_clean_lr1e4_fold2.yaml` | `0.7595` |
| 3 | `configs/exp_convnext_base_clean_lr1e4_fold3.yaml` | `0.7704` |

Итого: **mean `0.7559`, std `0.0135`**.

### 8) Финальный инференс (последний шаг)

Скрипт `src/inference_ensemble_3fold.py`:
- загружает 3 fold-модели ConvNeXt,
- применяет flip-TTA,
- усредняет logits,
- сохраняет итоговый сабмит в `outputs/ensemble_convnext_3fold_tta/submission.csv`.

---

## Структура репозитория

```text
.
├─ configs/                      # Все экспериментальные конфиги
├─ scripts/
│  ├─ build_full_train.py
│  ├─ build_full_train_clean.py
│  ├─ build_full_train_class_focus.py
│  └─ run_convnext_3fold.py
├─ src/
│  ├─ train.py
│  ├─ inference.py
│  ├─ inference_ensemble_3fold.py
│  ├─ dataset.py
│  └─ model.py
├─ outputs/
│  └─ ensemble_convnext_3fold_tta/submission.csv
├─ EXPERIMENT_REPORT.md          # Расширенный handover-отчет
├─ run_baseline.py
└─ requirements.txt
```

---

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt
```

Windows (рекомендуемый запуск из `.venv`):

```powershell
# Проверка CUDA
& "c:\Users\Константин\Desktop\avito_hackaton\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Baseline
& "c:\Users\Константин\Desktop\avito_hackaton\.venv\Scripts\python.exe" run_baseline.py

# Лучший single-model (class-focused ConvNeXt)
& "c:\Users\Константин\Desktop\avito_hackaton\.venv\Scripts\python.exe" src/train.py --config configs/exp_convnext_base_class_focus_lr1e4.yaml

# 3-fold CV
& "c:\Users\Константин\Desktop\avito_hackaton\.venv\Scripts\python.exe" scripts/run_convnext_3fold.py
```

---

## Данные и оговорки

- В исходной постановке фигурирует 19 классов, но в текущей разметке используется `0..19` (20 классов).
- Самые сложные/редкие классы: кабинет, универсальная, предметы интерьера, гардеробная.
- Метрики в репозитории — offline validation; public/private leaderboard может отличаться.
- Val-оптимизированные ансамблевые веса могут завышать локальную оценку.

---

## Финальный артефакт

- Текущий итоговый файл для отправки: `outputs/ensemble_convnext_3fold_tta/submission.csv`
- Формат:

```csv
image_id_ext,Predicted
12345,11
12346,7
```

Для детального пошагового отчета см. `EXPERIMENT_REPORT.md`.

