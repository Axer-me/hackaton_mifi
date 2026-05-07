# Полный технический отчет (handover) по задаче Room Type Classification

Этот документ написан как **самодостаточный handover**: его можно открыть в новом чате и продолжить работу без потери контекста.

---

## 1) Цель проекта и метрика

- **Задача:** классификация изображений комнат по классам конкурса.
- **Целевая метрика:** `Macro F1`.
- **Важно:** `Macro F1` усредняет качество по классам **без весов**, поэтому:
  - провал в редком/сложном классе может сильно снизить общий скор,
  - просто высокая accuracy не гарантирует высокий Macro F1.

---

## 2) Исходные проблемы и их решение

### 2.1 Критическая проблема baseline (в начале)

Симптом:
- метрика была около `0.006`,
- модель предсказывала почти один класс.

Причина:
- неверные пути к изображениям в конфигах,
- датасет подсовывал заглушки (черные картинки).

Что сделано:
- исправлены пути в `baseline/configs/*.yaml`,
- проведен sanity-check на 1 эпоху.

Результат:
- baseline вернулся к ожидаемому уровню `~0.60`.

### 2.2 Проблема с GPU

Симптом:
- несмотря на наличие RTX 3060, обучение шло на CPU.

Причина:
- в `.venv` была CPU-сборка `torch`.
- часть запусков шла через системный `python` (`C:\\Program Files\\Python312\\python.exe`), а не через `.venv`.

Что сделано:
- установлены wheel-файлы CUDA из локальной папки `torch/`:
  - `torch-2.9.1+cu128`
  - `torchvision-0.24.1+cu128`
  - `torchaudio-2.9.1+cu128`
- подтверждено:
  - `torch.cuda.is_available() == True`
  - `device_name = NVIDIA GeForce RTX 3060`

---

## 3) Что было реализовано в коде

## 3.1 Улучшение train-loop

Файл: `baseline/src/train.py`

Добавлено:
- `warmup_epochs`,
- `early_stopping_patience`,
- сохранение `lr` в `history.csv`,
- явный вывод лучшей эпохи.

Зачем:
- стабилизировать обучение тяжелых моделей,
- уменьшить деградацию после пика,
- сократить лишние эпохи.

## 3.2 Скрипты подготовки данных

Файлы:
- `baseline/scripts/build_full_train.py`
- `baseline/scripts/build_full_train_clean.py`
- `baseline/scripts/build_full_train_class_focus.py`

Что они делают:
- объединяют базовый train и внешние данные,
- скачивают/переиспользуют изображения,
- фильтруют внешние источники (доменный контроль),
- ограничивают прирост примеров по классам (cap),
- делают class-focused вариант для трудных классов.

## 3.3 Скрипт CV-обучения

Файл: `baseline/scripts/run_convnext_3fold.py`

Что делает:
- строит stratified 3-fold split из `train_df_full_clean.csv`,
- генерирует fold-конфиги,
- последовательно запускает обучение ConvNeXt по 3 фолдам.

---

## 4) Сводка всех экспериментов (хронология + смысл)

> Все значения — `Best val Macro F1` из `classification_report.txt`.

### 4.1 EfficientNet этап

- `baseline_efficientnet_b0` -> **0.5999**  
  Базовая опорная точка после фикса путей.

- `exp2_ratio_filter` -> **0.5706**  
  Идея: убрать шум по `ratio>=0.7`.  
  Итог: данных стало слишком мало, качество упало.

- `exp3_weighted_loss` -> **0.5963**  
  Идея: компенсировать дисбаланс весами классов.  
  Итог: почти baseline, ощутимого выигрыша нет.

- `exp_full_train_efficientnet_b0` -> **0.5756**  
  Идея: добавить все внешние данные "как есть".  
  Итог: хуже из-за domain shift.

- `exp_full_train_clean_e1` -> **0.6101**  
  Идея: clean external + warmup/early stop.  
  Итог: лучше baseline.

- `exp_full_train_clean_e2_weighted` -> **0.6087**  
  Идея: clean + weighted.  
  Итог: чуть хуже E1.

- `exp_full_train_clean_e1_160` -> **0.5927**  
  Идея: ускорение на `img_size=160`.  
  Итог: потеря детализации, метрика ниже.

### 4.2 Первые ансамбли (до ConvNeXt этапа)

- `ensemble_b012_f017_e133_e238` -> **0.6427** (валидационно)  
  Улучшение против single EfficientNet, но ниже, чем поздний ConvNeXt.

### 4.3 ConvNeXt/EVA этап на GPU

- `exp_convnext_base_clean` -> **0.6880**  
  Первый крупный скачок, почти цель 0.70.

- `exp_convnext_base_original` -> **0.6847**  
  Проверка "без внешних". Чуть хуже clean (`-0.0033`).

- `exp_eva02_small_clean` -> **0.5814**  
  На данной настройке не взлетел (деградация после 1-й эпохи).

### 4.4 LR sweep ConvNeXt (последовательные GPU-запуски)

- `exp_convnext_base_clean_lr1e4` -> **0.7176**
- `exp_convnext_base_clean_lr1p5e4` -> **0.7147**
- `exp_convnext_base_clean_lr2e4` -> **0.6893**

Вывод:
- оптимальный LR в этой серии — **`1e-4`**.

### 4.5 TTA и ConvNeXt-only ensemble

На `exp_convnext_base_clean_lr1e4`:
- single -> `0.7176`
- flip-TTA -> **0.7215**

ConvNeXt-only ensemble с TTA (подбор весов на val):
- **0.7543**

Важно:
- это вал-оптимизация на том же `val_df`, показатель может быть **оптимистичным**.

### 4.6 Class-focused data + ConvNeXt

- `exp_convnext_base_class_focus_lr1e4` -> **0.7228**

Вывод:
- class-focused очистка дала небольшой, но реальный прирост против `0.7176`.

### 4.7 3-fold ConvNeXt CV

Эксперименты:
- `exp_convnext_base_clean_lr1e4_fold1` -> **0.7378**
- `exp_convnext_base_clean_lr1e4_fold2` -> **0.7595**
- `exp_convnext_base_clean_lr1e4_fold3` -> **0.7704**

Сводно:
- mean = **0.7559**
- std = **0.0135**
- диапазон = `0.7378..0.7704`

Интерпретация:
- это не "одна итоговая метрика модели", а качество по разным разбиениям.
- ожидаемый рабочий уровень текущего pipeline: примерно **0.75-0.76**.

---

## 5) Что показали собранные вами изображения

## Краткий итог

- **Да, они полезны**, но только после контроля качества.
- В "сыром" виде часто вредят.

## Почему вредят без очистки

- много каталожных/e-commerce изображений,
- студийный стиль, отличающийся от real-estate фото,
- сильный `domain shift` относительно `val_df`.

## Почему становятся полезны после очистки

- доменный фильтр удаляет явно нецелевые источники,
- cap по классам снижает перекос,
- class-focused правила уменьшают шум в трудных классах.

## Трудные классы и проблемы

Наибольшие системные риски:
- `кабинет`,
- `универсальная`,
- `предметы интерьера`,
- пограничные пары (`гостиная/спальня`, `ванная/санузел/туалет`).

Причины:
- визуальная близость классов,
- неоднозначные внешние примеры,
- разная стилистика источников.

---

## 6) Текущее лучшее состояние

В зависимости от того, что считать:

- **Лучший single-model на стандартном val (500):**  
  `exp_convnext_base_class_focus_lr1e4` -> **0.7228**

- **Лучший single-model среди "чистых" ConvNeXt LR sweep:**  
  `exp_convnext_base_clean_lr1e4` -> **0.7176**

- **Лучший single-model + TTA:**  
  `convnext_base_clean_lr1e4` -> **0.7215**

- **Best CV fold (не итог):**  
  fold3 -> **0.7704**

- **3-fold среднее:**  
  **0.7559**

- **ConvNeXt-only ensemble на val (weights fitted on val):**  
  **0.7543** (оптимистичная вал-оценка).

---

## 7) Полный список ключевых артефактов

## 7.1 Данные

- `baseline/data/train_df_full.csv`
- `baseline/data/train_df_full_clean.csv`
- `baseline/data/train_df_full_class_focus.csv`
- `train_images_full/train_images_full`
- `train_images_full_clean/train_images_full_clean`
- `train_images_full_class_focus/train_images_full_class_focus`

## 7.2 Скрипты

- `baseline/scripts/build_full_train.py`
- `baseline/scripts/build_full_train_clean.py`
- `baseline/scripts/build_full_train_class_focus.py`
- `baseline/scripts/run_convnext_3fold.py`

## 7.3 Конфиги (основные)

- `baseline/configs/exp_convnext_base_clean.yaml`
- `baseline/configs/exp_convnext_base_original.yaml`
- `baseline/configs/exp_convnext_base_clean_lr1e4.yaml`
- `baseline/configs/exp_convnext_base_clean_lr1p5e4.yaml`
- `baseline/configs/exp_convnext_base_clean_lr2e4.yaml`
- `baseline/configs/exp_convnext_base_class_focus_lr1e4.yaml`
- `baseline/configs/exp_convnext_base_clean_lr1e4_fold1.yaml`
- `baseline/configs/exp_convnext_base_clean_lr1e4_fold2.yaml`
- `baseline/configs/exp_convnext_base_clean_lr1e4_fold3.yaml`

## 7.4 Выходы

- `baseline/outputs/*/history.csv`
- `baseline/outputs/*/classification_report.txt`
- `baseline/outputs/*/best_model.pth`

---

## 8) Команды для быстрого воспроизведения

### 8.1 Обучение лучшего single-model

`c:\\Users\\Константин\\Desktop\\avito_hackaton\\.venv\\Scripts\\python.exe src/train.py --config configs/exp_convnext_base_class_focus_lr1e4.yaml`

### 8.2 LR sweep (последовательно)

1. `...python.exe src/train.py --config configs/exp_convnext_base_clean_lr1e4.yaml`
2. `...python.exe src/train.py --config configs/exp_convnext_base_clean_lr1p5e4.yaml`
3. `...python.exe src/train.py --config configs/exp_convnext_base_clean_lr2e4.yaml`

### 8.3 3-fold CV

`...python.exe baseline/scripts/run_convnext_3fold.py`

### 8.4 Проверка CUDA в окружении

`...python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

---

## 9) Рекомендованный финальный путь к submission

1. Брать 3 fold-модели ConvNeXt (`fold1/2/3`).
2. Делать инференс с TTA.
3. Усреднять logits fold-моделей.
4. Сохранять финальный `submission.csv`.

Это дает лучшую устойчивость, чем одиночный чекпоинт.

---

## 10) Риски и важные оговорки

- Все текущие метрики — **offline validation** на `val_df`; leaderboard может отличаться.
- Ensemble c подбором весов на этом же val может завышать оценку.
- Для честной оценки веса ансамбля лучше фиксировать заранее или подбирать на отдельном holdout.
- Для production/финального сабмита важнее стабильность (fold ensemble), чем пик на одном запуске.

---

## 11) Короткий статус на момент отчета

- Базовая цель `0.70+` достигнута.
- Наиболее надежный стек: `ConvNeXt + clean/class-focused data + TTA + CV ensemble`.
- Следующий практический шаг: собрать финальный 3-fold инференс-ансамбль и выгрузить `submission.csv`.

---

## 12) Финальные шаги, выполненные после плана 2-5

### 12.1 Шаг 2 — TTA на лучшем single-model

Модель:
- `exp_convnext_base_clean_lr1e4`

Результат:
- single: `0.7176`
- flip-TTA: `0.7215`

Вывод:
- TTA дал ожидаемый небольшой и стабильный прирост.

### 12.2 Шаг 3 — ConvNeXt-only ensemble (валидационный поиск весов)

В ансамбль входили:
- `clean_lr1e4`
- `clean_lr1p5e4`
- `clean_lr2e4`
- `clean_base`
- `orig_base`

Результат:
- `BEST_ENSEMBLE = 0.7543` на `val_df`.

Оговорка:
- веса подбирались на том же `val_df`, поэтому оценка может быть завышена (optimistic validation fit).

### 12.3 Шаг 4 — class-focused очистка + retrain

Скрипт:
- `baseline/scripts/build_full_train_class_focus.py`

Новый train:
- `baseline/data/train_df_full_class_focus.csv`
- размер: `4598` строк

Эксперимент:
- `exp_convnext_base_class_focus_lr1e4`

Результат:
- `Best val Macro F1 = 0.7228`

Вывод:
- class-focused чистка помогла улучшить single-model относительно `0.7176`.

### 12.4 Шаг 5 — 3-fold CV ConvNeXt

Скрипт:
- `baseline/scripts/run_convnext_3fold.py`

Fold-результаты:
- fold1: `0.7378`
- fold2: `0.7595`
- fold3: `0.7704`

Сводно:
- mean: `0.7559`
- std: `0.0135`

Вывод:
- модель устойчиво попадает в целевой диапазон `0.70-0.75`, среднее CV даже выше.

### 12.5 Финальный инференс-ансамбль 3 fold моделей

Добавлен скрипт:
- `baseline/src/inference_ensemble_3fold.py`

Что делает:
- грузит 3 fold-модели ConvNeXt,
- применяет flip-TTA на тесте,
- усредняет logits,
- формирует submission.

Итоговый файл:
- `baseline/outputs/ensemble_convnext_3fold_tta/submission.csv`
- строк: `48003`

Распределение предсказаний по классам сохранено в выводе запуска.

