# Собственная сцена

`stage4.ipynb` повторяет пять ячеек `src/stage2_transfer/stage2.ipynb`. В последней ячейке вход заменён на экспериментальный `media_processing/prepared/cup-002/input.npz`: при запуске в Colab notebook предлагает выбрать этот файл, если он ещё не загружен. Инференс использует тот же checkpoint, FP32, 10 шагов диффузии, chunk-size 256 и seeds 42/210921.

Вход построен из одного RGB-видео и оценённой глубины. Его совместимость с форматом LIBERO не означает физическую калибровку или соответствие руки роботическому захвату. Сравнение с отложенным видео даёт только качественную 2D-проверку.

Для сравнения сохранённого прогноза с разрешёнными после инференса кадрами используется `compare_heldout.py`:

```bash
python src/stage4_own_scene/compare_heldout.py \
  --input media_processing/prepared/cup-002/input.npz \
  --prediction outputs/stage4/cup-002/scene_flow.npz \
  --frames outputs/stage4/cup-002/heldout-frames \
  --output outputs/stage4/cup-002/comparison
```

Папка `--frames` содержит `t0.png`, `t05.png` и `t1.png`, извлечённые из отдельной отложенной секунды уже после завершения прогноза. Скрипт строит 2D-проекции и сводку предсказанных смещений. Он не вычисляет 3D ADE/FDE без эталонных соответствий.
