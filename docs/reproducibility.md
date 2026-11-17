# Воспроизведение и материалы экспериментов

## Приложение Все эпизоды policy

В таблице доступны все сорок попыток четырёх основных серий, включая одинаковые начальные состояния 0–9 в каждой. Номер эпизода обозначает начальное состояние, а не отдельную задачу. Это полный набор оценки; демонстрационные ролики в основном тексте не заменяют остальные попытки. Дополнительные серии с трудными задачами и с толчком (`outputs/stage3/stress/`) сохранены как архивы реплеев без штатных MP4; их записи выполняются пользователем из восстановленных сцен.

| Начальное состояние | Finetuned обычный свет | Finetuned уменьшенный свет | Scratch обычный свет | Scratch уменьшенный свет |
|---|---|---|---|---|
| 0 | [Видео 00](../outputs/stage3/baseline/libero_90_task72_episode000.mp4) | [Видео 00](../outputs/stage3/dim_light/libero_90_task72_episode000.mp4) | [Видео 00](../outputs/stage3/scratch/baseline/libero_90_task72_episode000.mp4) | [Видео 00](../outputs/stage3/scratch/dim_light/libero_90_task72_episode000.mp4) |
| 1 | [Видео 01](../outputs/stage3/baseline/libero_90_task72_episode001.mp4) | [Видео 01](../outputs/stage3/dim_light/libero_90_task72_episode001.mp4) | [Видео 01](../outputs/stage3/scratch/baseline/libero_90_task72_episode001.mp4) | [Видео 01](../outputs/stage3/scratch/dim_light/libero_90_task72_episode001.mp4) |
| 2 | [Видео 02](../outputs/stage3/baseline/libero_90_task72_episode002.mp4) | [Видео 02](../outputs/stage3/dim_light/libero_90_task72_episode002.mp4) | [Видео 02](../outputs/stage3/scratch/baseline/libero_90_task72_episode002.mp4) | [Видео 02](../outputs/stage3/scratch/dim_light/libero_90_task72_episode002.mp4) |
| 3 | [Видео 03](../outputs/stage3/baseline/libero_90_task72_episode003.mp4) | [Видео 03](../outputs/stage3/dim_light/libero_90_task72_episode003.mp4) | [Видео 03](../outputs/stage3/scratch/baseline/libero_90_task72_episode003.mp4) | [Видео 03](../outputs/stage3/scratch/dim_light/libero_90_task72_episode003.mp4) |
| 4 | [Видео 04](../outputs/stage3/baseline/libero_90_task72_episode004.mp4) | [Видео 04](../outputs/stage3/dim_light/libero_90_task72_episode004.mp4) | [Видео 04](../outputs/stage3/scratch/baseline/libero_90_task72_episode004.mp4) | [Видео 04](../outputs/stage3/scratch/dim_light/libero_90_task72_episode004.mp4) |
| 5 | [Видео 05](../outputs/stage3/baseline/libero_90_task72_episode005.mp4) | [Видео 05](../outputs/stage3/dim_light/libero_90_task72_episode005.mp4) | [Видео 05](../outputs/stage3/scratch/baseline/libero_90_task72_episode005.mp4) | [Видео 05](../outputs/stage3/scratch/dim_light/libero_90_task72_episode005.mp4) |
| 6 | [Видео 06](../outputs/stage3/baseline/libero_90_task72_episode006.mp4) | [Видео 06](../outputs/stage3/dim_light/libero_90_task72_episode006.mp4) | [Видео 06](../outputs/stage3/scratch/baseline/libero_90_task72_episode006.mp4) | [Видео 06](../outputs/stage3/scratch/dim_light/libero_90_task72_episode006.mp4) |
| 7 | [Видео 07](../outputs/stage3/baseline/libero_90_task72_episode007.mp4) | [Видео 07](../outputs/stage3/dim_light/libero_90_task72_episode007.mp4) | [Видео 07](../outputs/stage3/scratch/baseline/libero_90_task72_episode007.mp4) | [Видео 07](../outputs/stage3/scratch/dim_light/libero_90_task72_episode007.mp4) |
| 8 | [Видео 08](../outputs/stage3/baseline/libero_90_task72_episode008.mp4) | [Видео 08](../outputs/stage3/dim_light/libero_90_task72_episode008.mp4) | [Видео 08](../outputs/stage3/scratch/baseline/libero_90_task72_episode008.mp4) | [Видео 08](../outputs/stage3/scratch/dim_light/libero_90_task72_episode008.mp4) |
| 9 | [Видео 09](../outputs/stage3/baseline/libero_90_task72_episode009.mp4) | [Видео 09](../outputs/stage3/dim_light/libero_90_task72_episode009.mp4) | [Видео 09](../outputs/stage3/scratch/baseline/libero_90_task72_episode009.mp4) | [Видео 09](../outputs/stage3/scratch/dim_light/libero_90_task72_episode009.mp4) |

Штатные MP4 записаны при 30 кадрах в секунду для управления с частотой 20 Гц: движение ускорено примерно в 1,5 раза, ожидание ответа модели не показано, финальное состояние после последнего действия отсутствует. Для оценки последнего состояния использованы отдельно сохранённые состояния MuJoCo, а не длительность MP4.

## Приложение Воспроизведение и источники результатов

Все приведённые результаты получены с опубликованными весами, без обучения и дообучения. Инференс выполнялся в Colab на Tesla T4; для совместимости в рабочих копиях отключён BF16-autocast и использован FP32. Указанные ниже команды описывают запуск после подготовки окружений и весов; заново в ходе подготовки отчёта они не выполнялись.

| Компонент | Зафиксированная версия |
|---|---|
| Исходники UNITAS | `c92ee809abd2862a911c61d4b19389308f7eb57e` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Архив кода исследования | `4eb7ae021f07b285e56fb1b1fd4d45c4afd0ca54` |
| Python модели | 3.10; для SceneFlow 3.10.22 |
| PyTorch и torchvision | 2.5.1+cu121 и 0.20.1+cu121 |
| Transformers и NumPy | 4.57.1 и 1.26.4 |
| MuJoCo и robosuite | 3.2.3 и 1.4.0 |
| Hugging Face UNITAS | `1488f49761a56d7f44ab22a7e3b05a20799d5f51` |
| Hugging Face Florence 2 large | `21a599d414c4d928c9032694c424fb94458e3594` |

Код исследования доступен в [репозитории](https://github.com/leonidnn/unitas-wam-evaluation/tree/4eb7ae021f07b285e56fb1b1fd4d45c4afd0ca54). Для повторения следует использовать зафиксированные ревизии, а не текущие ветки upstream. Подробный пример установки двух окружений и загрузки весов с закреплёнными ревизиями находится в `src/stage3_policy/stage3_scratch.ipynb`. Для finetuned нужно выбрать соответствующий checkpoint и сохранить родные длины истории и прогноза: 20 шагов у finetuned и 30 у scratch, при 20 Гц. Исходные ноутбуки stage1–3 местами клонируют текущую ветку без `checkout`, поэтому сами по себе не закрепляют окружение эксперимента. Точные версии и FP32-патчи сохранены рядом с журналами.

### Запуск SceneFlow

```bash
python src/unitas/scene_flow/inference.py \
  --checkpoint checkpoints/unitas_scene_flow.pt \
  --florence-dir checkpoints/Florence-2-large \
  --input assets/scene_flow/observations/robotwin_7c59d8eb33_0088.npz \
  --output outputs/repeat-stage1/scene_flow.npz \
  --device cuda --steps 10 --chunk-size 256 \
  --policy-seed 42 --scene-seed 210921
```

Команда выполняется из корня UNITAS в подготовленном GPU-окружении с указанной FP32-адаптацией. Для второго примера заменяются вход на `libero_3d040e000e_0077.npz` и выходной каталог. Размер порции 256 сохраняется: смена разбиения точек меняет последовательность генерации случайного шума и может изменить конкретный прогноз. Для собственного примера нужен отдельно подготовленный `input.npz`; обычное RGB-видео не является готовым входом этой команды.

```bash
python src/unitas/scene_flow/visualize.py \
  --input outputs/repeat-stage1/scene_flow.npz \
  --host 127.0.0.1 --port 8080
```

Просмотрщик открывает уже сохранённые координаты и не требует нового инференса. Он не исполняет действия и не проверяет физические контакты. Проекции этапа 2 воспроизводятся скриптом `src/stage2_transfer/project_prediction.py` с аргументами `--input`, `--prediction`, `--output-dir`; детали выбора точек сохранены в файле [проверки проекций](../outputs/stage2/libero_3d040e000e_0077/projections-20261112/verification.json).

### Две серии policy

```bash
SIM=/content/miniforge/envs/libero/bin/python
"$SIM" /content/stage3_policy/run_evaluation.py \
  --condition baseline --episodes 10 \
  --checkpoint checkpoints/unitas_libero_finetuned.pt \
  --output-dir /content/UNITAS/outputs/repeat-policy/baseline
"$SIM" /content/stage3_policy/run_evaluation.py \
  --condition dim_light --episodes 10 \
  --checkpoint checkpoints/unitas_libero_finetuned.pt \
  --output-dir /content/UNITAS/outputs/repeat-policy/dim_light
```

Эти команды рассчитаны на структуру Colab из notebook: исходники в `/content/UNITAS`, LIBERO в `/content/LIBERO`, обёртки в `/content/stage3_policy`, отдельные окружения модели и симулятора. Выходные каталоги должны быть новыми. Для scratch выбирается `unitas_libero_scratch.pt`. Сравнение четырёх готовых серий выполняет `src/stage3_policy/compare_checkpoints.py` с `--finetuned`, `--scratch`, `--native-temporal-config` и `--output`: разрешается только предусмотренное различие горизонтов, а не произвольные различия протокола.

### Серия с физическим толчком

```bash
"$SIM" /content/stage3_policy/run_evaluation.py \
  --condition stress --light-scale 1.0 --task-id 72 \
  --push-force-n 0.4 --push-start-step 60 --push-duration-steps 3 \
  --episodes 10 --checkpoint checkpoints/unitas_libero_finetuned.pt \
  --output-dir /content/UNITAS/outputs/stage3/stress/task072-push040
```

Сила приложена к кружке по мировой оси X через `xfrc_applied` на управляющих шагах 60–62; калибровка амплитуд сохранена в `outputs/stage3/stress/push-calibration*.json`. Серия с толчком повторялась на новом runtime после потери предыдущего; эпизод 0 воспроизвёлся побитово — SHA-256 исходов совпали с сохранёнными копиями, что подтверждает детерминизм воспроизведения при зафиксированных seeds и ревизиях.

### Контрольные суммы весов

| Файл | SHA 256 |
|---|---|
| `unitas_scene_flow.pt` | `49c44021c9e32ac7ccf7d69eabeaab0f7480a383a19955d4803d4fc52352b71b` |
| `unitas_libero_finetuned.pt` | `a60972420b4049128170592af9953e6ee7fa591a7cfdf6d26560a23371d7b299` |
| `unitas_libero_scratch.pt` | `20c13ae9e580a9279cba236ba599c4600506f98486127658232379b56af1e81d` |

Результаты SceneFlow сохранены в `/outputs/stage1/` и `/outputs/stage2/`. Для policy доступны [сводка основного сравнения](../outputs/stage3/summary.json), [сравнение четырёх серий](../outputs/stage3/scratch/comparison.json), индивидуальные исходы и времена запросов в `/outputs/stage3/`. Дополнительные серии с трудными задачами и толчком сохранены в `/outputs/stage3/stress/`: полные реплеи (`scene.mjb`, `states.npz`, `manifest.json`), исходы эпизодов, тайминги и [диагностика контактов неудачного и парного успешного эпизодов](../outputs/stage3/stress/task072-push040/diagnostics/episode000/summary.json). Для собственной сцены — [журнал](../outputs/stage4/cup-002/run.log), [сводка запуска](../outputs/stage4/cup-002/run-summary.json) и [проверка проекций](../outputs/stage4/cup-002/comparison/projection-summary.json). Все пути указаны от корня репозитория.

Веса и результаты в `outputs/` игнорируются Git и доступны в локальном проекте. Исходное RGB-видео и подготовленные материалы `media_processing/` включены в приватный репозиторий. Все пути и команды выше относятся к корню проекта.

