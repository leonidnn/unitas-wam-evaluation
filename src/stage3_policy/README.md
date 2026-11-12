# Запись и просмотр эпизода LIBERO

`stage3.ipynb` запускает policy UNITAS и штатный LIBERO-клиент. В первой ячейке нужно загрузить с ноутбука три файла из этой папки: `record_libero.py`, `replay_data.py` и `serve_timed.py`. Они сохраняются в `/content/stage3_policy/`; сам UNITAS клонируется отдельно в `/content/UNITAS`. Последняя ячейка вызывает обёртку `record_libero.py` с теми же аргументами, что у `evaluation/libero/client.py`.

Обёртка использует авторские `main`, `run_episode`, создание среды и settle-процедуру из `evaluation/libero/client.py`. Она добавляет наблюдение за `env.step` без изменения переданных действий, возвращаемых наблюдений и критериев успеха. Исходники авторов при этом не переписываются. В notebook сохраняется прежняя FP32-адаптация policy в копии Colab.

`serve_timed.py` вызывает авторский policy-сервер, измеряя `Policy.act` с синхронизацией CUDA. Время включает подготовку входа и преобразование выхода в команды. Клиентская обёртка отдельно пишет `policy-request-timing.jsonl` с полным временем HTTP-запроса, включая сериализацию. Эти два показателя не следует смешивать.

Для сравнения освещения используется `UNITAS_LIGHT_SCALE`: `1.0` сохраняет сцену, `0.5` уменьшает вдвое ambient/diffuse/specular RGB всех именованных источников света через штатный `robosuite.utils.mjmod.LightingModder`. Геометрия, физика, параметры камер и headlight не меняются. Модификация выполняется до первого наблюдения policy, затем наблюдения обновляются; изменённое освещение входит в сохранённый MJB. Условие записывается в протокол и влияет на его идентификатор. Это изменение условий эксперимента, а не исправление модели.

Рядом с MP4 и JSON результата появляется папка `<suite>_taskXX_episodeXXX_replay/`:

- `scene.mjb` — скомпилированная сцена MuJoCo, включая меши и текстуры;
- `states.npz` — состояния, время симуляции, действия и метки начального состояния / settle / policy;
- `manifest.json` — версия MuJoCo, параметры эпизода, протокол policy, контрольные суммы и статус завершения.

Записывается начальное состояние и состояние после каждого шага управления, включая финальное. Внутренние физические подшаги отдельно не записываются. Действие в строке `i` привело от состояния `i-1` к состоянию `i`; первая строка действий содержит NaN, поскольку до начального состояния записанного действия нет. Используется штатный `mjSTATE_INTEGRATION`, а не только положения суставов.

При обычном завершении и Python-исключении накопленные кадры сохраняются, неудачный прогон помечается в манифесте. При принудительном уничтожении процесса или потере Colab до сохранения кадры могут быть потеряны. Существующую папку эпизода запись не перезаписывает.

## Перенос на ноутбук

Из Colab нужно скачать всю папку запуска `outputs/stage3/<время>/`, а не только MP4 или NPZ. Например, после запуска можно упаковать её командой в терминале Colab:

```bash
zip -r /content/stage3-recordings.zip /content/UNITAS/outputs/stage3
```

Скачать архив через Files, распаковать на ноутбуке. Для просмотра не нужны UNITAS-веса, CUDA, robosuite или LIBERO. Требуется та же версия MuJoCo, которая записана в `manifest.json`; notebook устанавливает 3.2.3. Перенос Linux → Windows проверен на реальной записи: сохранённая сцена с текстурами загрузилась, все кадры восстановились.

```powershell
conda create -n unitas-replay python=3.10 pip -y
conda activate unitas-replay
python -m pip install mujoco==3.2.3 numpy
python src/stage3_policy/replay_libero.py --input outputs/stage3/<время>/<эпизод>_replay
```

`--input` указывает на папку с `manifest.json`, `scene.mjb` и `states.npz`. Программа открывает локальное окно MuJoCo в паузе на первом кадре:

- мышь — стандартные вращение, приближение и перемещение камеры;
- пробел — воспроизведение / пауза;
- стрелки влево / вправо — предыдущий / следующий кадр;
- Home / End — начало / конец;
- `+` / `-` — скорость воспроизведения.

Воспроизводятся записанные состояния с их временем; `mj_step` и policy не вызываются. Камеру можно менять независимо от камер, которые получала нейросеть. Это просмотр прошедшего эпизода, а не интерактивное управление роботом.

Для проверки файла без графического окна:

```powershell
python src/stage3_policy/replay_libero.py --input outputs/stage3/<время>/<эпизод>_replay --check-only
```

В проигрывателе скрыта служебная геометрия столкновений; видимость поверхностей соответствует настройкам robosuite. Боковые панели MuJoCo скрыты при открытии, чтобы оставить место сцене.

Для экспорта сравнения начала, середины и конца движения с двух неподвижных ракурсов:

```powershell
python -m pip install pillow
python src/stage3_policy/render_replay.py --input outputs/stage3/<время>/<эпизод>_replay --output-dir outputs/stage3/<время>/figures
```

`render_replay.py` сохраняет отдельные PNG и `comparison.png`. Верхний ряд — камера `agentview`, нижний — свободная боковая камера. Время взято из записи MuJoCo; новые шаги физики не выполняются.

Основа записи — `mj_saveModel`, `mj_getState`, `mj_setState` из [MuJoCo 3.2.3](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html); доступ к исходным `MjModel`/`MjData` взят из [robosuite 1.4.0 binding_utils.py](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/binding_utils.py). Интерфейс просмотра — штатный [MuJoCo passive viewer](https://mujoco.readthedocs.io/en/3.2.3/python.html#passive-viewer).

## Две серии по десять эпизодов

Для полной оценки в первую ячейку вместе с тремя обязательными файлами загружаются `run_evaluation.py` и `summarize_runs.py`. После установки окружений и скачивания весов выполняются команды ниже. Пробный эпизод последней ячейки должен закончиться; порт 8000 должен быть свободен. Каталоги результатов должны быть новыми.

```bash
SIM=/content/miniforge/envs/libero/bin/python
"$SIM" /content/stage3_policy/run_evaluation.py --condition baseline --episodes 10 --output-dir /content/UNITAS/outputs/stage3/baseline
"$SIM" /content/stage3_policy/run_evaluation.py --condition dim_light --episodes 10 --output-dir /content/UNITAS/outputs/stage3/dim_light
"$SIM" /content/stage3_policy/summarize_runs.py --baseline /content/UNITAS/outputs/stage3/baseline --changed /content/UNITAS/outputs/stage3/dim_light --output /content/UNITAS/outputs/stage3/summary.json
```

`run_evaluation.py` последовательно запускает авторские сервер и клиент, ждёт готовности policy и завершает свои процессы после прогона. Используются `libero_90`, задача с точной инструкцией `put the white mug on the plate`, начальные состояния 0–9, client seed 42, model seed 160484, 10 шагов денойзинга, 10 исполняемых действий, 10 settle-шагов и горизонт 800 действий. `run.json` сохраняет команды, ревизию UNITAS, хеши обёрток и статус. `summarize_runs.py` проверяет совпадение протоколов, показывает отсутствующие эпизоды и считает парное сравнение результатов. При неполной серии SR не объявляется.

`plot_results.py` строит из итогового JSON график SR, парных длин эпизодов и времени policy. Файл загружается в Colab вместе с остальными скриптами и вызывается после завершения обеих серий:

```bash
"$SIM" /content/stage3_policy/plot_results.py --summary /content/UNITAS/outputs/stage3/summary.json --output /content/UNITAS/outputs/stage3/metrics.png
```

## Scratch-checkpoint

`stage3_scratch.ipynb` сделан на основе пяти ячеек `stage3.ipynb`: закреплены ревизии исходников и весов, выбран scratch-checkpoint, последняя ячейка вызывает две серии по десять эпизодов. Исходный notebook сохранён. `run_evaluation.py --checkpoint checkpoints/unitas_libero_scratch.pt` выбирает веса; без аргумента используются прежние finetuned-веса.

У опубликованного scratch-checkpoint горизонт прогноза и истории равен 30 шагам при 20 Hz; у ранее использованного finetuned — 20. Сравниваются штатные конфигурации: временное представление поддерживает разные длины траекторий, а принудительная замена горизонта изменила бы настройки выпущенных весов. В обоих случаях исполняются первые 10 действий, затем обновляется наблюдение. Четвёртая ячейка проверяет родную scratch-конфигурацию 30/30. Это сравнение готовых policies; эффект предобучения отдельно от горизонта и объёма истории не изолирован.

`compare_checkpoints.py --finetuned outputs/stage3 --scratch outputs/stage3/scratch --native-temporal-config --output outputs/stage3/scratch/comparison.json` проверяет четыре серии, полноту и совпадение начальных состояний/settle перед сравнением SR, шагов и времени. Аргумент разрешает только указанное различие временных настроек 20/30; остальные отличия протокола отклоняются. Без аргумента требуется совпадение временных настроек. При недостающих эпизодах SR соответствующей серии остаётся неопределённым. Время относится ко всему синхронизированному `Policy.act`; сравнение разных сеансов описательное.
