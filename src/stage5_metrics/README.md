# Метрическая проверка и ограничения наблюдаемости

Протокол фиксируется в `docs/scene-metrics-protocol.md` до расчёта новых ошибок. Скрипты рассчитаны на окружения Colab из `src/stage3_policy/setup_colab.sh`. Для симулятора нужны PYTHONPATH=/content/LIBERO, LIBERO_CONFIG_PATH=/content/libero-config, MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl и MPLBACKEND=Agg.

`window.py capture` восстанавливает записанный якорь и историю робота, получает метрические RGB-D, камеры и сегментацию геометрии. Выбирает до128точек на не-роботическое тело (seed42), сохраняет их локальные координаты. Будущие состояния не используются для подготовки входа.

```bash
SIM=/content/miniforge/envs/libero/bin/python
"$SIM" /content/stage5_metrics/window.py capture \
  --recording /content/UNITAS/outputs/stage3/stress/task008-drawer-bowl/libero_90_task08_episode000_replay \
  --frame 60 --output /content/UNITAS/outputs/stage5/window050
```

Кадр60 соответствует 50управляющим действиям после10settle-шагов. Для второго окна используется кадр110. В model-окружении подготовить input.npz штатным `unitas.scene_flow.prepare` с `--observation observation.json --queries queries.npz --query-frame world`, SceneFlow-checkpoint и output input.npz. Затем выполнить штатный SceneFlow-инференс с FP32-адаптацией для T4,10шагами,chunk256,seeds42/210921. Каждый каталог должен быть новым.

`predict_windows.py --windows <окно50> <окно100> --checkpoint <SceneFlow.pt> --florence-dir <каталог Florence>` готовит входы, загружает модель один раз и сохраняет прогнозы и commands `actions.npz`. На T4 требуется FP32-патч из setup_colab.sh. Команды конвертируются штатным LiberoEEFActionConverter в model-окружении, где доступны его геометрические зависимости.

`window.py execute --window <каталог окна> --prediction <прогноз.npz> --output <новый каталог>` в simulator-окружении восстанавливает тот же якорь и исполняет сохранённые actions.npz без перепланирования, проверяя хеш прогноза. Будущие соответствия получаются из реально достигнутых поз жёстких тел, видимость — по будущим истинным камерам/depth. Сохраняются reference.npz и replay. Продолжение старого эпизода не подставляется вместо эталона для новых действий.

`score.py --prediction <прогноз.npz> --reference <reference.npz> --output <metrics.json>` рассчитывает ADE/FDE для all/moving/static, все соответствия/видимые, и stationary baseline. Дополнительно измеряет RMS-остаток жёсткого совмещения каждого тела. Файл результата не перезаписывается.

`observability.py --output <новый каталог>` строит два состояния кружки с совпадающим наблюдением и противоположными не передаваемыми модели скоростями. Проверяет равенство хешей входа, записывает эталонные положения центра и нижнюю границу ошибки общего прогноза. Это проверка наблюдаемости, а не эмпирическая ADE/FDE UNITAS. До успешного выполнения equality-check вывод о неразличимости не делать.

Полный capture→predict→execute→score выполнен на task8/эпизод0, окнах50/100. Прогнозы, соответствия, маски и метрики сохранены в outputs/stage5. `plot_results.py --root outputs/stage5 --output outputs/stage5/figures/errors.png` строит график численных ошибок; скриншоты и видео пользователь записывает отдельно. Проверки наблюдаемости не выполнялись и оставлены в резерве.
