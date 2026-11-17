#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ ! -d /content/UNITAS/.git ]; then
  git clone https://github.com/DexForce/UNITAS.git /content/UNITAS
fi
git -C /content/UNITAS checkout c92ee809abd2862a911c61d4b19389308f7eb57e
set -e
if [ ! -x /content/miniforge/bin/conda ]; then
  curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /content/Miniforge3.sh
  bash /content/Miniforge3.sh -b -p /content/miniforge
fi
/content/miniforge/bin/conda --version

set -e
if [ ! -x /content/miniforge/envs/unitas/bin/python ]; then
  /content/miniforge/bin/conda create -n unitas python=3.10 pip -y
fi
/content/miniforge/envs/unitas/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
/content/miniforge/envs/unitas/bin/python -m pip install -e /content/UNITAS 'huggingface_hub[cli]>=0.34,<1'
if [ ! -d /content/LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /content/LIBERO
fi
if [ ! -x /content/miniforge/envs/libero/bin/python ]; then
  /content/miniforge/bin/conda create -n libero python=3.10 pip -y
fi
git -C /content/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
PY=/content/miniforge/envs/libero/bin/python
"$PY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install numpy==1.26.4 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 easydict==1.9 cloudpickle==2.1.0 h5py pyyaml termcolor future==0.18.2 matplotlib
"$PY" -m pip install -r /content/UNITAS/evaluation/requirements.txt mujoco==3.2.3
"$PY" -m pip install --no-deps -e /content/LIBERO -e /content/UNITAS
export LIBERO_CONFIG_PATH=/content/libero-config
export PYTHONPATH=/content/LIBERO${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /content/LIBERO/datasets
printf 'n\n' | "$PY" -c 'import libero.libero'

set -e
cd /content/UNITAS
/content/miniforge/bin/conda run --no-capture-output -n unitas hf download microsoft/Florence-2-large --revision 21a599d414c4d928c9032694c424fb94458e3594 --local-dir checkpoints/Florence-2-large --exclude '*.bin'
/content/miniforge/bin/conda run --no-capture-output -n unitas hf download RobbinWang123/Unitas unitas_libero_finetuned.pt --revision 1488f49761a56d7f44ab22a7e3b05a20799d5f51 --local-dir checkpoints
/content/miniforge/envs/unitas/bin/python - <<'PY'
from pathlib import Path
p = Path('src/unitas/inference/policy.py')
text = p.read_text()
old = 'self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"'
new = 'self.device.type, dtype=torch.bfloat16, enabled=False'
if old in text:
    assert text.count(old) == 1
    if not p.with_suffix('.py.original').exists():
        p.with_suffix('.py.original').write_text(text)
    p.write_text(text.replace(old, new))
else:
    assert new in text, 'Unexpected policy.py contents'
p = Path('src/unitas/scene_flow/inference.py')
text = p.read_text()
old = 'enabled=device.type == "cuda"'
new = 'enabled=False'
if old in text:
    assert text.count(old) == 1
    if not p.with_suffix('.py.original').exists():
        p.with_suffix('.py.original').write_text(text)
    p.write_text(text.replace(old, new))
else:
    assert new in text, 'Unexpected SceneFlow inference contents'
PY
