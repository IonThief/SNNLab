## Installation

_Create a virtual environment_

```bash
python3.10 -m venv .venv_snnlab
```

_Activate it_

```bash
source .venv_snnlab/bin/activate
```

_Install prerequisites_

```bash
pip install -U torch openmim
```

```bash
pip install "git+https://github.com/facebookresearch/detectron2.git"
```

```bash
mim install mmengine "mmdet>=3.0.0" "mmdet3d>=1.1.0" "mmcv>=2.0.0rc4, <2.2.0"
```

_Install SNNLab_

```bash
pip install .
```
