# LabelYolo26

A desktop image-annotation tool for building **YOLO26** datasets and training models, all in one app.
LabelYolo26 is a fork of [labelme](https://github.com/wkentaro/labelme) (5.8.3). It adds YOLO dataset export, GPU training and retraining, and AI-assisted labelling with your own YOLO models or with text prompts (Grounding DINO + SAM2).

It supports three task types:

| Mode | Shapes you draw | YOLO task |
|---|---|---|
| **Segmentation** | Polygons | `yolo26*-seg` |
| **Bounding Box** | Axis-aligned rectangles | `yolo26*` (detect) |
| **OBB** | Rotated rectangles | `yolo26*-obb` |

## Features

- **Task type switcher**: pick Segmentation, Bounding Box or OBB when you start. Menus, export and training follow the mode you choose.
- **Auto Label**: load a trained YOLO model and label the current image or a whole folder in one batch, with a confidence threshold.
- **Magic Label**: type the class names (e.g. `car seat, buckle, headrest`). Grounding DINO finds the objects and SAM2 turns each box into a precise polygon. You don't need a trained model. It runs on GPU if one is available and falls back to CPU.
- **Verified Empty** (`E`): mark an image as having no objects so it's exported as a background image.
- **Export YOLO**: writes a ready-to-train dataset (`images/`, `labels/`, `data.yaml`) with a train/val split slider. It merges classes with any existing dataset and updates images already in the dataset in place.
- **Train YOLO / Retrain YOLO**: train from the YOLO26 base models (n/s/m/l/x) or fine-tune existing weights. You get live loss and mAP charts, per-class metrics at the end, and an instant Cancel, because training runs in its own process.
- **Dataset tools**: Augment Dataset, Explore Dataset (thumbnail browser for YOLO datasets), and Dataset Stats / Annotation Statistics.
- **Dark theme** and a **standalone Windows `.exe`** built with PyInstaller.

## Requirements

- Windows 10/11 (the `.exe` build and DLL handling are Windows-specific; running from source should also work on Linux)
- **NVIDIA GPU with CUDA** for training. CPU training is disabled on purpose. Magic Label and Auto Label also work on CPU, just more slowly.
- Python 3.11 (developed with 3.11.13)

Tested with: `torch 2.5.1+cu121`, `torchvision 0.20.1`, `ultralytics 8.4.37`, `transformers 5.3.0`, `sam2 1.1.0`, `PyQt5 5.15.11`, `pyinstaller 6.16.0`.

## Setup (from source)

```bash
git clone https://github.com/ayhemmoussa/Labelyolo26.git
cd Labelyolo26

conda create -n labelme python=3.11 -y
conda activate labelme

# 1. PyTorch with CUDA (pick the index URL that matches your driver)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2. The app itself (editable install) + Magic Label dependencies
pip install -e ./labelme
pip install transformers sam2
```

Run it:

```bash
labelme            # or: python -m labelme
```

### Model weights (not stored in git)

Weights are too large for GitHub, so they're excluded by `.gitignore`.

- **YOLO26 base models** (`yolo26n-seg.pt`, `yolo26n-obb.pt`, …): Ultralytics downloads them automatically the first time you train. To bundle them into the `.exe`, put them in the repository root.
- **Magic Label models**: when these are missing, they're downloaded from Hugging Face on first use. To use them offline or bundle them into the `.exe`, download them into `labelme/labelme/models/`:

  ```bash
  hf download IDEA-Research/grounding-dino-tiny --local-dir labelme/labelme/models/grounding_dino_tiny
  mkdir -p labelme/labelme/models/sam2
  curl -L https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt \
       -o labelme/labelme/models/sam2/sam2_hiera_tiny.pt
  ```

## Typical workflow

1. **Choose the task type** (Segmentation / Bounding Box / OBB).
2. **Open Dir** with your images.
3. **Set Dataset Path**: the folder where the YOLO dataset and training runs are written.
4. **Annotate**: draw shapes by hand, or speed things up with **Magic Label** (text prompt) or **Auto Label** (trained model). Press `E` on images that have no objects.
5. **Export YOLO**: choose the train/val split.
6. **Train YOLO**: pick model size, epochs, batch size, image size and GPU.
7. Use the new `best.pt` with **Auto Label** on more images, fix the results, re-export, then **Retrain YOLO**.

The dataset folder ends up looking like this:

```
<dataset>/
├── data.yaml
├── .labels.json            # class list, kept in sync with the project folder
├── images/{train,val}/
├── labels/{train,val}/
└── runs/
    ├── train-seg/exp*/     # segmentation runs
    ├── train/exp*/         # bounding-box runs
    └── train-obb/exp*/     # OBB runs   (weights in exp*/weights/best.pt)
```

## Building the Windows executable

1. Download the Magic Label models into `labelme/labelme/models/` (see above). The build won't run without them.
2. Delete any leftover `labelme/labelme/models/grounding_dino_tiny/.cache/` folder so partial downloads don't get bundled.
3. Optionally, put the `yolo26*.pt` base weights in the repository root.
4. Build:

```bash
cd labelme
pip install pyinstaller
pyinstaller LabelMe-YOLO-NoONNX.spec
```

The output goes to `labelme/dist/labelyolo26/labelyolo26.exe`. Ship the whole `labelyolo26/` folder.

## Project structure

```
Labelyolo26/
├── labelme/                        # Python project (fork of labelme 5.8.3)
│   ├── labelme/
│   │   ├── __main__.py             # Entry point, dark theme, frameless dialogs
│   │   ├── app.py                  # Main window, menus, AI panel, batch labelling
│   │   ├── auto_label.py           # Trained-YOLO inference → labelme shapes
│   │   ├── magic_label.py          # Grounding DINO + SAM2 text-prompted labelling
│   │   ├── yolo_export*.py         # Export: segmentation / bbox / OBB
│   │   ├── yolo_train*.py          # Training dialogs: segmentation / bbox / OBB
│   │   ├── yolo_retrain*.py        # Fine-tuning from existing weights
│   │   ├── yolo_train_process.py   # Training subprocess entry points
│   │   ├── data_augmentation.py    # Dataset augmentation
│   │   ├── data_split.py           # Train/val split dialog
│   │   ├── explore_yolo.py         # Dataset thumbnail explorer
│   │   ├── stats_yolo.py           # Dataset statistics
│   │   ├── widgets/                # Canvas, label dialogs, AI prompt widget, …
│   │   └── models/                 # (git-ignored) bundled Magic Label weights
│   ├── LabelMe-YOLO-NoONNX.spec    # PyInstaller build spec
│   ├── rthook_torch_dlls.py        # PyInstaller runtime hook for torch DLLs
│   ├── pyproject.toml
│   ├── requirements-exe.txt
│   ├── tests/                      # Tests from the original labelme
│   └── examples/                   # Examples from the original labelme
└── yolo26*.pt                      # (git-ignored) base weights bundled into the .exe
```

## License and credits

- LabelYolo26 is based on **[labelme](https://github.com/wkentaro/labelme)** by Kentaro Wada and inherits its **GPL-3.0** license (see [`labelme/LICENSE`](labelme/LICENSE)).
- Training and inference use **[Ultralytics](https://github.com/ultralytics/ultralytics)** YOLO, which is **AGPL-3.0**. Commercial use may require an Ultralytics Enterprise license.
- Magic Label uses **[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny)** (IDEA Research) and **[SAM 2](https://github.com/facebookresearch/sam2)** (Meta).
