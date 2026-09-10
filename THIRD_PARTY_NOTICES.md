# Licensing and third-party notices

## LabelYolo26

LabelYolo26 is a modified version of [labelme](https://github.com/wkentaro/labelme) 5.8.3 by Kentaro Wada and contributors, which is licensed under the GNU General Public License v3.0.

- Modifications Copyright (C) 2026 Ayhem Moussa.
- The whole project, including the modifications, is licensed under **GPL-3.0-only**. See [LICENSE](LICENSE).
- Main changes from labelme 5.8.3, listed here as GPLv3 section 5(a) requires:
  - YOLO dataset export for segmentation, bounding boxes and OBB
  - YOLO training and retraining, and Auto Label with YOLO models
  - Magic Label (Grounding DINO + SAM 2)
  - rotated-box annotation
  - dataset tools (augmentation, explorer, statistics)
  - a dark theme and a PyInstaller Windows build

  The git history has the details.

## Main third-party components

| Component | Used for | License |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO training and inference | **AGPL-3.0** |
| YOLO26 pretrained weights (`yolo26*.pt`) | Base models for training | **AGPL-3.0** (Ultralytics) |
| [PyQt5](https://riverbankcomputing.com/software/pyqt/) | User interface | **GPL-3.0** |
| Qt 5 (`PyQt5-Qt5`) | User interface runtime | LGPL-3.0 |
| [PyTorch](https://pytorch.org) and torchvision | Deep-learning runtime | BSD-3-Clause. The bundled CUDA/cuDNN libraries are covered by the NVIDIA terms included in PyTorch's license file |
| [Transformers](https://github.com/huggingface/transformers) | Runs Grounding DINO | Apache-2.0 |
| [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) weights | Magic Label detection | Apache-2.0 |
| [SAM 2](https://github.com/facebookresearch/sam2) and `sam2_hiera_tiny.pt` | Magic Label masks | Apache-2.0 |
| [OpenCV](https://opencv.org) | Image processing | Apache-2.0 |
| NumPy, SciPy, pandas, scikit-image, Pillow, matplotlib, Hydra, OmegaConf, loguru, PyYAML and others | Various | BSD, MIT or PSF-style permissive licenses |

The Windows build includes `THIRD_PARTY_LICENSES.txt`, which has the full license text of every bundled package. `labelme/tools/gen_third_party_licenses.py` generates it during each build.

## What this means in practice

This is a summary, not legal advice. Check with your legal team before giving the app to anyone outside your organisation.

- **Using it inside your organisation.** Running the app and sharing it within the same company is generally not considered "distribution" under the GPL or AGPL, so the source-sharing obligations below don't apply.
- **Giving the `.exe` to anyone outside the organisation** (customers, partners, the public). You must:
  - ship it with `LICENSE`, `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES.txt` (the build puts these next to the exe automatically), and
  - provide the complete corresponding source code under GPL-3.0 / AGPL-3.0, or a written offer to provide it. The source is this repository at the matching commit.
- **Ultralytics (AGPL-3.0).** Using Ultralytics in a closed-source or commercial product without releasing its source requires an [Ultralytics Enterprise License](https://www.ultralytics.com/license). Ultralytics' stated position is that this also applies to models trained with its software.
- **PyQt5 (GPL-3.0).** A closed-source version of the app would need a commercial PyQt license from Riverbank Computing.

## Trademarks

Ultralytics and the Ultralytics logo are trademarks of Ultralytics Inc. LabelYolo26 is an independent project. It is not affiliated with or endorsed by Ultralytics, IDEA Research, Meta or the labelme authors.
