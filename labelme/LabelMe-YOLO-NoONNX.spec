# -*- mode: python ; coding: utf-8 -*-

import sys
import os
import cv2
import sam2
import transformers
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# collect_all pulls in every DLL, data file, and hidden import for a package —
# this is the correct way to bundle PyTorch and avoid WinError 1114 on c10.dll.
torch_datas,       torch_binaries,       torch_hiddenimports       = collect_all('torch')
torchvision_datas, torchvision_binaries, torchvision_hiddenimports = collect_all('torchvision')

# Collect ALL labelme Python modules explicitly.
# Without this, modules only imported inside functions (lazy imports) or
# modules listed only in hiddenimports are sometimes missed in the frozen archive.
labelme_datas, labelme_binaries, labelme_hiddenimports = collect_all('labelme')

# Paths still needed for other packages
cv2_path = os.path.dirname(cv2.__file__)
sam2_path = os.path.dirname(sam2.__file__)
transformers_path = os.path.dirname(transformers.__file__)

# Define the main analysis for labelme
a = Analysis(
    ['labelme/__main__.py'],
    pathex=['.'],
    binaries=(
        torch_binaries + torchvision_binaries + labelme_binaries +
        # OpenCV binaries
        ([(os.path.join(cv2_path, 'opencv_videoio_ffmpeg4120_64.dll'), '.')]
         if os.path.exists(os.path.join(cv2_path, 'opencv_videoio_ffmpeg4120_64.dll'))
         else [])
    ),
    datas=(
        torch_datas + torchvision_datas + labelme_datas +
        [
        # Labelme data files
        ('labelme/icons', 'labelme/icons'),
        ('labelme/translate', 'labelme/translate'),
        ('labelme/config', 'labelme/config'),

        # Magic Label bundled models (Grounding DINO + SAM2 checkpoint)
        ('labelme/models/grounding_dino_tiny', 'labelme/models/grounding_dino_tiny'),
        ('labelme/models/sam2',                'labelme/models/sam2'),

        # SAM2 hydra configs (required at runtime by build_sam2)
        (os.path.join(sam2_path, 'configs'), 'sam2/configs'),

        # Transformers tokenizer / model data (sentencepiece, vocab files, etc.)
        (os.path.join(transformers_path, 'models', 'auto'),           'transformers/models/auto'),
        (os.path.join(transformers_path, 'models', 'grounding_dino'), 'transformers/models/grounding_dino'),

        # YOLO26 base weights — bundle whatever exists; l/x auto-download at runtime
        *[(f'../yolo26{s}.pt', '.')     for s in ('n','s','m','l','x')
          if os.path.exists(f'../yolo26{s}.pt')],
        *[(f'../yolo26{s}-seg.pt', '.') for s in ('n','s','m','l','x')
          if os.path.exists(f'../yolo26{s}-seg.pt')],
        *[(f'../yolo26{s}-obb.pt', '.') for s in ('n','s','m','l','x')
          if os.path.exists(f'../yolo26{s}-obb.pt')],

        # Configuration files
        ('labelme/data.yaml', 'labelme'),
        ]
    ),
    hiddenimports=(torch_hiddenimports + torchvision_hiddenimports + labelme_hiddenimports + [
        # Core PyQt5
        'PyQt5.QtCore',
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'PyQt5.QtPrintSupport',
        'PyQt5.QtChart',  # For training progress charts
        
        # Labelme modules - COMPLETE LIST
        'labelme',
        'labelme.app',
        'labelme.widgets',
        'labelme.widgets.canvas',
        'labelme.widgets.label_dialog',
        'labelme.widgets.label_list_widget',
        'labelme.widgets.file_dialog_preview',
        'labelme.widgets.brightness_contrast_dialog',
        'labelme.widgets.tool_bar',
        'labelme.widgets.unique_label_qlist_widget',
        'labelme.widgets.zoom_widget',
        'labelme.widgets.ai_prompt_widget',
        'labelme.widgets.escapable_qlist_widget',
        'labelme.utils',
        'labelme.config',
        'labelme.translate',
        'labelme.cli',
        'labelme.shape',
        'labelme.label_file',
        'labelme.auto_label',
        'labelme.data_split',
        'labelme.data_augmentation',
        'labelme.stats_yolo',
        'labelme.yolo_export',
        'labelme.yolo_export_bbx',
        'labelme.yolo_export_obb',
        'labelme.yolo_export_common',
        'labelme.yolo_train',
        'labelme.yolo_train_bbx',
        'labelme.yolo_train_obb',
        'labelme.yolo_retrain',
        'labelme.yolo_retrain_bbx',
        'labelme.yolo_retrain_obb',
        'labelme.yolo_train_process',
        'labelme.explore_yolo',
        'labelme._automation',
        'labelme.magic_label',

        # ── Magic Label: Grounding DINO (transformers) ──────────────────────
        'transformers',
        'transformers.models.grounding_dino',
        'transformers.models.grounding_dino.modeling_grounding_dino',
        'transformers.models.grounding_dino.processing_grounding_dino',
        'transformers.models.grounding_dino.image_processing_grounding_dino',
        'transformers.models.grounding_dino.image_processing_grounding_dino_fast',
        'transformers.models.auto',
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.processing_auto',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_fast',
        'transformers.modeling_utils',
        'transformers.feature_extraction_utils',
        'transformers.image_processing_utils',
        'transformers.image_transforms',
        'transformers.utils',
        'transformers.utils.hub',
        'transformers.file_utils',
        'tokenizers',
        'sentencepiece',
        'huggingface_hub',
        'huggingface_hub.file_download',
        'safetensors',
        'safetensors.torch',

        # ── Magic Label: SAM2 ───────────────────────────────────────────────
        'sam2',
        'sam2.sam2_image_predictor',
        'sam2.build_sam',
        'sam2.modeling',
        'sam2.modeling.sam2_base',
        'sam2.modeling.backbones',
        'sam2.modeling.backbones.hieradet',
        'sam2.modeling.backbones.image_encoder',
        'sam2.modeling.memory_attention',
        'sam2.modeling.memory_encoder',
        'sam2.modeling.position_encoding',
        'sam2.modeling.sam',
        'sam2.modeling.sam.mask_decoder',
        'sam2.modeling.sam.prompt_encoder',
        'sam2.modeling.sam.transformer',
        'sam2.utils',
        'sam2.utils.misc',
        'sam2.utils.transforms',

        # ── SAM2 / hydra config system ──────────────────────────────────────
        'hydra',
        'hydra._internal',
        'hydra._internal.utils',
        'hydra.core',
        'hydra.core.global_hydra',
        'hydra.core.plugins',
        'hydra.core.utils',
        'hydra.types',
        'omegaconf',
        'omegaconf._utils',
        'antlr4',
        'iopath',
        'iopath.common',
        'iopath.common.file_io',

        # Ultralytics (YOLO26 uses ultralytics package)
        'ultralytics',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.models.yolo.segment',
        'ultralytics.models.yolo.obb',
        'ultralytics.nn',
        'ultralytics.nn.tasks',
        'ultralytics.nn.modules',
        'ultralytics.utils',
        'ultralytics.utils.ops',
        'ultralytics.utils.metrics',
        'ultralytics.utils.plotting',
        'ultralytics.utils.torch_utils',
        'ultralytics.utils.checks',
        'ultralytics.utils.downloads',
        'ultralytics.utils.loss',
        'ultralytics.utils.callbacks',
        'ultralytics.data',
        'ultralytics.data.dataset',
        'ultralytics.data.loaders',
        'ultralytics.data.augment',
        'ultralytics.cfg',
        'ultralytics.engine',
        'ultralytics.engine.model',
        'ultralytics.engine.trainer',
        'ultralytics.engine.validator',
        'ultralytics.engine.predictor',
        'ultralytics.engine.results',
        'thop',
        'polars',
        
        # Image processing
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        'cv2',
        'imgviz',
        
        # Data science
        'numpy',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.patches',
        'matplotlib.backends',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
        'matplotlib.colors',
        'seaborn',
        'pandas',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
        'scipy.ndimage',
        'scipy.interpolate',
        'scipy.optimize',
        'sklearn',
        'sklearn.metrics',
        'imgviz',
        
        # Additional common dependencies
        'pkg_resources',
        'setuptools',
        'distutils',
        'six',
        'dateutil',
        'pytz',
        'certifi',
        'charset_normalizer',
        'idna',
        'urllib3',
        
        # YAML and JSON
        'yaml',
        'json',
        'configparser',
        
        # Logging and utilities
        'loguru',
        'termcolor',
        'tqdm',
        'requests',
        'pathlib',
        'glob',
        'shutil',
        'platform',
        'psutil',
        
        # Git integration (YOLOv5 checks for git info)
        'git',
        'git.repo',
        'git.config',
        'git.util',
        'gitdb',
        'gitdb.db',
        'smmap',
        
        # Python standard library modules needed by PyTorch
        'pdb',
        'trace',
        'bdb',
        'cmd',
        'code',
        'codeop',
        'linecache',
        'inspect',
        'dis',
        'opcode',
        'keyword',
        'token',
        'tokenize',
        'ast',
        'types',
        'typing',
        'collections',
        'collections.abc',
        'functools',
        'operator',
        'itertools',
        'weakref',
        'gc',
        'sys',
        'os',
        'io',
        'contextlib',
        'warnings',
        'unittest',
        'unittest.mock',
        'unittest.case',
        'unittest.suite',
        'unittest.loader',
        'unittest.runner',
        'test',
        
        # Logging modules
        'logging',
        'logging.config',
        'logging.handlers',
        
        # Additional standard library modules often needed
        'argparse',
        'configparser',
        'datetime',
        'decimal',
        'email',
        'email.mime',
        'email.mime.text',
        'email.mime.multipart',
        'email.utils',
        'hashlib',
        'hmac',
        'html',
        'html.parser',
        'http',
        'http.client',
        'http.server',
        'importlib',
        'importlib.util',
        'json',
        'math',
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.dummy',
        'pickle',
        'platform',
        'random',
        're',
        'shutil',
        'socket',
        'sqlite3',
        'string',
        'subprocess',
        'tempfile',
        'threading',
        'time',
        'urllib',
        'urllib.parse',
        'urllib.request',
        'urllib.error',
        'uuid',
        'xml',
        'xml.etree',
        'xml.etree.ElementTree',
        'zipfile',
        
        # File system and path modules
        'pathlib',
        'glob',
        'fnmatch',
        'os.path',
        'stat',
        
        # Data structures and algorithms
        'heapq',
        'bisect',
        'array',
        'copy',
        'deepcopy',
        
        # Network and internet modules
        'base64',
        'binascii',
        'zlib',
        'gzip',
        'tarfile',
        
        # Concurrency
        'concurrent',
        'concurrent.futures',
        'asyncio',
        
        # Additional for data augmentation
        'random',
        'pathlib',
        
        # Color space conversions
        'colorsys',
        
        # Exclude ONNX and SAM related modules
        # 'onnx',
        # 'onnxruntime',
        # 'osam',
    ]),  # end of hiddenimports tuple
    hookspath=[],
    runtime_hooks=['rthook_torch_dlls.py'],
    excludes=[
        # Exclude ONNX and related
        'onnx',
        'onnxruntime',
        'osam',
        'onnxruntime-tools',
        
    # Exclude development tools
    'pytest',
    # 'unittest',  # Don't exclude - PyTorch needs unittest.mock
    # 'test',      # Don't exclude - may be needed by torch
    'tests',
    'doctest',
    # 'pydoc',  # keep pydoc available; some libs import it at runtime
    'tkinter',
    'turtle',
        
        # Exclude unused ML frameworks
        'tensorflow',
        'keras',
        'jax',
        'flax',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],  # Remove binaries, zipfiles, datas from here
    exclude_binaries=True,
    name='labelyolo26',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Console disabled for production
    icon='labelme/icons/icon2.ico' if os.path.exists('labelme/icons/icon2.ico') else None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='labelyolo26'
)