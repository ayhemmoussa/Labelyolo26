"""
Subprocess entry points for YOLO26 training.

These functions run inside a separate OS process spawned by the training workers.
Keeping them at module level makes them picklable for multiprocessing on Windows.

Communication with the parent process is done entirely through a multiprocessing.Queue:
  ('status',  message)              -> status bar text
  ('progress', epoch, result_dict)  -> per-epoch metrics
  ('done',    best_path, last_path, trainer_metrics_dict)
  ('error',   message, traceback)
"""

import os


def _resolve_device(params, progress_queue):
    """
    Resolve the training device.  Always targets a CUDA GPU.
    Returns the device string to pass to model.train(), or None if GPU is
    unavailable (in which case an 'error' message is put on progress_queue).
    """
    import torch

    requested = (params.get('device', '') or 'auto').strip()

    if requested == 'cpu':
        progress_queue.put(('error',
            "CPU training is disabled.\n\n"
            "This application requires an NVIDIA GPU with CUDA support.\n"
            "Please select a GPU device (0, 1, …) in training settings.",
            ''))
        return None

    if not torch.cuda.is_available():
        progress_queue.put(('error',
            "No CUDA GPU detected.\n\n"
            "Training requires an NVIDIA GPU with CUDA drivers installed.\n\n"
            "Possible causes:\n"
            "• No NVIDIA GPU in this machine\n"
            "• NVIDIA drivers not installed or outdated\n"
            "• CUDA toolkit version mismatch\n\n"
            f"torch.version.cuda = {torch.version.cuda}\n"
            f"torch.__version__  = {torch.__version__}",
            ''))
        return None

    if requested in ('auto', ''):
        device = '0'   # first GPU
    else:
        device = requested   # explicit index like '0', '1', …

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Allow TF32 on Ampere+ GPUs for faster matmul and convolutions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(int(device))

    # Pascal GPUs (compute capability < 7.0) have unstable FP16/AMP support
    cc = torch.cuda.get_device_capability(int(device))
    params['_use_amp'] = (cc[0] >= 7)

    return device


def _attach_per_class_metrics(tm, model):
    """Extract per-class P/R/mAP50/mAP from the trainer's final validator and
    store them in tm['_per_class'] as a list of dicts.  Safe to call even if
    the validator attributes are not present (older ultralytics versions).

    For segmentation models the validator stores .box and .seg sub-metrics;
    for detection/OBB the metrics are on the root object directly.
    """
    try:
        validator = getattr(model.trainer, 'validator', None)
        vm = getattr(validator, 'metrics', None) if validator else None
        if vm is None:
            return
        # Segmentation: metrics are split into .box and .seg sub-objects
        vm_box = getattr(vm, 'box', vm)
        vm_seg = getattr(vm, 'seg', None)

        ap_idx = getattr(vm_box, 'ap_class_index', None)
        if ap_idx is None:
            return
        per_class = []
        for i, cls_idx in enumerate(ap_idx):
            cls_name = model.names.get(int(cls_idx), str(cls_idx))
            entry = {
                'name':   cls_name,
                'P':      float(vm_box.p[i])    if hasattr(vm_box, 'p')    and i < len(vm_box.p)    else 0.0,
                'R':      float(vm_box.r[i])    if hasattr(vm_box, 'r')    and i < len(vm_box.r)    else 0.0,
                'mAP50':  float(vm_box.ap50[i]) if hasattr(vm_box, 'ap50') and i < len(vm_box.ap50) else 0.0,
                'mAP':    float(vm_box.ap[i])   if hasattr(vm_box, 'ap')   and i < len(vm_box.ap)   else 0.0,
            }
            if vm_seg is not None:
                entry['mask_P']     = float(vm_seg.p[i])    if hasattr(vm_seg, 'p')    and i < len(vm_seg.p)    else 0.0
                entry['mask_R']     = float(vm_seg.r[i])    if hasattr(vm_seg, 'r')    and i < len(vm_seg.r)    else 0.0
                entry['mask_mAP50'] = float(vm_seg.ap50[i]) if hasattr(vm_seg, 'ap50') and i < len(vm_seg.ap50) else 0.0
                entry['mask_mAP']   = float(vm_seg.ap[i])   if hasattr(vm_seg, 'ap')   and i < len(vm_seg.ap)   else 0.0
            per_class.append(entry)
        tm['_per_class'] = per_class
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation  (train from scratch  OR  retrain from existing .pt)
# ─────────────────────────────────────────────────────────────────────────────

def train_segmentation(data_yaml, params, progress_queue, initial_weights=None):
    """
    Run YOLO26 segmentation training inside a subprocess.

    Args:
        data_yaml:       Path to data.yaml
        params:          Dict with keys: model, epochs, batch_size, imgsz
        progress_queue:  multiprocessing.Queue for sending updates to parent
        initial_weights: None  → fresh training using params['model']
                         str   → path to existing .pt for retraining
    """
    try:
        from ultralytics import YOLO

        device = _resolve_device(params, progress_queue)
        if device is None:
            return  # error already queued

        if initial_weights:
            progress_queue.put(('status', f"Loading weights: {os.path.basename(initial_weights)}"))
            model = YOLO(initial_weights)
        else:
            model_name = params['model']
            if not model_name.endswith('.pt'):
                model_name += '.pt'
            if not os.path.exists(model_name):
                progress_queue.put(('status', f"Downloading model {model_name} (this may take a minute)…"))
            else:
                progress_queue.put(('status', f"Loading model: {model_name}"))
            model = YOLO(model_name)

        project_dir = os.path.join(os.path.dirname(data_yaml), 'runs/train-seg')
        total_epochs = params['epochs']
        _q = progress_queue

        def on_fit_epoch_end(trainer):
            epoch = trainer.epoch
            m = trainer.metrics or {}
            result = {
                'epoch':       epoch,
                'box_loss':    float(m.get('val/box_loss', 0)),
                'seg_loss':    float(m.get('val/seg_loss', 0)),
                'dfl_loss':    float(m.get('val/dfl_loss', 0)),
                'cls_loss':    float(m.get('val/cls_loss', 0)),
                'box_map50':   float(m.get('metrics/mAP50(B)', 0)),
                'box_map5095': float(m.get('metrics/mAP50-95(B)', 0)),
                'mask_map50':  float(m.get('metrics/mAP50(M)', 0)),
                'mask_map5095':float(m.get('metrics/mAP50-95(M)', 0)),
                '_trainer_metrics': dict(m),
            }
            _q.put(('progress', epoch, result))
            _q.put(('status', f"Epoch {epoch + 1}/{total_epochs}"))

        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

        use_amp = params.get('_use_amp', True)
        progress_queue.put(('status', f"Training YOLO26 segmentation on GPU:{device} — {total_epochs} epochs {'(AMP)' if use_amp else '(FP32)'}"))

        model.train(
            data=data_yaml,
            epochs=total_epochs,
            batch=params['batch_size'],
            imgsz=params['imgsz'],
            lr0=params.get('lr0', 0.01),
            device=device,
            project=project_dir,
            name='exp',
            workers=0,
            cache=True,
            exist_ok=False,
            patience=min(50, total_epochs),
            optimizer='SGD',
            cos_lr=True,
            close_mosaic=max(10, total_epochs // 10),
            verbose=True,
            amp=use_amp,
        )

        best = str(model.trainer.best)
        last = str(model.trainer.last)
        tm = dict(model.trainer.metrics or {})
        _attach_per_class_metrics(tm, model)
        progress_queue.put(('done', best, last, tm))

    except Exception as e:
        import traceback
        progress_queue.put(('error', str(e), traceback.format_exc()))


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box detection  (train from scratch  OR  retrain)
# ─────────────────────────────────────────────────────────────────────────────

def train_detection(data_yaml, params, progress_queue, initial_weights=None):
    """
    Run YOLO26 bounding-box detection training inside a subprocess.

    Args identical to train_segmentation; no initial_weights → fresh training.
    """
    try:
        from ultralytics import YOLO

        device = _resolve_device(params, progress_queue)
        if device is None:
            return

        if initial_weights:
            progress_queue.put(('status', f"Loading weights: {os.path.basename(initial_weights)}"))
            model = YOLO(initial_weights)
        else:
            model_name = params['model']
            if not model_name.endswith('.pt'):
                model_name += '.pt'
            if not os.path.exists(model_name):
                progress_queue.put(('status', f"Downloading model {model_name} (this may take a minute)…"))
            else:
                progress_queue.put(('status', f"Loading model: {model_name}"))
            model = YOLO(model_name)

        project_dir = os.path.join(os.path.dirname(data_yaml), 'runs/train')
        total_epochs = params['epochs']
        _q = progress_queue

        def on_fit_epoch_end(trainer):
            epoch = trainer.epoch
            m = trainer.metrics or {}
            result = {
                'epoch':    epoch,
                'box_loss': float(m.get('val/box_loss', 0)),
                'dfl_loss': float(m.get('val/dfl_loss', 0)),
                'cls_loss': float(m.get('val/cls_loss', 0)),
                'P':        float(m.get('metrics/precision(B)', 0)),
                'R':        float(m.get('metrics/recall(B)', 0)),
                'mAP50':    float(m.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(m.get('metrics/mAP50-95(B)', 0)),
                '_trainer_metrics': dict(m),
            }
            _q.put(('progress', epoch, result))
            _q.put(('status', f"Epoch {epoch + 1}/{total_epochs}"))

        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

        use_amp = params.get('_use_amp', True)
        progress_queue.put(('status', f"Training YOLO26 detection on GPU:{device} — {total_epochs} epochs {'(AMP)' if use_amp else '(FP32)'}"))

        model.train(
            data=data_yaml,
            epochs=total_epochs,
            batch=params['batch_size'],
            imgsz=params['imgsz'],
            lr0=params.get('lr0', 0.01),
            device=device,
            project=project_dir,
            name='exp',
            workers=0,
            cache=True,
            exist_ok=False,
            patience=min(50, total_epochs),
            optimizer='SGD',
            cos_lr=True,
            close_mosaic=max(10, total_epochs // 10),
            verbose=True,
            amp=use_amp,
        )

        best = str(model.trainer.best)
        last = str(model.trainer.last)
        tm = dict(model.trainer.metrics or {})
        _attach_per_class_metrics(tm, model)
        progress_queue.put(('done', best, last, tm))

    except Exception as e:
        import traceback
        progress_queue.put(('error', str(e), traceback.format_exc()))


# ─────────────────────────────────────────────────────────────────────────────
# Oriented Bounding Box  (train from scratch  OR  retrain)
# ─────────────────────────────────────────────────────────────────────────────

def train_obb(data_yaml, params, progress_queue, initial_weights=None):
    """
    Run YOLO26 OBB training inside a subprocess.

    Args identical to train_detection; no initial_weights → fresh training.
    """
    try:
        from ultralytics import YOLO

        device = _resolve_device(params, progress_queue)
        if device is None:
            return

        if initial_weights:
            progress_queue.put(('status', f"Loading weights: {os.path.basename(initial_weights)}"))
            model = YOLO(initial_weights)
        else:
            model_name = params['model']
            if not model_name.endswith('.pt'):
                model_name += '.pt'
            if not os.path.exists(model_name):
                progress_queue.put(('status', f"Downloading model {model_name} (this may take a minute)…"))
            else:
                progress_queue.put(('status', f"Loading model: {model_name}"))
            model = YOLO(model_name)

        project_dir = os.path.join(os.path.dirname(data_yaml), 'runs/train-obb')
        total_epochs = params['epochs']
        _q = progress_queue

        def on_fit_epoch_end(trainer):
            epoch = trainer.epoch
            m = trainer.metrics or {}
            result = {
                'epoch':    epoch,
                'box_loss': float(m.get('val/box_loss', 0)),
                'dfl_loss': float(m.get('val/dfl_loss', 0)),
                'cls_loss': float(m.get('val/cls_loss', 0)),
                'P':        float(m.get('metrics/precision(B)', 0)),
                'R':        float(m.get('metrics/recall(B)', 0)),
                'mAP50':    float(m.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(m.get('metrics/mAP50-95(B)', 0)),
                '_trainer_metrics': dict(m),
            }
            _q.put(('progress', epoch, result))
            _q.put(('status', f"Epoch {epoch + 1}/{total_epochs}"))

        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

        use_amp = params.get('_use_amp', True)
        progress_queue.put(('status', f"Training YOLO26 OBB on GPU:{device} — {total_epochs} epochs {'(AMP)' if use_amp else '(FP32)'}"))

        model.train(
            data=data_yaml,
            epochs=total_epochs,
            batch=params['batch_size'],
            imgsz=params['imgsz'],
            lr0=params.get('lr0', 0.01),
            device=device,
            project=project_dir,
            name='exp',
            workers=0,
            cache=True,
            exist_ok=False,
            patience=min(50, total_epochs),
            optimizer='SGD',
            cos_lr=True,
            close_mosaic=max(10, total_epochs // 10),
            verbose=True,
            amp=use_amp,
        )

        best = str(model.trainer.best)
        last = str(model.trainer.last)
        tm = dict(model.trainer.metrics or {})
        _attach_per_class_metrics(tm, model)
        progress_queue.put(('done', best, last, tm))

    except Exception as e:
        import traceback
        progress_queue.put(('error', str(e), traceback.format_exc()))
