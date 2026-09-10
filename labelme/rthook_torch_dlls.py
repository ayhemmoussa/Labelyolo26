# Runtime hook: register all torch DLL directories with the Windows OS loader
# BEFORE any Python import runs. This must happen before torch/__init__.py
# calls _load_dll_libraries(), otherwise c10.dll can't find its own deps.
#
# Why both collect_all AND this hook are needed:
#   collect_all  → puts the DLL files into the bundle
#   this hook    → tells the OS *where to search* for those DLLs at load time
#                  (Python 3.8+ no longer uses PATH for DLL resolution by default)

import os
import sys

# Fix for PyInstaller console=False: stdout/stderr are None in every process
# (main window AND every spawned training worker). Redirect to devnull so any
# print() or write() in ultralytics/torch during training doesn't crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

if sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
    _base = sys._MEIPASS  # …\dist\LabelYolo\_internal

    _candidate_dirs = [
        _base,
        os.path.join(_base, "torch", "lib"),
        os.path.join(_base, "torch", "bin"),
        os.path.join(_base, "torch"),
    ]

    for _d in _candidate_dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

    # Also pre-load c10.dll explicitly so its deps are resolved
    # from the directories we just registered, before torch.__init__ tries.
    import ctypes
    _c10 = os.path.join(_base, "torch", "lib", "c10.dll")
    if os.path.isfile(_c10):
        try:
            ctypes.CDLL(_c10)
        except OSError:
            pass
