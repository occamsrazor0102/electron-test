# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for LLM Detector Pipeline.

Supports two build modes controlled by the ONEFILE environment variable:

  Directory bundle (default):
    pyinstaller llm_detector.spec

  Single-file executable:
    ONEFILE=1 pyinstaller llm_detector.spec          # Linux / macOS
    $env:ONEFILE=1; pyinstaller llm_detector.spec     # PowerShell
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
onefile = os.environ.get('ONEFILE', '0') == '1'

# Collect all submodules of the package
hiddenimports = collect_submodules('llm_detector')
datas = []

# Submodules that reference uninstalled optional packages or trigger
# deprecation warnings — exclude them from collection to keep the build
# log clean and the bundle lean.
_exclude_prefixes = [
    'sklearn.externals.array_api_compat.dask',      # requires dask
    'torch.utils.tensorboard',                       # requires tensorboard
    'torch.distributed._sharding_spec',              # deprecated alias
    'torch.distributed._sharded_tensor',             # deprecated alias
    'torch.distributed._shard.checkpoint',           # deprecated alias
    'streamlit.external.langchain',                   # requires langchain
    'catalogue._importlib_metadata',                  # requires zipp
]

def _filter_excludes(imports):
    """Remove hiddenimports that match any excluded prefix."""
    return [m for m in imports
            if not any(m == p or m.startswith(p + '.') for p in _exclude_prefixes)]

# Optional deps — include if installed, skip gracefully if not
for mod in ['anthropic', 'openai', 'pypdf', 'spacy', 'ftfy',
            'sentence_transformers', 'sklearn', 'transformers', 'torch',
            'streamlit']:
    try:
        __import__(mod)
        hiddenimports += _filter_excludes(collect_submodules(mod))
        datas += collect_data_files(mod)
    except ImportError:
        pass

# spaCy companion libraries — needed for the bundled sentencizer
for mod in ['thinc', 'blis', 'cymem', 'preshed', 'murmurhash',
            'srsly', 'catalogue', 'confection', 'weasel', 'wasabi',
            'langcodes', 'language_data']:
    try:
        __import__(mod)
        hiddenimports += _filter_excludes(collect_submodules(mod))
        datas += collect_data_files(mod)
    except ImportError:
        pass

a = Analysis(
    ['llm_detector/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_exclude_prefixes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if onefile:
    # ── Single-file executable ──────────────────────────────────────
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='llm-detector',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
    )
else:
    # ── Directory bundle (default) ──────────────────────────────────
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='llm-detector',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,   # windowed mode — GUI opens by default
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='llm-detector',
    )
