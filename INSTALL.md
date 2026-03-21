# Installing LLM Detector as a Standalone Executable

Three ways to get a standalone `llm-detector` binary — no Python needed at runtime.

---

## Option 1: Download a pre-built release (easiest)

1. Go to the [Releases](../../releases) page.
2. Download the binary for your platform:
   - `llm-detector-linux-x86_64`
   - `llm-detector-macos-x86_64`
   - `llm-detector-windows-x86_64.exe`
3. Make it executable (Linux/macOS):
   ```bash
   chmod +x llm-detector-linux-x86_64
   mv llm-detector-linux-x86_64 /usr/local/bin/llm-detector
   ```
4. Run it:
   ```bash
   llm-detector --help
   ```

---

## Option 2: One-command install script

The install scripts build from source and place the executable on your PATH.

### Linux / macOS

```bash
./install.sh
```

Options:

| Flag | Description |
|------|-------------|
| `--prefix DIR` | Install to `DIR/bin` (default: `~/.local`) |
| `--build-only` | Build the exe in `dist/` without installing |

### Windows (PowerShell)

```powershell
.\install.ps1
```

Options:

| Flag | Description |
|------|-------------|
| `-Prefix DIR` | Install to `DIR\bin` (default: `%LOCALAPPDATA%\llm-detector`) |
| `-BuildOnly` | Build the exe in `dist\` without installing |

The Windows script automatically adds the install directory to your user PATH.

---

## Option 3: Manual build with PyInstaller

Requires Python 3.9+.

### Single-file executable (recommended)

```bash
pip install ".[bundle]"
ONEFILE=1 pyinstaller llm_detector.spec --noconfirm --clean
# Output: dist/llm-detector  (or dist/llm-detector.exe on Windows)
```

On Windows PowerShell:
```powershell
pip install ".[bundle]"
$env:ONEFILE = "1"
pyinstaller llm_detector.spec --noconfirm --clean
# Output: dist\llm-detector.exe
```

### Directory bundle

Produces a folder with the exe and its dependencies — starts faster than single-file but is less portable.

```bash
pip install ".[bundle]"
pyinstaller llm_detector.spec --noconfirm --clean
# Output: dist/llm-detector/llm-detector
```

---

## What's included in the executable

The standalone exe bundles **all features** — built-in analyzers, lexicon packs, CLI, GUI, Streamlit web dashboard, and all optional NLP dependencies (spaCy, sentence-transformers, scikit-learn, ftfy, PyTorch/transformers, Anthropic/OpenAI API clients, and pypdf).

To build the **standard** exe (all features, recommended):
```bash
pip install ".[bundle]"
ONEFILE=1 pyinstaller llm_detector.spec --noconfirm --clean
```

To build a **minimal** exe (smallest size, core features only):
```bash
pip install pyinstaller pandas openpyxl
ONEFILE=1 pyinstaller llm_detector.spec --noconfirm --clean
```

---

## Verifying your installation

```bash
llm-detector --version
llm-detector --text "Analyze this sample text for LLM signals"
llm-detector --gui   # opens the desktop GUI
```
