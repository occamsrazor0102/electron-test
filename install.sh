#!/usr/bin/env bash
# install.sh — Build and install llm-detector as a standalone executable.
#
# Usage:
#   ./install.sh                                  # build + install to ~/.local/bin
#   ./install.sh --prefix /usr/local              # install to /usr/local/bin
#   ./install.sh --install-dir /opt/llm-detector  # install executable + all components to one directory
#   ./install.sh --build-only                     # build without installing
#
set -euo pipefail

PREFIX="${HOME}/.local"
INSTALL_DIR_OVERRIDE=""
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)   PREFIX="$2"; shift 2 ;;
        --install-dir)  INSTALL_DIR_OVERRIDE="$2"; shift 2 ;;
        --build-only) BUILD_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--prefix DIR] [--install-dir DIR] [--build-only]"
            echo ""
            echo "Options:"
            echo "  --prefix DIR       Install to DIR/bin (default: ~/.local)"
            echo "  --install-dir DIR  Install executable and all components to DIR"
            echo "                     (centralised single-directory installation)"
            echo "  --build-only       Only build, do not install"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -n "${INSTALL_DIR_OVERRIDE}" ]]; then
    INSTALL_DIR="${INSTALL_DIR_OVERRIDE}"
else
    INSTALL_DIR="${PREFIX}/bin"
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Checking Python environment..."
python3 -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'" 2>/dev/null \
    || { echo "Error: Python 3.9 or later is required."; exit 1; }

echo "==> Installing build dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "${SCRIPT_DIR}[bundle]"

echo "==> Building single-file executable..."
cd "${SCRIPT_DIR}"
ONEFILE=1 pyinstaller llm_detector.spec --noconfirm --clean

EXE_PATH="${SCRIPT_DIR}/dist/llm-detector"
if [[ ! -f "${EXE_PATH}" ]]; then
    echo "Error: Build failed — executable not found at ${EXE_PATH}"
    exit 1
fi

echo "==> Build successful: ${EXE_PATH}"

if ${BUILD_ONLY}; then
    echo "Done (build only). Executable is at: ${EXE_PATH}"
    exit 0
fi

echo "==> Installing to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
cp -f "${EXE_PATH}" "${INSTALL_DIR}/llm-detector"
chmod +x "${INSTALL_DIR}/llm-detector"

# When using --install-dir, copy additional components for a centralised install
if [[ -n "${INSTALL_DIR_OVERRIDE}" ]]; then
    echo "==> Copying components to ${INSTALL_DIR}..."
    cp -f "${SCRIPT_DIR}/requirements.txt" "${INSTALL_DIR}/" 2>/dev/null || true
    cp -f "${SCRIPT_DIR}/pyproject.toml" "${INSTALL_DIR}/" 2>/dev/null || true
    cp -f "${SCRIPT_DIR}/README.md" "${INSTALL_DIR}/" 2>/dev/null || true
    mkdir -p "${INSTALL_DIR}/config"
    echo "Install directory: ${INSTALL_DIR}" > "${INSTALL_DIR}/config/install_info.txt"
    echo "Installed at: $(date -Iseconds)" >> "${INSTALL_DIR}/config/install_info.txt"
fi

# Check if install dir is on PATH
if ! echo "${PATH}" | tr ':' '\n' | grep -qx "${INSTALL_DIR}"; then
    echo ""
    echo "NOTE: ${INSTALL_DIR} is not on your PATH."
    echo "Add it by appending this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo ""
    echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
    echo ""
fi

echo "==> Installed successfully!"
echo "    Run with:  llm-detector --help"
