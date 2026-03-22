# BEET Desktop — Electron Wrapper

Packages the BEET 2.0 React GUI and Python backend into a single installable
desktop application.  Users just double-click the app — no Python or Node.js
installation required.

## Quick start (development)

```bash
# 1. Install dependencies
cd beet/electron
npm install

# Also install GUI deps if not already done
cd ../gui && npm install && cd ../electron

# 2. Install the Python package (for the dev server)
pip install -e ../../

# 3. Run in development mode (Electron + React hot-reload + Python server)
npm run dev
```

The `dev` script starts three things in parallel:
- The React Vite dev server on `http://localhost:5173`
- TypeScript watch mode for the Electron main process
- Electron pointing at the Vite dev server

## Building a distributable installer

```bash
# Windows (requires Windows or a cross-compile environment)
npm run dist:win

# macOS (requires macOS for code-signing)
npm run dist:mac

# Linux AppImage
npm run dist:linux
```

The build pipeline:
1. Builds the React app (`beet/gui/`) with `vite build`.
2. Compiles the Electron main-process TypeScript.
3. Runs `electron-builder` to package everything into an installer.

The bundled Python executable (`llm-detector.exe` / `llm-detector`) is
expected at `../../dist_single/llm-detector[.exe]` — this is the output of
the existing PyInstaller pipeline in the repository root.

## Architecture

```
beet/electron/
├── src/
│   ├── main.ts       Main process — app lifecycle, server spawn, tray icon
│   └── preload.ts    Context bridge — exposes typed IPC API to the renderer
├── assets/
│   └── tray-icon.png System-tray icon (16×16 teal square)
├── package.json      electron-builder config
└── tsconfig.json
```

### How it works at runtime

1. **App launches** → Electron creates the `BrowserWindow` (hidden) and the
   system-tray icon (showing "Starting…").
2. **Python server starts** → the bundled `llm-detector` binary is spawned
   with `serve --port 8000 --host 127.0.0.1`.
3. **Health polling** → main process polls `GET /health` every 800 ms until
   the server responds 200.
4. **UI loads** → once healthy, the `BrowserWindow` loads the React app from
   `resources/renderer/index.html` (production) or `http://localhost:5173`
   (development).
5. **System tray** → shows server status; "Check Text from Clipboard" reads
   the clipboard and sends the text to the renderer via IPC.
6. **Quit** → `before-quit` event kills the Python server process.

### Auto-update

`electron-updater` checks GitHub Releases on startup.  A new release is
triggered by pushing a `v*` tag (handled by the existing CI workflow).

### IPC bridge (`window.beetElectron`)

The preload script exposes a `window.beetElectron` global to the renderer:

| Method | Description |
|--------|-------------|
| `getServerStatus()` | Returns `{ ready, apiBase }` |
| `getClipboardText()` | Reads the system clipboard |
| `writeClipboardText(text)` | Writes to the clipboard |
| `openExternal(url)` | Opens a URL in the default browser |
| `onCheckClipboardText(cb)` | Subscribes to tray "Check Text" events |
| `onUpdateAvailable(cb)` | Notified when a new version is found |
| `onUpdateDownloaded(cb)` | Notified when download completes |
