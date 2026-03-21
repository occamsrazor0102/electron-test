/**
 * BEET 2.0 Desktop — Electron main process
 *
 * Responsibilities:
 *  1. Spawn the bundled Python BEET API server (llm-detector-serve).
 *  2. Poll until the server is healthy, then open the BrowserWindow.
 *  3. Manage a system-tray icon with server status and a "Check Text" action.
 *  4. Gracefully stop the Python process when the app quits.
 *  5. Check for updates via electron-updater on startup.
 */

import {
  app,
  BrowserWindow,
  clipboard,
  ipcMain,
  Menu,
  nativeImage,
  shell,
  Tray,
} from "electron";
import { autoUpdater } from "electron-updater";
import { ChildProcess, spawn } from "child_process";
import * as fs from "fs";
import * as http from "http";
import * as path from "path";

// ─── Constants ────────────────────────────────────────────────────────────────

const API_PORT = 8000;
const API_HOST = "127.0.0.1";
const API_BASE = `http://${API_HOST}:${API_PORT}`;

const IS_DEV = !app.isPackaged;
const RENDERER_DEV_URL = "http://localhost:5173";
const RENDERER_PROD = path.join(process.resourcesPath, "renderer", "index.html");

// In development, the Python server binary is resolved from PATH / virtual-env.
// In production (packaged), we look in resources/bin/.
function getServerBinaryPath(): string {
  if (IS_DEV) {
    // In development, use `llm-detector-serve` installed into the active
    // Python environment and available on PATH.  The OS resolves .exe on
    // Windows automatically, so we don't need to add the extension here.
    return "llm-detector-serve";
  }
  const ext = process.platform === "win32" ? ".exe" : "";
  return path.join(process.resourcesPath, "bin", `llm-detector${ext}`);
}

// ─── State ────────────────────────────────────────────────────────────────────

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let serverProcess: ChildProcess | null = null;
let serverReady = false;

// ─── Tray icon ────────────────────────────────────────────────────────────────

const TRAY_ICON_PATH = path.join(__dirname, "..", "assets", "tray-icon.png");

function getTrayIcon(status: "starting" | "ready" | "error"): Electron.NativeImage {
  if (fs.existsSync(TRAY_ICON_PATH)) {
    const img = nativeImage.createFromPath(TRAY_ICON_PATH);
    // On macOS, mark as template so the OS handles dark/light mode
    if (process.platform === "darwin") img.setTemplateImage(true);
    return img;
  }
  // Fallback: create a 16×16 solid-color PNG from raw bytes
  const color =
    status === "ready"
      ? Buffer.from([0x0d, 0x73, 0x77]) // teal
      : status === "error"
        ? Buffer.from([0xdc, 0x26, 0x26]) // red
        : Buffer.from([0xd9, 0x77, 0x06]); // amber

  const row = Buffer.concat([Buffer.from([0]), ...Array(16).fill(color)]);
  const rows = Buffer.concat(Array(16).fill(row));
  return nativeImage.createFromBuffer(rows);
}

function buildTrayMenu(): Electron.Menu {
  const statusLabel = serverReady
    ? "● Server running"
    : "○ Server starting…";

  return Menu.buildFromTemplate([
    { label: "BEET — AI Text Detection", enabled: false },
    { label: statusLabel, enabled: false },
    { type: "separator" },
    {
      label: "Open BEET",
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        } else {
          createMainWindow();
        }
      },
    },
    {
      label: "Check Text from Clipboard",
      accelerator: "CmdOrCtrl+Shift+B",
      click: () => {
        const text = clipboard.readText();
        if (!text.trim()) return;
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
          mainWindow.webContents.send("check-clipboard-text", text);
        } else {
          createMainWindow(() => {
            mainWindow?.webContents.send("check-clipboard-text", text);
          });
        }
      },
    },
    { type: "separator" },
    {
      label: "Quit BEET",
      role: "quit",
    },
  ]);
}

function updateTray() {
  if (!tray) return;
  const status = serverReady ? "ready" : "starting";
  tray.setImage(getTrayIcon(status));
  tray.setToolTip(serverReady ? "BEET — Server ready" : "BEET — Starting server…");
  tray.setContextMenu(buildTrayMenu());
}

// ─── Python server management ─────────────────────────────────────────────────

function startServer(): void {
  const binPath = getServerBinaryPath();

  // Dev mode: binary is `llm-detector-serve` (separate entry point, no subcommand)
  // Prod mode: binary is `llm-detector` (monolithic binary, needs `serve` subcommand)
  const spawnArgs: string[] = IS_DEV
    ? ["--port", String(API_PORT), "--host", API_HOST]
    : ["serve", "--port", String(API_PORT), "--host", API_HOST];

  console.log(`[BEET] Starting server: ${binPath} ${spawnArgs.join(" ")}`);

  serverProcess = spawn(binPath, spawnArgs, {
    env: { ...process.env },
    stdio: ["ignore", "pipe", "pipe"],
    // Windows: don't open a console window
    windowsHide: true,
  });

  serverProcess.stdout?.on("data", (d: Buffer) => {
    const line = d.toString().trim();
    if (line) console.log(`[server] ${line}`);
    if (line.includes("listening")) {
      // Fast path: server told us it's up
      onServerReady();
    }
  });

  serverProcess.stderr?.on("data", (d: Buffer) => {
    const line = d.toString().trim();
    if (line) console.error(`[server err] ${line}`);
  });

  serverProcess.on("exit", (code) => {
    console.log(`[BEET] Server exited with code ${code}`);
    serverReady = false;
    updateTray();
  });

  // Start polling for readiness
  pollServerHealth(30);
}

function pollServerHealth(retriesLeft: number): void {
  if (retriesLeft <= 0) {
    console.error("[BEET] Server did not become healthy in time");
    return;
  }

  const req = http.get(`${API_BASE}/health`, (res) => {
    if (res.statusCode === 200) {
      onServerReady();
    } else {
      setTimeout(() => pollServerHealth(retriesLeft - 1), 800);
    }
  });

  req.on("error", () => {
    setTimeout(() => pollServerHealth(retriesLeft - 1), 800);
  });

  req.setTimeout(600, () => {
    req.destroy();
    setTimeout(() => pollServerHealth(retriesLeft - 1), 800);
  });
}

function onServerReady(): void {
  if (serverReady) return; // idempotent
  serverReady = true;
  console.log("[BEET] Server is ready");
  updateTray();
  // Load the app UI once server is healthy
  mainWindow?.loadURL(IS_DEV ? RENDERER_DEV_URL : `file://${RENDERER_PROD}`);
  mainWindow?.show();
}

function stopServer(): void {
  if (!serverProcess) return;
  console.log("[BEET] Stopping server…");
  if (process.platform === "win32") {
    // Windows: use taskkill to ensure children are also killed
    spawn("taskkill", ["/pid", String(serverProcess.pid), "/f", "/t"]);
  } else {
    serverProcess.kill("SIGTERM");
  }
  serverProcess = null;
}

// ─── BrowserWindow ────────────────────────────────────────────────────────────

function createMainWindow(onReady?: () => void): void {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 750,
    minWidth: 800,
    minHeight: 600,
    show: false, // shown after server is ready
    backgroundColor: "#FAFAF8",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    title: "BEET",
  });

  // Show a loading page while the server starts
  mainWindow.loadURL(
    "data:text/html,<html><body style=\"font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#FAFAF8\"><p style=\"color:#6b7280\">Starting BEET…</p></body></html>"
  );

  mainWindow.once("ready-to-show", () => {
    if (serverReady) mainWindow?.show();
    onReady?.();
  });

  // Open external links in the default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ─── IPC handlers ─────────────────────────────────────────────────────────────

function setupIpcHandlers(): void {
  ipcMain.handle("get-server-status", () => ({
    ready: serverReady,
    apiBase: API_BASE,
  }));

  ipcMain.handle("get-clipboard-text", () => clipboard.readText());

  ipcMain.handle("write-clipboard-text", (_event, text: string) => {
    clipboard.writeText(text);
  });

  ipcMain.handle("open-external", (_event, url: string) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
  });
}

// ─── Auto-updater ─────────────────────────────────────────────────────────────

function setupAutoUpdater(): void {
  if (IS_DEV) return; // skip in development

  autoUpdater.checkForUpdatesAndNotify().catch((err) => {
    console.warn("[updater] Could not check for updates:", err.message);
  });

  autoUpdater.on("update-available", () => {
    mainWindow?.webContents.send("update-available");
  });

  autoUpdater.on("update-downloaded", () => {
    mainWindow?.webContents.send("update-downloaded");
  });
}

// ─── App lifecycle ────────────────────────────────────────────────────────────

app.whenReady().then(() => {
  // Prevent multiple instances
  if (!app.requestSingleInstanceLock()) {
    app.quit();
    return;
  }

  setupIpcHandlers();

  // Create the tray before starting the server so the user sees it immediately
  tray = new Tray(getTrayIcon("starting"));
  tray.setToolTip("BEET — Starting…");
  tray.setContextMenu(buildTrayMenu());

  // On macOS, clicking the dock icon should re-open the window
  app.on("activate", () => {
    if (!mainWindow) createMainWindow();
    else { mainWindow.show(); mainWindow.focus(); }
  });

  createMainWindow();
  startServer();
  setupAutoUpdater();
});

// Second instance: focus the existing window
app.on("second-instance", () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

// On macOS, keep the app running when all windows are closed (menu-bar app)
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

// Graceful shutdown: stop the Python server before quitting
app.on("before-quit", () => {
  stopServer();
});
