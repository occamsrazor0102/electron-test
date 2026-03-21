/**
 * BEET 2.0 Desktop — Electron preload script
 *
 * Exposes a minimal, typed bridge from the renderer (React app) to the
 * Electron main process via contextBridge.  Only the specific APIs listed
 * here are available to the renderer; Node.js / Electron internals are not
 * exposed directly.
 */

import { contextBridge, ipcRenderer } from "electron";

// ─── Type declarations (also used by the renderer) ────────────────────────────

export interface BeetElectronApi {
  /** True when running inside the Electron shell. */
  isElectron: true;

  /** Returns the current server status from the main process. */
  getServerStatus(): Promise<{ ready: boolean; apiBase: string }>;

  /** Returns the current clipboard text. */
  getClipboardText(): Promise<string>;

  /** Writes text to the clipboard. */
  writeClipboardText(text: string): Promise<void>;

  /** Opens a URL in the system default browser. */
  openExternal(url: string): Promise<void>;

  /** Subscribe to a "check-clipboard-text" event pushed from the main process. */
  onCheckClipboardText(callback: (text: string) => void): () => void;

  /** Subscribe to update-available notifications. */
  onUpdateAvailable(callback: () => void): () => void;

  /** Subscribe to update-downloaded notifications. */
  onUpdateDownloaded(callback: () => void): () => void;
}

// ─── Context bridge ───────────────────────────────────────────────────────────

const beetApi: BeetElectronApi = {
  isElectron: true,

  getServerStatus: () => ipcRenderer.invoke("get-server-status"),

  getClipboardText: () => ipcRenderer.invoke("get-clipboard-text"),

  writeClipboardText: (text) => ipcRenderer.invoke("write-clipboard-text", text),

  openExternal: (url) => ipcRenderer.invoke("open-external", url),

  onCheckClipboardText: (callback) => {
    const handler = (_event: Electron.IpcRendererEvent, text: string) => callback(text);
    ipcRenderer.on("check-clipboard-text", handler);
    // Return an unsubscribe function
    return () => ipcRenderer.removeListener("check-clipboard-text", handler);
  },

  onUpdateAvailable: (callback) => {
    const handler = () => callback();
    ipcRenderer.on("update-available", handler);
    return () => ipcRenderer.removeListener("update-available", handler);
  },

  onUpdateDownloaded: (callback) => {
    const handler = () => callback();
    ipcRenderer.on("update-downloaded", handler);
    return () => ipcRenderer.removeListener("update-downloaded", handler);
  },
};

contextBridge.exposeInMainWorld("beetElectron", beetApi);

// ─── Type augmentation for the renderer window global ─────────────────────────

declare global {
  interface Window {
    beetElectron?: BeetElectronApi;
  }
}
