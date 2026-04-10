const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Window controls
  minimize:    () => ipcRenderer.send('window-minimize'),
  maximize:    () => ipcRenderer.send('window-maximize'),
  close:       () => ipcRenderer.send('window-close'),

  // Python ML bridge
  runPrediction: (params) => ipcRenderer.invoke('run-prediction', params),
  loadHistory:   (dbPath) => ipcRenderer.invoke('load-history', dbPath),
});
