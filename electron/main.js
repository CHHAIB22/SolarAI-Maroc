const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs   = require('fs');
const { spawn, spawnSync } = require('child_process');
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

// ── Detect Python ───────────────────────────────────────────────
function getPythonCmd() {
  const cfgPath = path.join(__dirname, 'python_path.txt');
  if (fs.existsSync(cfgPath)) {
    let saved = fs.readFileSync(cfgPath, 'utf8')
      .split('\n')[0].trim().replace(/^"(.*)"$/, '$1');
    if (saved.length > 0) return saved;
  }
  const candidates = [
    'py', 'python', 'python3',
    `C:\\Users\\${process.env.USERNAME || ''}\\AppData\\Local\\Programs\\Python\\Python313\\python.exe`,
    `C:\\Users\\${process.env.USERNAME || ''}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe`,
    `C:\\Users\\${process.env.USERNAME || ''}\\AppData\\Local\\Programs\\Python\\Python311\\python.exe`,
    'C:\\Program Files\\Python313\\python.exe',
    'C:\\Program Files\\Python312\\python.exe',
  ];
  for (const c of candidates) {
    try {
      const r = spawnSync(c, ['--version'], { timeout: 3000 });
      if (r.status === 0) return c;
    } catch {}
  }
  return 'python';
}

const PYTHON_CMD = getPythonCmd();
console.log('[Electron] Python:', PYTHON_CMD);

// ── Window ──────────────────────────────────────────────────────
let mainWindow;
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400, height: 900, minWidth: 1100, minHeight: 700,
    frame: false, titleBarStyle: 'hidden',
    backgroundColor: '#0a0e1a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
  });
  mainWindow.once('ready-to-show', () => mainWindow.show());
  if (isDev) {
    mainWindow.loadURL('http://localhost:3000');
  } else {
    mainWindow.loadFile(path.join(__dirname, '../build/index.html'));
  }
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });

// ── Window controls ─────────────────────────────────────────────
ipcMain.on('window-minimize', () => mainWindow.minimize());
ipcMain.on('window-maximize', () => mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize());
ipcMain.on('window-close',    () => mainWindow.close());

// ── Python runner ───────────────────────────────────────────────
function runPython(scriptPath, args) {
  return new Promise((resolve, reject) => {
    // Split "py -3.13" style into [cmd, ...flags]
    let cmd, cmdArgs;
    if (PYTHON_CMD.includes(' ')) {
      const parts = PYTHON_CMD.split(/\s+/);
      cmd     = parts[0];
      cmdArgs = [...parts.slice(1), scriptPath, ...args];
    } else {
      cmd     = PYTHON_CMD;
      cmdArgs = [scriptPath, ...args];
    }

    console.log('[Python] cmd:', cmd, cmdArgs[0]);

    const py = spawn(cmd, cmdArgs, { shell: true });
    let out = '', err = '';

    py.stdout.on('data', d => { out += d.toString(); });
    py.stderr.on('data', d => { err += d.toString(); });

    py.on('close', code => {
      console.log('[Python] exit:', code);
      console.log('[Python] stdout:', out.slice(0, 500));
      if (err) console.log('[Python] stderr:', err.slice(0, 500));

      if (code === 0 && out.trim()) {
        try {
          resolve(JSON.parse(out.trim()));
        } catch (e) {
          // Return the raw output as error string (NOT an object)
          reject('Erreur JSON:\n' + out.slice(0, 400));
        }
      } else {
        // Always reject with a plain STRING so React can display it
        const msg = (err || out || '').trim();
        reject(msg || `Python a quitte avec le code ${code}`);
      }
    });

    py.on('error', e => {
      reject(`Impossible de lancer Python.\nCommande: "${PYTHON_CMD}"\nErreur: ${e.message}\n\nSolution: relancez install-and-run.bat`);
    });
  });
}

// ── IPC: Prediction ─────────────────────────────────────────────
ipcMain.handle('run-prediction', async (event, params) => {
  const scriptPath = isDev
    ? path.join(__dirname, '../python/predict.py')
    : path.join(process.resourcesPath, 'python/predict.py');

  const dbPath = isDev
    ? path.join(__dirname, '../python/energie_renouvelable.db')
    : path.join(app.getPath('userData'), 'energie_renouvelable.db');

  // runPython rejects with a plain string — wrap in { error } for React
  try {
    return await runPython(scriptPath, [
      String(params.solaire), String(params.vent),
      String(params.eau),     String(params.biomasse),
      String(params.terrain), String(params.temperature),
      dbPath,
      String(params.nomSite || 'Site'),
    ]);
  } catch (msg) {
    throw new Error(String(msg));
  }
});

// ── IPC: History ────────────────────────────────────────────────
ipcMain.handle('load-history', async () => {
  const scriptPath = isDev
    ? path.join(__dirname, '../python/history.py')
    : path.join(process.resourcesPath, 'python/history.py');

  const dbPath = isDev
    ? path.join(__dirname, '../python/energie_renouvelable.db')
    : path.join(app.getPath('userData'), 'energie_renouvelable.db');

  try { return await runPython(scriptPath, [dbPath]); }
  catch (e) { return []; }
});
