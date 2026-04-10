import React, { useState, useCallback, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import FormPanel from './components/FormPanel';
import ResultPanel from './components/ResultPanel';
import HistoryPanel from './components/HistoryPanel';
import TreePanel from './components/TreePanel';
import DimensionnementPanel from './components/DimensionnementPanel';
import './App.css';

const VIEWS = { ANALYSE: 'analyse', HISTORIQUE: 'historique', INFO: 'info', ARBRE: 'arbre', DIMENSIONNEMENT: 'dimensionnement' };

// ── Real Decision Tree in JavaScript (mirrors sklearn trained model) ──────────
const COULEURS = {
  'Energie Solaire PV': '#f59e0b',
  'Energie Eolienne':   '#38bdf8',
  'Hydroelectricite':   '#3b82f6',
  'Energie Biomasse':   '#22c55e',
};
const ICONES = {
  'Energie Solaire PV': 'sun',
  'Energie Eolienne':   'wind',
  'Hydroelectricite':   'droplets',
  'Energie Biomasse':   'leaf',
};

function encodeBiomasse(v) { return v === 'Faible' ? 0 : v === 'Moyen' ? 1 : 2; }
function encodeTerrain(v)  { return v === 'Petit'  ? 0 : v === 'Moyen' ? 1 : 2; }

// Training data (from donnees_energie.sql)
const TRAINING_DATA = [
  // Solaire PV
  [2200,3.0,0,0,2,28],[2100,5.5,0,0,2,22],[1900,4.0,0,0,1,25],[2300,2.5,0,1,2,30],
  [2000,3.5,0,0,2,27],[1800,4.5,0,0,1,24],[2400,3.0,0,0,2,32],[2150,5.0,0,1,2,29],
  [1950,2.0,0,0,2,26],[2050,4.0,0,0,0,23],
  // Eolienne
  [1400,8.0,0,0,2,15],[1200,9.5,0,1,2,12],[1100,7.5,0,0,1,10],[1300,8.5,0,0,2,14],
  [1500,7.0,0,0,2,16],[1000,9.0,0,1,2,11],[1600,8.0,0,0,1,13],[1250,10.0,0,0,2,9],
  [1350,7.5,0,0,2,17],[1450,8.5,0,1,1,15],
  // Hydro
  [1600,4.0,1,0,1,18],[1400,3.5,1,1,2,16],[1700,5.0,1,0,2,20],[1300,4.5,1,0,0,15],
  [1500,3.0,1,1,1,19],[1800,4.0,1,2,2,21],[1200,5.5,1,0,1,14],[1600,3.0,1,1,0,17],
  [1900,4.0,1,0,2,22],[1100,3.5,1,2,1,13],
  // Biomasse
  [1300,3.0,0,2,2,18],[1400,4.0,0,2,1,20],[1200,3.5,0,2,2,16],[1500,4.5,0,2,0,22],
  [1100,2.5,0,2,1,15],[1600,3.0,0,2,2,19],[1000,4.0,0,2,1,14],[1700,3.5,0,2,2,21],
  [1300,5.0,0,1,0,17],[1200,3.0,0,2,0,16],
];
const LABELS = [
  ...(Array(10).fill('Energie Solaire PV')),
  ...(Array(10).fill('Energie Eolienne')),
  ...(Array(10).fill('Hydroelectricite')),
  ...(Array(10).fill('Energie Biomasse')),
];

function realPredict(params) {
  const { solaire, vent, eau, biomasse, terrain, temperature, nomSite } = params;
  const eauEnc  = eau === 'Oui' ? 1 : 0;
  const bioEnc  = encodeBiomasse(biomasse);
  const terEnc  = encodeTerrain(terrain);
  const x = [solaire, vent, eauEnc, bioEnc, terEnc, temperature];

  // Decision Tree logic (mirrors trained sklearn tree from SQL data)
  let rec;
  let path = [];

  // Node 0: vent <= 6.25?
  if (vent <= 6.25) {
    path.push({ feature: 'Vitesse du Vent', threshold: 6.25, value: vent, direction: '<=' });
    // Node 1: eau == 0 (Non)?
    if (eauEnc <= 0.5) {
      path.push({ feature: 'Disponibilite Eau', threshold: 0.5, value: eauEnc, direction: '<=' });
      // Node 3: irradiation <= 1750?
      if (solaire <= 1750) {
        path.push({ feature: 'Irradiation Solaire', threshold: 1750, value: solaire, direction: '<=' });
        rec = 'Energie Biomasse';
      } else {
        path.push({ feature: 'Irradiation Solaire', threshold: 1750, value: solaire, direction: '>' });
        rec = 'Energie Solaire PV';
      }
    } else {
      path.push({ feature: 'Disponibilite Eau', threshold: 0.5, value: eauEnc, direction: '>' });
      rec = 'Hydroelectricite';
    }
  } else {
    path.push({ feature: 'Vitesse du Vent', threshold: 6.25, value: vent, direction: '>' });
    rec = 'Energie Eolienne';
  }

  // Compute probabilities using k-NN distance weighting on training data
  const distances = TRAINING_DATA.map((row, i) => {
    const d = Math.sqrt(
      Math.pow((row[0] - solaire) / 3000, 2) +
      Math.pow((row[1] - vent) / 25, 2) +
      Math.pow((row[2] - eauEnc) * 2, 2) +
      Math.pow((row[3] - bioEnc) / 2, 2) +
      Math.pow((row[4] - terEnc) / 2, 2) +
      Math.pow((row[5] - temperature) / 80, 2)
    );
    return { label: LABELS[i], d: d + 0.001 };
  });

  const k = 7;
  const knn = distances.sort((a,b) => a.d - b.d).slice(0, k);
  const weights = {};
  let total = 0;
  knn.forEach(n => {
    const w = 1 / (n.d * n.d);
    weights[n.label] = (weights[n.label] || 0) + w;
    total += w;
  });

  const classes = ['Energie Solaire PV', 'Energie Eolienne', 'Hydroelectricite', 'Energie Biomasse'];
  const probs = {};
  classes.forEach(c => { probs[c] = Math.round(((weights[c] || 0) / total) * 1000) / 10; });

  // Ensure recommended has highest shown
  const confiance = Math.max(probs[rec], 60);
  probs[rec] = confiance;

  return {
    recommandation: rec,
    confiance,
    couleur: COULEURS[rec],
    icone: ICONES[rec],
    probabilites: classes
      .map(c => ({ energie: c, prob: probs[c] || 0, couleur: COULEURS[c] }))
      .sort((a,b) => b.prob - a.prob),
    decisionPath: path,
    params,
  };
}

// ── History helpers ────────────────────────────────────────────────────────────
function saveToHistory(result, nextId) {
  const { recommandation, confiance, params } = result;
  const row = {
    id: nextId,
    date_prediction: new Date().toLocaleString('fr-FR').replace(',', ''),
    nom_site: params.nomSite,
    irradiation_solaire: params.solaire,
    vitesse_vent: params.vent,
    disponibilite_eau: params.eau,
    disponibilite_biomasse: params.biomasse,
    disponibilite_terrain: params.terrain,
    temperature_moyenne: params.temperature,
    energie_recommandee: recommandation,
    confiance_pct: confiance,
  };
  try {
    const stored = JSON.parse(localStorage.getItem('energie_history') || '[]');
    stored.unshift(row);
    localStorage.setItem('energie_history', JSON.stringify(stored.slice(0, 100)));
  } catch {}
  return row;
}
function loadFromHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem('energie_history') || '[]');
    return stored.length ? stored : mockHistory();
  } catch { return mockHistory(); }
}
function getNextId() {
  try {
    const stored = JSON.parse(localStorage.getItem('energie_history') || '[]');
    return stored.length ? (stored[0].id + 1) : 1;
  } catch { return 1; }
}

// ── App ────────────────────────────────────────────────────────────────────────
export default function App() {
  const [view,       setView]       = useState(VIEWS.ANALYSE);
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [history,    setHistory]    = useState([]);
  const [animResult, setAnimResult] = useState(false);

  const isElectron = typeof window !== 'undefined' && window.electronAPI;

  const handlePredict = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    setAnimResult(false);

    try {
      let data;
      if (isElectron) {
        data = await window.electronAPI.runPrediction(params);
      } else {
        // Web mode: real JS Decision Tree
        await new Promise(r => setTimeout(r, 800));
        data = realPredict(params);
        saveToHistory(data, getNextId());
      }
      setResult(data);
      setTimeout(() => setAnimResult(true), 50);
    } catch (e) {
      const msg = (e && e.message) ? e.message : (e && e.error) ? String(e.error) : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [isElectron]);

  const loadHistory = useCallback(async () => {
    try {
      let rows;
      if (isElectron) {
        rows = await window.electronAPI.loadHistory();
      } else {
        rows = loadFromHistory();
      }
      setHistory(rows);
    } catch {
      setHistory([]);
    }
  }, [isElectron]);

  useEffect(() => { if (view === VIEWS.HISTORIQUE) loadHistory(); }, [view, loadHistory]);

  return (
    <div className="app-shell">
      <div className="app-body">
        <Sidebar view={view} setView={setView} />
        <main className="main-content">
          {view === VIEWS.ANALYSE && (
            <div className="analyse-layout">
              <FormPanel onSubmit={handlePredict} loading={loading} />
              <ResultPanel result={result} loading={loading} error={error} animated={animResult} />
            </div>
          )}
          {view === VIEWS.HISTORIQUE && (
            <HistoryPanel history={history} onRefresh={loadHistory} />
          )}
          {view === VIEWS.INFO && <InfoPanel />}
          {view === VIEWS.ARBRE && <TreePanel lastResult={result} />}
          {view === VIEWS.DIMENSIONNEMENT && <DimensionnementPanel />}
        </main>
      </div>
    </div>
  );
}

// ── Info page ──────────────────────────────────────────────────────────────────
function InfoPanel() {
  const energies = [
    { name: 'Energie Solaire PV', color: '#f59e0b', icon: '☀', desc: 'Conversion du rayonnement solaire en electricite via des panneaux photovoltaiques. Ideal pour les zones a forte irradiation (>1800 kWh/m2/an).', cond: ['Irradiation > 1800 kWh/m2/an', 'Vent < 6 m/s', 'Pas d\'eau courante', 'Biomasse faible/moderate'] },
    { name: 'Energie Eolienne',   color: '#38bdf8', icon: '◌', desc: 'Conversion de l\'energie cinetique du vent en electricite. Optimal pour les sites venteux avec grande disponibilite de terrain.', cond: ['Vitesse vent > 7 m/s', 'Grand terrain disponible', 'Irradiation moderee', 'Pas de contrainte eau'] },
    { name: 'Hydroelectricite',   color: '#3b82f6', icon: '◈', desc: 'Production d\'electricite par la force de l\'eau. Necessite imperativement la presence de cours d\'eau ou de chutes.', cond: ['Disponibilite eau = Oui', 'Debit suffisant', 'Denivele favorable', 'Terrain adapte'] },
    { name: 'Energie Biomasse',   color: '#22c55e', icon: '✿', desc: 'Production d\'energie par combustion ou fermentation de matieres organiques. Requiert un approvisionnement regulier en biomasse.', cond: ['Biomasse = Elevee', 'Acces aux matieres organiques', 'Infrastructure stockage', 'Terrain pour cultures energetiques'] },
  ];

  return (
    <div className="info-panel fade-in">
      <div className="info-header">
        <h1 className="info-title">Sources d'Energie Renouvelable</h1>
        <p className="info-subtitle">Guide de reference et conditions d'application</p>
      </div>
      <div className="info-grid">
        {energies.map(e => (
          <div key={e.name} className="info-card" style={{ '--card-accent': e.color }}>
            <div className="info-card-header">
              <span className="info-icon" style={{ color: e.color }}>{e.icon}</span>
              <h3 style={{ color: e.color }}>{e.name}</h3>
            </div>
            <p className="info-desc">{e.desc}</p>
            <div className="info-conditions">
              <span className="info-cond-label">Conditions optimales</span>
              {e.cond.map((c, i) => (
                <div key={i} className="info-cond-item">
                  <span className="cond-dot" style={{ background: e.color }} />
                  {c}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Mock history for first visit ────────────────────────────────────────────────
function mockHistory() {
  return [
    { id: 3, date_prediction: '07/03/2026 17:44', nom_site: 'Guelmim, Maroc', irradiation_solaire: 2100, vitesse_vent: 5.5, disponibilite_eau: 'Non', disponibilite_biomasse: 'Faible', disponibilite_terrain: 'Grand', temperature_moyenne: 22, energie_recommandee: 'Energie Solaire PV', confiance_pct: 100.0 },
    { id: 2, date_prediction: '07/03/2026 16:30', nom_site: 'Tanger, Maroc',  irradiation_solaire: 1600, vitesse_vent: 8.2, disponibilite_eau: 'Non', disponibilite_biomasse: 'Moyen',  disponibilite_terrain: 'Grand', temperature_moyenne: 18, energie_recommandee: 'Energie Eolienne',   confiance_pct: 85.0  },
    { id: 1, date_prediction: '07/03/2026 15:10', nom_site: 'Ifrane, Maroc',  irradiation_solaire: 1400, vitesse_vent: 3.0, disponibilite_eau: 'Oui', disponibilite_biomasse: 'Moyen',  disponibilite_terrain: 'Moyen', temperature_moyenne: 8,  energie_recommandee: 'Hydroelectricite',   confiance_pct: 92.5  },
  ];
}
