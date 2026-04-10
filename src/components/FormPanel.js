import React, { useState, useRef, useEffect } from 'react';
import VILLES_MAROC from '../data/villesMaroc';
import './FormPanel.css';

const DEFAULTS = {
  nomSite: 'Guelmim, Maroc', solaire: 2100, vent: 5.5,
  eau: 'Non', biomasse: 'Faible', terrain: 'Grand', temperature: 22,
};

const SLIDERS = [
  { key: 'solaire',     label: 'Irradiation Solaire', unit: 'kWh/m²/an', min: 500,  max: 3000, step: 50,  color: '#f59e0b', icon: '☀' },
  { key: 'vent',        label: 'Vitesse du Vent',      unit: 'm/s',       min: 0,    max: 25,   step: 0.5, color: '#38bdf8', icon: '◌' },
  { key: 'temperature', label: 'Temperature Moyenne',  unit: '°C',        min: -20,  max: 60,   step: 1,   color: '#f97316', icon: '◉' },
];

const SELECTS = [
  { key: 'eau',      label: 'Disponibilite Eau',     opts: ['Non','Oui'],              icon: '◈' },
  { key: 'biomasse', label: 'Disponibilite Biomasse', opts: ['Faible','Moyen','Eleve'], icon: '✿' },
  { key: 'terrain',  label: 'Disponibilite Terrain',  opts: ['Petit','Moyen','Grand'],  icon: '⬡' },
];

// Group cities by region
const REGIONS = VILLES_MAROC.reduce((acc, v) => {
  if (!acc[v.region]) acc[v.region] = [];
  acc[v.region].push(v);
  return acc;
}, {});

export default function FormPanel({ onSubmit, loading }) {
  const [form,        setForm]        = useState(DEFAULTS);
  const [search,      setSearch]      = useState('');
  const [dropOpen,    setDropOpen]    = useState(false);
  const [selectedVille, setSelectedVille] = useState('Guelmim');
  const dropRef = useRef(null);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => { if (dropRef.current && !dropRef.current.contains(e.target)) setDropOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const selectVille = (ville) => {
    setSelectedVille(ville.nom);
    setForm({
      nomSite:     `${ville.nom}, Maroc`,
      solaire:     ville.solaire,
      vent:        ville.vent,
      eau:         ville.eau,
      biomasse:    ville.biomasse,
      terrain:     ville.terrain,
      temperature: ville.temperature,
    });
    setDropOpen(false);
    setSearch('');
  };

  // Filter cities by search
  const query = search.toLowerCase();
  const filtered = VILLES_MAROC.filter(v =>
    v.nom.toLowerCase().includes(query) || v.region.toLowerCase().includes(query)
  );

  return (
    <div className="form-panel">
      <div className="form-panel-header">
        <h2 className="form-title">Parametres du Site</h2>
        <p className="form-subtitle">Selectionnez une ville ou saisissez les valeurs manuellement</p>
      </div>

      <form onSubmit={e => { e.preventDefault(); onSubmit(form); }} className="form-body">

        {/* ── City selector ───────────────────────── */}
        <div className="form-section">
          <div className="field-label"><span className="field-icon">◎</span>Ville du Maroc</div>
          <div className="city-selector" ref={dropRef}>
            <button
              type="button"
              className="city-trigger"
              onClick={() => setDropOpen(o => !o)}
            >
              <span className="city-flag">🇲🇦</span>
              <span className="city-selected">{selectedVille}</span>
              <span className={`city-arrow ${dropOpen ? 'open' : ''}`}>▾</span>
            </button>

            {dropOpen && (
              <div className="city-dropdown">
                <div className="city-search-wrap">
                  <input
                    className="city-search"
                    placeholder="Rechercher une ville..."
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    autoFocus
                  />
                </div>
                <div className="city-list">
                  {query
                    ? filtered.map(v => (
                        <button key={v.nom} type="button" className="city-item" onClick={() => selectVille(v)}>
                          <span className="city-item-name">{v.nom}</span>
                          <span className="city-item-region">{v.region}</span>
                          <span className="city-item-sol">{v.solaire} kWh</span>
                        </button>
                      ))
                    : Object.entries(REGIONS).map(([region, villes]) => (
                        <div key={region} className="city-group">
                          <div className="city-group-label">{region}</div>
                          {villes.map(v => (
                            <button
                              key={v.nom}
                              type="button"
                              className={`city-item ${selectedVille === v.nom ? 'active' : ''}`}
                              onClick={() => selectVille(v)}
                            >
                              <span className="city-item-name">{v.nom}</span>
                              <span className="city-item-region">{v.region}</span>
                              <span className="city-item-sol">{v.solaire} kWh</span>
                            </button>
                          ))}
                        </div>
                      ))
                  }
                  {query && filtered.length === 0 && (
                    <div className="city-empty">Aucune ville trouvee</div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Editable site name */}
          <input
            className="text-input"
            style={{ marginTop: 8 }}
            value={form.nomSite}
            onChange={e => set('nomSite', e.target.value)}
            placeholder="Nom du site"
          />
        </div>

        {/* ── Sliders ─────────────────────────────── */}
        <div className="form-section">
          <div className="section-title">Parametres Numeriques</div>
          {SLIDERS.map(s => (
            <SliderField key={s.key} {...s} value={form[s.key]} onChange={v => set(s.key, v)} />
          ))}
        </div>

        {/* ── Selects ─────────────────────────────── */}
        <div className="form-section">
          <div className="section-title">Disponibilites</div>
          <div className="selects-grid">
            {SELECTS.map(s => (
              <SelectField key={s.key} {...s} value={form[s.key]} onChange={v => set(s.key, v)} />
            ))}
          </div>
        </div>

        {/* ── Submit ──────────────────────────────── */}
        <button type="submit" className={`submit-btn ${loading ? 'loading' : ''}`} disabled={loading}>
          {loading ? (
            <><span className="spinner" />Analyse en cours...</>
          ) : (
            <><svg width="16" height="16" viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round"><path d="M9 18l6-6-6-6" stroke="currentColor"/></svg>Lancer l'Analyse IA</>
          )}
        </button>
      </form>
    </div>
  );
}

function SliderField({ label, unit, min, max, step, color, icon, value, onChange }) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  return (
    <div className="slider-field">
      <div className="slider-header">
        <span className="slider-icon" style={{ color }}>{icon}</span>
        <span className="slider-label">{label}</span>
        <span className="slider-value" style={{ color }}>
          {Number.isInteger(step) ? value : Number(value).toFixed(1)}
          <span className="slider-unit"> {unit}</span>
        </span>
      </div>
      <div className="slider-track-wrap">
        <div className="slider-fill" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}60, ${color})` }} />
        <div className="slider-thumb" style={{ left: `${pct}%`, borderColor: color, boxShadow: `0 0 10px ${color}99, 0 2px 6px rgba(0,0,0,0.5)` }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))} className="slider-input" />
      </div>
      <div className="slider-minmax"><span>{min} {unit}</span><span>{max} {unit}</span></div>
    </div>
  );
}

function SelectField({ label, opts, icon, value, onChange }) {
  return (
    <div className="select-field">
      <div className="select-label"><span className="select-icon">{icon}</span>{label}</div>
      <div className="select-pills">
        {opts.map(opt => (
          <button key={opt} type="button" className={`pill ${value === opt ? 'active' : ''}`} onClick={() => onChange(opt)}>
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}
