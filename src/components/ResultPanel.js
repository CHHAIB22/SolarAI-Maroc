import React, { useEffect, useState } from 'react';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';
import './ResultPanel.css';

const ICONS = {
  sun:      <SunIcon />,
  wind:     <WindIcon />,
  droplets: <DropletsIcon />,
  leaf:     <LeafIcon />,
};

export default function ResultPanel({ result, loading, error, animated }) {
  const [displayedConf, setDisplayedConf] = useState(0);

  useEffect(() => {
    if (!result) return;
    setDisplayedConf(0);
    let start = null;
    const target = result.confiance;
    const dur = 1200;
    const raf = requestAnimationFrame(function tick(ts) {
      if (!start) start = ts;
      const p = Math.min((ts - start) / dur, 1);
      setDisplayedConf(Math.round(p * target * 10) / 10);
      if (p < 1) requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(raf);
  }, [result]);

  if (!result && !loading && !error) return <EmptyState />;
  if (loading) return <LoadingState />;
  if (error)   return <ErrorState error={error} />;

  const { recommandation, confiance, couleur, icone, probabilites, decisionPath, params } = result;

  // Radar data from params
  const radarData = [
    { subject: 'Solaire',     value: Math.min(100, (params.solaire / 3000) * 100) },
    { subject: 'Vent',        value: Math.min(100, (params.vent / 25) * 100) },
    { subject: 'Eau',         value: params.eau === 'Oui' ? 100 : 0 },
    { subject: 'Biomasse',    value: params.biomasse === 'Eleve' ? 100 : params.biomasse === 'Moyen' ? 50 : 15 },
    { subject: 'Terrain',     value: params.terrain === 'Grand' ? 100 : params.terrain === 'Moyen' ? 50 : 20 },
    { subject: 'Temperature', value: Math.min(100, ((params.temperature + 20) / 80) * 100) },
  ];

  return (
    <div className={`result-panel ${animated ? 'visible' : ''}`}>
      <div className="result-scroll">

        {/* ── Hero recommendation ── */}
        <div className="hero-card" style={{ '--e-color': couleur }}>
          <div className="hero-bg-glow" />
          <div className="hero-content">
            <div className="hero-icon-wrap">
              <div className="hero-rings">
                <div className="ring r1" /><div className="ring r2" /><div className="ring r3" />
              </div>
              <div className="hero-icon">{ICONS[icone] || ICONS.sun}</div>
            </div>
            <div className="hero-text">
              <div className="hero-label">Source Recommandee</div>
              <div className="hero-name" style={{ color: couleur }}>{recommandation}</div>
              <div className="hero-conf-row">
                <div className="conf-bar-wrap">
                  <div className="conf-bar-fill" style={{ width: `${confiance}%`, background: couleur }} />
                </div>
                <span className="conf-value" style={{ color: couleur }}>{displayedConf.toFixed(1)}%</span>
              </div>
              <div className="hero-site">
                <span className="hero-site-dot" />
                {params.nomSite}
              </div>
            </div>
          </div>
        </div>

        <div className="charts-row">
          {/* Probability bars */}
          <div className="chart-card">
            <div className="chart-title">Probabilites par Energie</div>
            <div className="prob-bars">
              {probabilites.map((p, i) => (
                <div key={p.energie} className="prob-row" style={{ animationDelay: `${i * 80}ms` }}>
                  <span className="prob-name">{p.energie.replace('Energie ', '')}</span>
                  <div className="prob-track">
                    <div
                      className="prob-fill"
                      style={{
                        width: `${p.prob}%`,
                        background: `linear-gradient(90deg, ${p.couleur}60, ${p.couleur})`,
                        animationDelay: `${i * 80 + 200}ms`
                      }}
                    />
                  </div>
                  <span className="prob-pct" style={{ color: p.couleur }}>{p.prob.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Radar chart */}
          <div className="chart-card">
            <div className="chart-title">Profil du Site</div>
            <ResponsiveContainer width="100%" height={180}>
              <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <PolarGrid stroke="#1e3050" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#8ba3c0', fontSize: 10 }} />
                <Radar dataKey="value" stroke={couleur} fill={couleur} fillOpacity={0.18} strokeWidth={1.5} dot={{ fill: couleur, r: 3 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bar chart */}
        <div className="chart-card chart-card-full">
          <div className="chart-title">Comparaison des Confidences</div>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={probabilites} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}
                      barCategoryGap="30%">
              <XAxis dataKey="energie" tick={{ fill: '#8ba3c0', fontSize: 10 }}
                     tickFormatter={v => v.replace('Energie ', '').replace('Hydroelectricite','Hydro')} />
              <YAxis tick={{ fill: '#8ba3c0', fontSize: 9 }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#0f1929', border: '1px solid #1e3050', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#e8f4ff' }}
                formatter={v => [`${v.toFixed(1)}%`, 'Confiance']}
              />
              <Bar dataKey="prob" radius={[4,4,0,0]}>
                {probabilites.map((p) => (
                  <Cell key={p.energie} fill={p.couleur} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Decision path */}
        {decisionPath && decisionPath.length > 0 && (
          <div className="chart-card chart-card-full">
            <div className="chart-title">Chemin de Decision</div>
            <div className="decision-path">
              {decisionPath.map((step, i) => (
                <React.Fragment key={i}>
                  <div className="path-step">
                    <div className="path-node">
                      <span className="path-feat">{step.feature}</span>
                      <span className="path-op" style={{ color: step.direction === '<=' ? '#f59e0b' : '#38bdf8' }}>
                        {step.direction} {step.threshold}
                      </span>
                      <span className="path-val">Valeur: {step.value}</span>
                    </div>
                  </div>
                  {i < decisionPath.length - 1 && <div className="path-arrow">↓</div>}
                </React.Fragment>
              ))}
              <div className="path-arrow">↓</div>
              <div className="path-result" style={{ borderColor: couleur, color: couleur }}>
                ✓ {recommandation}
              </div>
            </div>
          </div>
        )}

        {/* Parameters summary */}
        <div className="chart-card chart-card-full">
          <div className="chart-title">Parametres Analyses</div>
          <div className="params-grid">
            {[
              { label: 'Irradiation', value: `${params.solaire} kWh/m²/an`, color: '#f59e0b' },
              { label: 'Vent', value: `${params.vent} m/s`, color: '#38bdf8' },
              { label: 'Eau', value: params.eau, color: params.eau === 'Oui' ? '#3b82f6' : '#4a6380' },
              { label: 'Biomasse', value: params.biomasse, color: '#22c55e' },
              { label: 'Terrain', value: params.terrain, color: '#a855f7' },
              { label: 'Temperature', value: `${params.temperature}°C`, color: '#f97316' },
            ].map(p => (
              <div key={p.label} className="param-chip" style={{ '--chip-color': p.color }}>
                <span className="chip-label">{p.label}</span>
                <span className="chip-value" style={{ color: p.color }}>{p.value}</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="state-center">
      <div className="empty-icon">
        <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
          <circle cx="28" cy="28" r="27" stroke="#1e3050" strokeWidth="2"/>
          <circle cx="28" cy="28" r="18" stroke="#263d5f" strokeWidth="1.5" strokeDasharray="4 3"/>
          <circle cx="28" cy="28" r="6" fill="#1e3050"/>
          <circle cx="28" cy="28" r="2" fill="#00d4ff" opacity="0.5"/>
        </svg>
      </div>
      <h3 className="empty-title">Aucune Analyse</h3>
      <p className="empty-sub">Renseignez les parametres du site<br/>et lancez l'analyse pour voir les resultats</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="state-center">
      <div className="loading-rings">
        <div className="l-ring" style={{ animationDelay: '0s' }} />
        <div className="l-ring" style={{ animationDelay: '0.15s' }} />
        <div className="l-ring" style={{ animationDelay: '0.3s' }} />
      </div>
      <p className="loading-text">Analyse en cours...</p>
      <p className="loading-sub">Arbre de Decision en traitement</p>
    </div>
  );
}

function ErrorState({ error }) {
  const isPythonMissing = error && (error.includes('Impossible') || error.includes('ENOENT') || error.includes('Cannot'));
  const isModuleMissing = error && (error.includes('ModuleNotFoundError') || error.includes('No module'));
  const isJsonError     = error && error.includes('JSON');

  return (
    <div className="state-center">
      <div className="error-icon">⚠</div>
      <h3 className="error-title">Erreur de Prediction</h3>
      <pre className="error-detail">{error}</pre>
      {isPythonMissing && (
        <div className="error-hint">
          💡 Python introuvable — relancez <code>install-and-run.bat</code>
        </div>
      )}
      {isModuleMissing && (
        <div className="error-hint">
          💡 Module manquant — exécutez : <code>pip install scikit-learn numpy</code>
        </div>
      )}
      {isJsonError && (
        <div className="error-hint">
          💡 Erreur Python — vérifiez la console Electron (F12)
        </div>
      )}
    </div>
  );
}

// SVG Icons
function SunIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round">
      <circle cx="12" cy="12" r="5" stroke="currentColor"/>
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M17.66 6.34l-1.41 1.41M6.34 17.66l-1.41 1.41" stroke="currentColor"/>
    </svg>
  );
}
function WindIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round">
      <path d="M9.59 4.59A2 2 0 1112 8H2" stroke="currentColor"/>
      <path d="M12.59 19.41A2 2 0 1015 16H2" stroke="currentColor"/>
      <path d="M6.8 13.8A2 2 0 109 17H2" stroke="currentColor"/>
    </svg>
  );
}
function DropletsIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round">
      <path d="M7 16.3c2.2 0 4-1.83 4-4.05 0-1.16-.57-2.26-1.71-3.19S7.29 6.75 7 5.3c-.29 1.45-1.14 2.84-2.29 3.76S3 12.09 3 13.25c0 2.22 1.8 4.05 4 4.05z" stroke="currentColor"/>
      <path d="M12.56 6.6A10.97 10.97 0 0014 3.02c.5 2.5 2 4.9 4 6.5s3 3.5 3 5.5a6.98 6.98 0 01-11.91 4.97" stroke="currentColor"/>
    </svg>
  );
}
function LeafIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round">
      <path d="M11 20A7 7 0 0118 13V6c0 0-4 0-7 3S5 14 5 17c0 0 3-3 6-3" stroke="currentColor"/>
      <path d="M11 20c0-1.5.5-5 4-7" stroke="currentColor"/>
    </svg>
  );
}
