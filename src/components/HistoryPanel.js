import React from 'react';
import VILLES_MAROC from '../data/villesMaroc';
import './HistoryPanel.css';

const ENERGIE_COLORS = {
  'Energie Solaire PV': '#f59e0b',
  'Energie Eolienne':   '#38bdf8',
  'Hydroelectricite':   '#3b82f6',
  'Energie Biomasse':   '#22c55e',
};
const ENERGIE_ICONS = {
  'Energie Solaire PV': '☀',
  'Energie Eolienne':   '◌',
  'Hydroelectricite':   '◈',
  'Energie Biomasse':   '✿',
};

// Build a quick lookup nom → region
const VILLE_MAP = Object.fromEntries(VILLES_MAROC.map(v => [v.nom, v]));

function getVilleFromSite(nomSite) {
  if (!nomSite) return null;
  const nom = nomSite.replace(', Maroc', '').replace(', Morocco', '').trim();
  return VILLE_MAP[nom] || null;
}

export default function HistoryPanel({ history, onRefresh }) {
  return (
    <div className="history-panel fade-in">
      <div className="history-header">
        <div>
          <h2 className="history-title">Historique des Predictions</h2>
          <p className="history-sub">
            {history.length} prediction{history.length !== 1 ? 's' : ''} enregistree{history.length !== 1 ? 's' : ''}
            {' · '}<span style={{ color: '#f59e0b' }}>Maroc 🇲🇦</span>
          </p>
        </div>
        <button className="refresh-btn" onClick={onRefresh}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round">
            <polyline points="23 4 23 10 17 10" stroke="currentColor"/>
            <polyline points="1 20 1 14 7 14" stroke="currentColor"/>
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" stroke="currentColor"/>
          </svg>
          Actualiser
        </button>
      </div>

      {history.length === 0 ? (
        <div className="history-empty">
          <div className="history-empty-icon">🇲🇦</div>
          <p>Aucune prediction enregistree</p>
          <p className="history-empty-hint">Analysez une ville marocaine pour voir l'historique</p>
          <div className="ville-preview">
            {VILLES_MAROC.slice(0, 6).map(v => (
              <span key={v.nom} className="ville-chip">{v.nom}</span>
            ))}
          </div>
        </div>
      ) : (
        <div className="history-table-wrap">
          <table className="history-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Date</th>
                <th>Site / Ville</th>
                <th>Irradiation</th>
                <th>Vent</th>
                <th>Eau</th>
                <th>Biomasse</th>
                <th>Terrain</th>
                <th>Temp.</th>
                <th>Recommandation</th>
                <th>Confiance</th>
              </tr>
            </thead>
            <tbody>
              {history.map((row, i) => {
                const color = ENERGIE_COLORS[row.energie_recommandee] || '#8ba3c0';
                const icon  = ENERGIE_ICONS[row.energie_recommandee]  || '⚡';
                const ville = getVilleFromSite(row.nom_site);

                return (
                  <tr key={row.id} style={{ animationDelay: `${i * 40}ms` }}>
                    <td className="cell-id">{row.id}</td>
                    <td className="cell-date">{row.date_prediction}</td>
                    <td className="cell-site">
                      <div className="site-cell">
                        <span className="site-flag">🇲🇦</span>
                        <div>
                          <div className="site-name">{row.nom_site}</div>
                          {ville && <div className="site-region">{ville.region}</div>}
                        </div>
                      </div>
                    </td>
                    <td className="cell-num" style={{ color: '#f59e0b' }}>{row.irradiation_solaire} <span className="unit">kWh</span></td>
                    <td className="cell-num" style={{ color: '#38bdf8' }}>{row.vitesse_vent} <span className="unit">m/s</span></td>
                    <td>
                      <span className={`tag ${row.disponibilite_eau === 'Oui' ? 'tag-oui' : 'tag-non'}`}>
                        {row.disponibilite_eau}
                      </span>
                    </td>
                    <td><span className="tag tag-neutral">{row.disponibilite_biomasse}</span></td>
                    <td><span className="tag tag-neutral">{row.disponibilite_terrain}</span></td>
                    <td className="cell-num" style={{ color: '#f97316' }}>{row.temperature_moyenne}°C</td>
                    <td>
                      <span className="energie-badge" style={{ color, borderColor: color + '40', background: color + '12' }}>
                        <span>{icon}</span> {row.energie_recommandee}
                      </span>
                    </td>
                    <td>
                      <div className="conf-cell">
                        <div className="conf-mini-bar">
                          <div className="conf-mini-fill" style={{ width: `${row.confiance_pct}%`, background: color }} />
                        </div>
                        <span style={{ color }}>{row.confiance_pct?.toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
