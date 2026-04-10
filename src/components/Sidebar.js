import React from 'react';
import './Sidebar.css';

const NAV = [
  {
    id: 'analyse', label: 'Analyse', sublabel: 'Classifier un site',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" stroke="currentColor"/>
        <path d="M21 21l-4.35-4.35M11 8v6M8 11h6" stroke="currentColor"/>
      </svg>
    ),
  },
  {
    id: 'historique', label: 'Historique', sublabel: 'Predictions passees',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2v10l4 2" stroke="currentColor"/>
        <circle cx="12" cy="12" r="10" stroke="currentColor"/>
      </svg>
    ),
  },
  {
    id: 'info', label: 'Energies', sublabel: 'Guide de reference',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" stroke="currentColor"/>
        <path d="M12 16v-4M12 8h.01" stroke="currentColor"/>
      </svg>
    ),
  },
  {
    id: 'arbre', label: 'Arbre', sublabel: 'Modele interactif',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="4"  r="2" stroke="currentColor"/>
        <circle cx="5"  cy="20" r="2" stroke="currentColor"/>
        <circle cx="19" cy="20" r="2" stroke="currentColor"/>
        <path d="M12 6v4M12 10l-5 8M12 10l5 8" stroke="currentColor"/>
      </svg>
    ),
  },
  {
    id: 'dimensionnement', label: 'Calcul', sublabel: 'Dimensionnement solaire',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="6" height="6" rx="1" stroke="currentColor"/>
        <rect x="14" y="4" width="6" height="6" rx="1" stroke="currentColor"/>
        <rect x="4" y="14" width="6" height="6" rx="1" stroke="currentColor"/>
        <path d="M14 17h6M17 14v6" stroke="currentColor"/>
      </svg>
    ),
  },
];

const isElectron = typeof window !== 'undefined' && window.electronAPI;

export default function Sidebar({ view, setView }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-glow" />
      <nav className="sidebar-nav">
        {NAV.map(item => (
          <button
            key={item.id}
            className={`nav-item ${view === item.id ? 'active' : ''}`}
            onClick={() => setView(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-text">
              <span className="nav-label">{item.label}</span>
              <span className="nav-sublabel">{item.sublabel}</span>
            </span>
            {view === item.id && <span className="nav-active-bar" />}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="db-indicator" style={isElectron ? {} : { borderColor: 'rgba(0,212,255,0.15)', background: 'rgba(0,212,255,0.08)' }}>
          <span className="db-dot" style={isElectron ? {} : { background: 'var(--accent)', boxShadow: '0 0 6px var(--accent)' }} />
          <span className="db-label" style={isElectron ? {} : { color: 'var(--accent)' }}>
            {isElectron ? 'SQLite connecte' : 'Mode Web'}
          </span>
        </div>
        <div className="model-badge"><span className="model-label">Modele</span><span className="model-value">DecisionTree</span></div>
        <div className="model-badge"><span className="model-label">Profondeur</span><span className="model-value">5</span></div>
        <div className="model-badge"><span className="model-label">Echantillons</span><span className="model-value">40</span></div>
        <div className="model-badge"><span className="model-label">Villes Maroc</span><span className="model-value">43</span></div>
      </div>
    </aside>
  );
}
