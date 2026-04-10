import React from 'react';
import './TitleBar.css';

export default function TitleBar() {
  const isElectron = typeof window !== 'undefined' && window.electronAPI;

  return (
    <div className="titlebar">
      <div className="titlebar-drag">
        <div className="titlebar-logo">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="#00d4ff" strokeWidth="1.5"/>
            <path d="M12 2 L12 6 M12 18 L12 22 M2 12 L6 12 M18 12 L22 12" stroke="#00d4ff" strokeWidth="1.5" strokeLinecap="round"/>
            <circle cx="12" cy="12" r="3" fill="#00d4ff" opacity="0.7"/>
          </svg>
          <span className="titlebar-name">Energie Renouvelable</span>
          <span className="titlebar-badge">IA</span>
        </div>
        <span className="titlebar-version">v1.0.0</span>
        {!isElectron && (
          <span className="titlebar-web-badge">🌐 Web</span>
        )}
      </div>

      {isElectron && (
        <div className="titlebar-controls">
          <button className="ctrl-btn minimize" onClick={() => window.electronAPI.minimize()} title="Réduire"><span /></button>
          <button className="ctrl-btn maximize" onClick={() => window.electronAPI.maximize()} title="Agrandir"><span /></button>
          <button className="ctrl-btn close"    onClick={() => window.electronAPI.close()}    title="Fermer"><span /></button>
        </div>
      )}
    </div>
  );
}
