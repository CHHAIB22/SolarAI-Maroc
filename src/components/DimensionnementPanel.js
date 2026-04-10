import React, { useState, useRef, useEffect } from 'react';
import VILLES_MAROC from '../data/villesMaroc';
import './DimensionnementPanel.css';

// ── Calcul engine ────────────────────────────────────────────────
function calculerSysteme(inputs) {
  const {
    consommation_wh, tension_systeme, autonomie_jours, irradiation,
    puissance_panneau, capacite_batterie, pertes_systeme, profondeur_decharge,
  } = inputs;
  const perte_coeff = 1 + pertes_systeme / 100;
  const energie_necessaire_wc = (consommation_wh * perte_coeff) / irradiation;
  const nb_panneaux = Math.ceil(energie_necessaire_wc / puissance_panneau);
  const puissance_crete_totale = nb_panneaux * puissance_panneau;
  const energie_stockage = consommation_wh * autonomie_jours * perte_coeff;
  const capacite_ah_totale = energie_stockage / (tension_systeme * (profondeur_decharge / 100));
  const nb_batteries = Math.ceil(capacite_ah_totale / capacite_batterie);
  const capacite_reelle = nb_batteries * capacite_batterie;
  const courant_panneaux = puissance_crete_totale / tension_systeme;
  const courant_regulateur = Math.ceil(courant_panneaux * 1.25);
  const type_regulateur = courant_regulateur > 30 ? 'MPPT' : 'PWM';
  const puissance_onduleur_raw = Math.ceil(consommation_wh / 4 * 1.3);
  const puissance_onduleur = [300,500,700,1000,1500,2000,3000,5000].find(p => p >= puissance_onduleur_raw) || puissance_onduleur_raw;
  const sections_std = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50];
  const sigma = 56;
  const deltaU = tension_systeme * 0.03;
  const sec = (I, L) => sections_std.find(s => s >= (I * L * 2) / (sigma * deltaU)) || 50;
  return {
    nb_panneaux, puissance_panneau, puissance_crete_totale,
    energie_necessaire_wc: Math.round(energie_necessaire_wc),
    nb_batteries, capacite_batterie, capacite_ah_totale: Math.round(capacite_ah_totale),
    capacite_reelle, energie_stockage: Math.round(energie_stockage), tension_systeme,
    courant_regulateur, type_regulateur,
    puissance_onduleur,
    type_onduleur: puissance_onduleur >= 1000 ? 'Onde sinusoïdale pure' : 'Onde modifiée',
    section_pv:  sec(puissance_crete_totale / tension_systeme, 10),
    section_bat: sec(capacite_ah_totale / 5, 2),
    section_ond: sec(puissance_onduleur / tension_systeme, 1.5),
  };
}

const APPAREILS = [
  { nom:'Ampoule LED',       puissance:10,   icon:'💡' },
  { nom:'Téléviseur LED',    puissance:80,   icon:'📺' },
  { nom:'Réfrigérateur',     puissance:150,  icon:'🧊' },
  { nom:'Ventilateur',       puissance:60,   icon:'🌀' },
  { nom:'Chargeur portable', puissance:15,   icon:'📱' },
  { nom:'Ordinateur',        puissance:150,  icon:'💻' },
  { nom:'Climatiseur',       puissance:1200, icon:'❄️'  },
  { nom:'Machine à laver',   puissance:500,  icon:'🫧' },
  { nom:'Pompe à eau',       puissance:750,  icon:'💧' },
  { nom:'Four micro-ondes',  puissance:800,  icon:'📦' },
];

const DEFAULTS = { tension_systeme:24, autonomie_jours:2, puissance_panneau:400, capacite_batterie:200, pertes_systeme:25, profondeur_decharge:80 };

export default function DimensionnementPanel() {
  const [ville,     setVille]     = useState(VILLES_MAROC[0]);
  const [search,    setSearch]    = useState('');
  const [dropOpen,  setDropOpen]  = useState(false);
  const [appareils, setAppareils] = useState([
    { nom:'Ampoule LED', puissance:10, heures:5, icon:'💡' },
    { nom:'Téléviseur LED', puissance:80, heures:4, icon:'📺' },
    { nom:'Chargeur portable', puissance:15, heures:3, icon:'📱' },
  ]);
  const [inputs,  setInputs]  = useState(DEFAULTS);
  const [resultat, setResultat] = useState(null);
  const dropRef = useRef(null);

  useEffect(() => {
    const h = e => { if (dropRef.current && !dropRef.current.contains(e.target)) setDropOpen(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  const irr   = +(ville.solaire / 365).toFixed(2);
  const conso = appareils.reduce((s, a) => s + a.puissance * a.heures, 0);
  const set   = (k, v) => setInputs(i => ({ ...i, [k]: v }));

  const calculer = () => {
    if (conso === 0) return;
    setResultat(calculerSysteme({ consommation_wh: conso, irradiation: irr, ...inputs }));
  };

  const query = search.toLowerCase();
  const filtrees = VILLES_MAROC.filter(v => v.nom.toLowerCase().includes(query) || v.region.toLowerCase().includes(query));

  return (
    <div className="dim-panel fade-in">

      {/* HEADER */}
      <div className="dim-header">
        <div>
          <h2 className="dim-title">Dimensionnement Solaire</h2>
          <p className="dim-sub">Calculez votre installation photovoltaïque autonome</p>
        </div>
        <div className="dim-hstats">
          <div className="dim-hstat"><span style={{fontFamily:'monospace',fontSize:18,fontWeight:700}}>{conso}</span><span className="dim-hstat-u">Wh/jour</span></div>
          <div className="dim-hsep"/>
          <div className="dim-hstat"><span style={{fontFamily:'monospace',fontSize:18,fontWeight:700,color:'#f59e0b'}}>{irr}</span><span className="dim-hstat-u">kWh/m²/j</span></div>
          <div className="dim-hsep"/>
          <div className="dim-hstat"><span style={{fontFamily:'monospace',fontSize:14,fontWeight:700,color:'#00d4ff'}}>{ville.nom}</span><span className="dim-hstat-u">🇲🇦</span></div>
        </div>
      </div>

      {/* BODY */}
      <div className="dim-body">

        {/* LEFT */}
        <div className="dim-left">

          {/* Ville */}
          <div className="dim-section">
            <div className="dim-sec-title"><span>📍</span> Localisation</div>
            <div className="dim-city" ref={dropRef}>
              <button className="dim-city-btn" onClick={() => setDropOpen(o => !o)}>
                <span>🇲🇦</span>
                <span style={{flex:1,textAlign:'left'}}>{ville.nom}</span>
                <span style={{color:'#f59e0b',fontFamily:'monospace',fontSize:11}}>{irr} kWh/m²/j</span>
                <span style={{color:'var(--text-muted)',fontSize:11}}>{dropOpen ? '▲' : '▾'}</span>
              </button>
              {dropOpen && (
                <div className="dim-dropdown">
                  <input className="dim-search" placeholder="Rechercher une ville..." autoFocus
                    value={search} onChange={e => setSearch(e.target.value)} />
                  <div className="dim-vlist">
                    {filtrees.map(v => (
                      <button key={v.nom} className={`dim-vitem ${v.nom === ville.nom ? 'active' : ''}`}
                        onClick={() => { setVille(v); setDropOpen(false); setSearch(''); }}>
                        <span className="dvi-nom">{v.nom}</span>
                        <span className="dvi-reg">{v.region}</span>
                        <span className="dvi-sol">{(v.solaire/365).toFixed(2)}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Appareils */}
          <div className="dim-section">
            <div className="dim-sec-title"><span>⚡</span> Consommation Journalière</div>
            {appareils.map((a, i) => (
              <div key={i} className="dim-app-row">
                <span className="dim-app-icon">{a.icon}</span>
                <span className="dim-app-nom">{a.nom}</span>
                <input type="number" className="dim-app-inp" value={a.puissance} min={1} max={5000}
                  onChange={e => setAppareils(p => p.map((x,j) => j===i ? {...x, puissance:+e.target.value} : x))} />
                <span className="dim-app-u">W ×</span>
                <input type="number" className="dim-app-inp" value={a.heures} min={0.5} max={24} step={0.5}
                  onChange={e => setAppareils(p => p.map((x,j) => j===i ? {...x, heures:+e.target.value} : x))} />
                <span className="dim-app-u">h</span>
                <span className="dim-app-tot">= {a.puissance * a.heures} Wh</span>
                <button className="dim-app-del" onClick={() => setAppareils(p => p.filter((_,j) => j!==i))}>✕</button>
              </div>
            ))}
            <div className="dim-conso-total">
              <span>Total consommation</span>
              <span style={{color:'#f59e0b',fontWeight:700,fontFamily:'monospace'}}>{conso} Wh/jour</span>
            </div>
            <div className="dim-catalogue-grid">
              {APPAREILS.filter(a => !appareils.find(x => x.nom === a.nom)).map(a => (
                <button key={a.nom} className="dim-cat-btn"
                  onClick={() => setAppareils(p => [...p, { ...a, heures: 4 }])}>
                  {a.icon} {a.nom} <span className="dim-cat-w">{a.puissance}W</span>
                </button>
              ))}
            </div>
          </div>

          {/* Paramètres */}
          <div className="dim-section">
            <div className="dim-sec-title"><span>⚙</span> Paramètres Système</div>
            <div className="dim-cfg-grid">
              <CfgSelect label="Tension"     value={inputs.tension_systeme}    unit="V"  opts={[12,24,48]}                  onChange={v => set('tension_systeme', v)} />
              <CfgStep   label="Autonomie"   value={inputs.autonomie_jours}    unit="j"  min={1}  max={7}                   onChange={v => set('autonomie_jours', v)} />
              <CfgSelect label="Panneau"     value={inputs.puissance_panneau}  unit="Wc" opts={[100,200,300,400,500,600]}   onChange={v => set('puissance_panneau', v)} />
              <CfgSelect label="Batterie"    value={inputs.capacite_batterie}  unit="Ah" opts={[50,100,150,200,250,300]}    onChange={v => set('capacite_batterie', v)} />
              <CfgStep   label="Pertes"      value={inputs.pertes_systeme}     unit="%"  min={10} max={40} step={5}         onChange={v => set('pertes_systeme', v)} />
              <CfgSelect label="Prof. déch." value={inputs.profondeur_decharge} unit="%" opts={[50,60,70,80]}               onChange={v => set('profondeur_decharge', v)} />
            </div>
          </div>

          <button className="dim-calc-btn" onClick={calculer} disabled={conso === 0}>
            ▶ &nbsp; Calculer le Dimensionnement
          </button>
        </div>

        {/* RIGHT */}
        <div className="dim-right">
          {!resultat ? (
            <div className="dim-empty">
              <div style={{fontSize:56,opacity:0.15,animation:'floatUp 4s ease-in-out infinite'}}>☀</div>
              <p style={{color:'var(--text-secondary)'}}>Configurez et cliquez sur Calculer</p>
              <p style={{fontSize:12,color:'var(--text-muted)'}}>Ajoutez vos appareils et choisissez une ville</p>
            </div>
          ) : (
            <div className="dim-results">

              {/* Hero */}
              <div className="dim-hero">
                <div>
                  <div className="dim-hero-lbl">INSTALLATION POUR</div>
                  <div className="dim-hero-city">🇲🇦 {ville.nom}</div>
                  <div className="dim-hero-info">{conso} Wh/j · {inputs.autonomie_jours}j autonomie · {irr} kWh/m²/j</div>
                </div>
                <div style={{textAlign:'right',flexShrink:0}}>
                  <div className="dim-hero-val">{resultat.puissance_crete_totale}<span className="dim-hero-vu"> Wc</span></div>
                  <div className="dim-hero-vsub">Puissance crête totale</div>
                </div>
              </div>

              {/* 4 cards */}
              <div className="dim-cards">
                <RCard color="#f59e0b" icon="☀"  title="Panneaux Solaires" main={resultat.nb_panneaux}       unit="panneaux"
                  rows={[['Unitaire',`${resultat.puissance_panneau} Wc`],['Crête totale',`${resultat.puissance_crete_totale} Wc`],['Irradiation',`${irr} kWh/m²/j`]]} />
                <RCard color="#a855f7" icon="🔋" title="Batteries"          main={resultat.nb_batteries}      unit="batteries"
                  rows={[['Capacité unit.',`${resultat.capacite_batterie} Ah`],['Total',`${resultat.capacite_reelle} Ah`],['Énergie',`${resultat.energie_stockage} Wh`]]} />
                <RCard color="#00d4ff" icon="⚡" title="Régulateur"         main={resultat.courant_regulateur} unit="A" badge={resultat.type_regulateur}
                  rows={[['Type',resultat.type_regulateur],['Courant max',`${resultat.courant_regulateur} A`],['Tension',`${inputs.tension_systeme} V`]]} />
                <RCard color="#22c55e" icon="🔌" title="Onduleur"           main={resultat.puissance_onduleur} unit="W" badge={resultat.puissance_onduleur>=1000?'Pur':'Modifié'}
                  rows={[['Puissance',`${resultat.puissance_onduleur} W`],['Signal',resultat.type_onduleur],['Sortie','220 V AC']]} />
              </div>

              {/* Câbles */}
              <div className="dim-cables">
                <div className="dim-cables-hdr">🔗 Section des Câbles <span className="dim-cables-sub">Cuivre σ=56 S/m · ΔU=3%</span></div>
                <CRow color="#f59e0b" label="Panneaux → Régulateur" sec={resultat.section_pv}  L="10m" />
                <CRow color="#a855f7" label="Régulateur → Batterie"  sec={resultat.section_bat} L="2m" />
                <CRow color="#22c55e" label="Batterie → Onduleur"    sec={resultat.section_ond} L="1.5m" />
              </div>

              {/* Schéma */}
              <div className="dim-schema">
                <div className="dim-schema-lbl">Schéma de Câblage</div>
                <svg width="660" height="165" style={{display:'block',margin:'0 auto',maxWidth:'100%'}}>
                  <rect width="660" height="165" fill="#080e1c" rx="10"/>
                  {[
                    {x1:130,x2:183,y:95,label:`${resultat.section_pv}mm²`,  color:'#f59e0b'},
                    {x1:293,x2:348,y:95,label:`${resultat.section_bat}mm²`, color:'#a855f7'},
                    {x1:458,x2:513,y:95,label:`${resultat.section_ond}mm²`, color:'#22c55e'},
                  ].map((a,i) => (
                    <g key={i}>
                      <line x1={a.x1} y1={a.y} x2={a.x2} y2={a.y} stroke={a.color} strokeWidth="2" strokeDasharray="5,3"/>
                      <polygon points={`${a.x2},${a.y} ${a.x2-7},${a.y-4} ${a.x2-7},${a.y+4}`} fill={a.color}/>
                      <text x={(a.x1+a.x2)/2} y={a.y-8} textAnchor="middle" fontSize="9" fill={a.color} fontFamily="monospace" fontWeight="700">{a.label}</text>
                    </g>
                  ))}
                  {[
                    {x:18, label:'☀ Panneaux',   sub:`${resultat.nb_panneaux}×${resultat.puissance_panneau}Wc`,  color:'#f59e0b'},
                    {x:183,label:'⚡ Régulateur', sub:`${resultat.courant_regulateur}A ${resultat.type_regulateur}`, color:'#00d4ff'},
                    {x:348,label:'🔋 Batteries',  sub:`${resultat.nb_batteries}×${resultat.capacite_batterie}Ah`,  color:'#a855f7'},
                    {x:513,label:'🔌 Onduleur',   sub:`${resultat.puissance_onduleur}W`,                           color:'#22c55e'},
                  ].map((b,i) => (
                    <g key={i}>
                      <rect x={b.x} y={68} width={145} height={52} rx="8" fill="#0d1829" stroke={b.color} strokeWidth="1.5" strokeOpacity="0.5"/>
                      <rect x={b.x} y={68} width={145} height={3}  rx="2" fill={b.color} opacity="0.7"/>
                      <text x={b.x+72} y={90}  textAnchor="middle" fontSize="12" fill={b.color} fontFamily="sans-serif" fontWeight="700">{b.label}</text>
                      <text x={b.x+72} y={108} textAnchor="middle" fontSize="9"  fill="#6b88a8" fontFamily="monospace">{b.sub}</text>
                    </g>
                  ))}
                  <text x={478} y={152} fontSize="9" fill="#4a6380" fontFamily="monospace">DC {inputs.tension_systeme}V</text>
                  <text x={572} y={152} fontSize="9" fill="#22c55e" fontFamily="monospace">AC 220V</text>
                </svg>
              </div>

            </div>
          )}
        </div>

      </div>
    </div>
  );
}

function CfgSelect({ label, value, unit, opts, onChange }) {
  return (
    <div className="dim-cfg-field">
      <div className="dim-cfg-lbl">{label}</div>
      <select className="dim-cfg-sel" value={value} onChange={e => onChange(+e.target.value)}>
        {opts.map(o => <option key={o} value={o}>{o} {unit}</option>)}
      </select>
    </div>
  );
}
function CfgStep({ label, value, unit, min, max, step = 1, onChange }) {
  return (
    <div className="dim-cfg-field">
      <div className="dim-cfg-lbl">{label}</div>
      <div className="dim-cfg-step">
        <button onClick={() => onChange(Math.max(min, value - step))}>−</button>
        <span>{value}<small> {unit}</small></span>
        <button onClick={() => onChange(Math.min(max, value + step))}>+</button>
      </div>
    </div>
  );
}
function RCard({ color, icon, title, main, unit, rows, badge }) {
  return (
    <div className="dim-rcard" style={{'--rc': color}}>
      <div className="dim-rcard-bar"/>
      <div className="dim-rcard-hd">
        <span style={{fontSize:18}}>{icon}</span>
        <span className="dim-rcard-title">{title}</span>
        {badge && <span className="dim-rcard-badge" style={{color, borderColor:color+'50', background:color+'15'}}>{badge}</span>}
      </div>
      <div style={{display:'flex',alignItems:'baseline',gap:6,marginBottom:10}}>
        <span style={{fontFamily:'monospace',fontSize:38,fontWeight:700,color,lineHeight:1}}>{main}</span>
        <span style={{fontSize:13,color:'var(--text-muted)'}}>{unit}</span>
      </div>
      {rows.map(([k,v],i) => (
        <div key={i} className="dim-rcard-row">
          <span>{k}</span><span>{v}</span>
        </div>
      ))}
    </div>
  );
}
function CRow({ color, label, sec, L }) {
  return (
    <div className="dim-crow">
      <div style={{display:'flex',alignItems:'center',width:40}}>
        <div style={{flex:1,height:2,background:color,opacity:0.5}}/>
        <div style={{width:8,height:8,borderRadius:'50%',background:color,flexShrink:0}}/>
      </div>
      <div style={{flex:1}}>
        <div style={{fontSize:12,fontWeight:500,color:'var(--text-primary)'}}>{label}</div>
        <div style={{fontSize:10,color:'var(--text-muted)',fontFamily:'monospace'}}>L = {L}</div>
      </div>
      <div className="dim-crow-sec" style={{color, borderColor:color+'50', background:color+'12'}}>{sec} mm²</div>
    </div>
  );
}
