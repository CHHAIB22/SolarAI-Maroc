import React, { useState, useRef, useCallback, useEffect } from 'react';
import './TreePanel.css';

// ── Decision tree structure (matches the trained model logic) ────
// Built manually to mirror the scikit-learn tree from donnees_energie.sql
const TREE = {
  id: 0,
  feature: "Vitesse du Vent",
  threshold: "≤ 6.25 m/s",
  gini: 0.75, samples: 40,
  left: {
    id: 1,
    feature: "Eau",
    threshold: "= Non",
    gini: 0.667, samples: 30,
    left: {
      id: 3,
      feature: "Irradiation Solaire",
      threshold: "≤ 1750 kWh",
      gini: 0.5, samples: 20,
      left: {
        id: 7, leaf: true, classe: "Energie Biomasse",
        gini: 0.0, samples: 10, value: [10,0,0,0], purity: 100
      },
      right: {
        id: 8, leaf: true, classe: "Energie Solaire PV",
        gini: 0.0, samples: 10, value: [0,0,10,0], purity: 100
      }
    },
    right: {
      id: 4, leaf: true, classe: "Hydroelectricite",
      gini: 0.0, samples: 10, value: [0,0,0,10], purity: 100
    }
  },
  right: {
    id: 2, leaf: true, classe: "Energie Eolienne",
    gini: 0.0, samples: 10, value: [0,10,0,0], purity: 100
  }
};

const ENERGIE_COLORS = {
  "Energie Solaire PV": "#f59e0b",
  "Energie Eolienne":   "#38bdf8",
  "Hydroelectricite":   "#3b82f6",
  "Energie Biomasse":   "#22c55e",
};
const ENERGIE_ICONS = {
  "Energie Solaire PV": "☀",
  "Energie Eolienne":   "◌",
  "Hydroelectricite":   "◈",
  "Energie Biomasse":   "✿",
};

// ── Layout: assign x,y to each node ─────────────────────────────
function layoutTree(node, depth = 0, counter = { val: 0 }) {
  if (!node) return null;
  const n = { ...node };
  if (n.leaf) {
    n.x = counter.val * 220;
    n.y = depth * 160;
    counter.val++;
  } else {
    n.left  = layoutTree(n.left,  depth + 1, counter);
    n.right = layoutTree(n.right, depth + 1, counter);
    n.x = (n.left.x + n.right.x) / 2;
    n.y = depth * 160;
  }
  return n;
}

function collectNodes(node, nodes = [], edges = []) {
  if (!node) return;
  nodes.push(node);
  if (node.left)  { edges.push({ from: node, to: node.left,  label: "Vrai" });  collectNodes(node.left,  nodes, edges); }
  if (node.right) { edges.push({ from: node, to: node.right, label: "Faux" }); collectNodes(node.right, nodes, edges); }
}

export default function TreePanel({ lastResult }) {
  const [hoveredId, setHoveredId]   = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [pan,  setPan]  = useState({ x: 60, y: 40 });
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef(null);
  const svgRef = useRef(null);

  const laid  = layoutTree(TREE);
  const nodes = [], edges = [];
  collectNodes(laid, nodes, edges);

  // Highlight path of last prediction
  const highlightPath = useRef(new Set());
  useEffect(() => {
    if (!lastResult) return;
    const p = lastResult.params;
    const path = new Set();
    path.add(0);
    // Node 0: vent <= 6.25?
    if (p.vent <= 6.25) {
      path.add(1);
      // Node 1: eau == Non (encoded 0 <= 0.5)?
      if (p.eau === 'Non') {
        path.add(3);
        if (p.solaire <= 1750) path.add(7); else path.add(8);
      } else {
        path.add(4);
      }
    } else {
      path.add(2);
    }
    highlightPath.current = path;
  }, [lastResult]);

  // SVG dimensions
  const W = Math.max(...nodes.map(n => n.x)) + 320;
  const H = Math.max(...nodes.map(n => n.y)) + 200;

  // Selected node info
  const selectedNode = nodes.find(n => n.id === selectedId);

  // Pan/zoom handlers
  const onMouseDown = useCallback((e) => {
    if (e.target.closest('.tree-node-g')) return;
    setDragging(true);
    dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  }, [pan]);
  const onMouseMove = useCallback((e) => {
    if (!dragging) return;
    setPan({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
  }, [dragging]);
  const onMouseUp = useCallback(() => setDragging(false), []);
  const onWheel = useCallback((e) => {
    e.preventDefault();
    setZoom(z => Math.max(0.3, Math.min(2.5, z - e.deltaY * 0.001)));
  }, []);

  const isHighlighted = (id) => highlightPath.current.has(id);

  return (
    <div className="tree-panel fade-in">
      {/* Header */}
      <div className="tree-header">
        <div>
          <h2 className="tree-title">Arbre de Décision</h2>
          <p className="tree-sub">Modèle interactif — cliquez sur un nœud pour les détails · glissez pour naviguer</p>
        </div>
        <div className="tree-actions">
          <button className="tree-btn" onClick={() => setZoom(z => Math.min(z + 0.2, 2.5))}>＋ Zoom</button>
          <button className="tree-btn" onClick={() => setZoom(z => Math.max(z - 0.2, 0.3))}>－ Zoom</button>
          <button className="tree-btn" onClick={() => { setZoom(1); setPan({ x: 60, y: 40 }); }}>Réinitialiser</button>
          {lastResult && (
            <div className="tree-prediction-badge" style={{ borderColor: ENERGIE_COLORS[lastResult.recommandation] + '60', color: ENERGIE_COLORS[lastResult.recommandation] }}>
              {ENERGIE_ICONS[lastResult.recommandation]} Dernière: {lastResult.recommandation} · {lastResult.confiance}%
            </div>
          )}
        </div>
      </div>

      <div className="tree-body">
        {/* SVG Tree */}
        <div className="tree-svg-wrap"
          onMouseDown={onMouseDown} onMouseMove={onMouseMove}
          onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
          onWheel={onWheel}
          style={{ cursor: dragging ? 'grabbing' : 'grab' }}
          ref={svgRef}
        >
          <svg width={W} height={H}
            style={{ transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`, transformOrigin: '0 0', transition: dragging ? 'none' : 'transform 0.05s' }}
          >
            <defs>
              {Object.entries(ENERGIE_COLORS).map(([cls, col]) => (
                <radialGradient key={cls} id={`grad-${cls.replace(/\s/g,'')}`} cx="50%" cy="40%" r="60%">
                  <stop offset="0%" stopColor={col} stopOpacity="0.25"/>
                  <stop offset="100%" stopColor={col} stopOpacity="0.06"/>
                </radialGradient>
              ))}
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>

            {/* Edges */}
            {edges.map((e, i) => {
              const onPath = isHighlighted(e.from.id) && isHighlighted(e.to.id);
              return (
                <g key={i}>
                  <path
                    d={`M${e.from.x + 100},${e.from.y + 68} C${e.from.x + 100},${e.from.y + 110} ${e.to.x + 100},${e.to.y - 10} ${e.to.x + 100},${e.to.y}`}
                    fill="none"
                    stroke={onPath ? '#00d4ff' : '#1e3050'}
                    strokeWidth={onPath ? 2.5 : 1.5}
                    strokeDasharray={onPath ? 'none' : '4,3'}
                    opacity={onPath ? 1 : 0.6}
                  />
                  <text
                    x={(e.from.x + e.to.x) / 2 + 108}
                    y={(e.from.y + e.to.y) / 2 + 34}
                    fontSize="11" fill={onPath ? '#00d4ff' : '#4a6380'}
                    textAnchor="middle" fontFamily="JetBrains Mono,monospace" fontWeight={onPath ? '700' : '400'}
                  >
                    {e.label}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {nodes.map(node => {
              const isSelected  = selectedId === node.id;
              const isHovered   = hoveredId  === node.id;
              const onPath      = isHighlighted(node.id);
              const cls         = node.classe;
              const col         = cls ? ENERGIE_COLORS[cls] : '#00d4ff';
              const NW = 200, NH = node.leaf ? 90 : 80;

              return (
                <g key={node.id} className="tree-node-g"
                  transform={`translate(${node.x}, ${node.y})`}
                  onClick={() => setSelectedId(selectedId === node.id ? null : node.id)}
                  onMouseEnter={() => setHoveredId(node.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  style={{ cursor: 'pointer' }}
                  filter={onPath ? 'url(#glow)' : 'none'}
                >
                  {/* Node box */}
                  <rect
                    x={0} y={0} width={NW} height={NH} rx={12}
                    fill={cls ? `url(#grad-${cls.replace(/\s/g,'')})` : '#0f1929'}
                    stroke={isSelected ? '#fff' : onPath ? col : isHovered ? col + '80' : '#1e3050'}
                    strokeWidth={isSelected ? 2.5 : onPath ? 2 : 1}
                  />

                  {/* Top color bar */}
                  <rect x={0} y={0} width={NW} height={4} rx={12}
                    fill={col} opacity={node.leaf ? 0.9 : 0.4}
                  />

                  {node.leaf ? (
                    // ── Leaf node ──
                    <>
                      <text x={NW/2} y={26} textAnchor="middle" fontSize="22" fill={col}>{ENERGIE_ICONS[cls]}</text>
                      <text x={NW/2} y={46} textAnchor="middle" fontSize="10" fill={col} fontFamily="Space Grotesk,sans-serif" fontWeight="700">
                        {cls?.replace('Energie ','').replace('Hydroelectricite','Hydro')}
                      </text>
                      <text x={NW/2} y={60} textAnchor="middle" fontSize="9" fill="#4a6380" fontFamily="JetBrains Mono,monospace">
                        gini={node.gini.toFixed(1)} · n={node.samples}
                      </text>
                      {/* Purity bar */}
                      <rect x={20} y={70} width={NW-40} height={5} rx={3} fill="#1e3050"/>
                      <rect x={20} y={70} width={(NW-40)*(node.purity/100)} height={5} rx={3} fill={col} opacity={0.8}/>
                      <text x={NW/2} y={84} textAnchor="middle" fontSize="8" fill={col} fontFamily="JetBrains Mono,monospace">{node.purity}% pur</text>
                    </>
                  ) : (
                    // ── Decision node ──
                    <>
                      <text x={NW/2} y={22} textAnchor="middle" fontSize="9.5" fill="#8ba3c0" fontFamily="Space Grotesk,sans-serif" fontWeight="600" letterSpacing="0.05em">
                        {node.feature}
                      </text>
                      <rect x={10} y={28} width={NW-20} height={1} fill="#1e3050"/>
                      <text x={NW/2} y={46} textAnchor="middle" fontSize="11" fill={onPath ? '#00d4ff' : '#e8f4ff'} fontFamily="JetBrains Mono,monospace" fontWeight="600">
                        {node.threshold}
                      </text>
                      <text x={NW/2} y={62} textAnchor="middle" fontSize="9" fill="#4a6380" fontFamily="JetBrains Mono,monospace">
                        gini={node.gini.toFixed(3)} · n={node.samples}
                      </text>
                    </>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        {/* Detail panel */}
        {selectedNode && (
          <div className="tree-detail" style={{ '--det-color': selectedNode.classe ? ENERGIE_COLORS[selectedNode.classe] : '#00d4ff' }}>
            <button className="tree-detail-close" onClick={() => setSelectedId(null)}>✕</button>
            <div className="tree-detail-title">
              {selectedNode.leaf
                ? <><span>{ENERGIE_ICONS[selectedNode.classe]}</span> {selectedNode.classe}</>
                : <>🔀 Nœud de Décision</>
              }
            </div>
            {selectedNode.leaf ? (
              <div className="tree-detail-grid">
                <div className="det-row"><span>Classe</span><span style={{ color: ENERGIE_COLORS[selectedNode.classe] }}>{selectedNode.classe}</span></div>
                <div className="det-row"><span>Echantillons</span><span>{selectedNode.samples}</span></div>
                <div className="det-row"><span>Impureté Gini</span><span>{selectedNode.gini.toFixed(1)}</span></div>
                <div className="det-row"><span>Pureté</span><span style={{ color: ENERGIE_COLORS[selectedNode.classe] }}>{selectedNode.purity}%</span></div>
                <div className="det-row"><span>Valeur</span><span>[{selectedNode.value.join(', ')}]</span></div>
              </div>
            ) : (
              <div className="tree-detail-grid">
                <div className="det-row"><span>Variable</span><span>{selectedNode.feature}</span></div>
                <div className="det-row"><span>Seuil</span><span style={{ color: '#00d4ff' }}>{selectedNode.threshold}</span></div>
                <div className="det-row"><span>Echantillons</span><span>{selectedNode.samples}</span></div>
                <div className="det-row"><span>Impureté Gini</span><span>{selectedNode.gini.toFixed(3)}</span></div>
                <div className="det-row"><span>Branche gauche</span><span style={{ color: '#22c55e' }}>Vrai ✓</span></div>
                <div className="det-row"><span>Branche droite</span><span style={{ color: '#ef4444' }}>Faux ✗</span></div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="tree-footer">
        {[
          ['Algorithme', 'DecisionTreeClassifier'],
          ['Profondeur', '5'],
          ['Echantillons', '40'],
          ['Classes', '4'],
          ['Critère', 'Gini'],
        ].map(([label, val]) => (
          <div key={label} className="tree-stat">
            <span className="stat-label">{label}</span>
            <span className="stat-val">{val}</span>
          </div>
        ))}
        <div className="tree-stat" style={{ marginLeft: 'auto' }}>
          <span className="stat-label">Zoom</span>
          <span className="stat-val accent">{Math.round(zoom * 100)}%</span>
        </div>
        {lastResult && (
          <div className="tree-stat highlight" style={{ '--det-color': ENERGIE_COLORS[lastResult.recommandation] }}>
            <span className="stat-label">Chemin actif</span>
            <span className="stat-val" style={{ color: ENERGIE_COLORS[lastResult.recommandation] }}>{lastResult.recommandation}</span>
          </div>
        )}
      </div>
    </div>
  );
}
