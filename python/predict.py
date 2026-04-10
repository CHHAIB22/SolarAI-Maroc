"""
predict.py - Appele par Electron via IPC
Usage: python predict.py <solaire> <vent> <eau> <biomasse> <terrain> <temp> <db_path> <nom_site>
"""
import sys
import json
import os
import sqlite3

# ── Args ────────────────────────────────────────────────────────
if len(sys.argv) < 8:
    print(json.dumps({"error": "Arguments manquants. Requis: solaire vent eau biomasse terrain temp db_path"}))
    sys.exit(1)

solaire   = float(sys.argv[1])
vent      = float(sys.argv[2])
eau       = sys.argv[3]
biomasse  = sys.argv[4]
terrain   = sys.argv[5]
temp      = float(sys.argv[6])
db_path   = sys.argv[7]
nom_site  = sys.argv[8] if len(sys.argv) > 8 else "Site Utilisateur"

SQL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donnees_energie.sql")

# ── Imports ─────────────────────────────────────────────────────
try:
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    print(json.dumps({"error": f"Module manquant: {e}\nInstallez: pip install scikit-learn numpy"}))
    sys.exit(1)

# ── Init DB ─────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='donnees_entrainement'")
    if cur.fetchone():
        cur.execute("SELECT COUNT(*) FROM donnees_entrainement")
        if cur.fetchone()[0] > 0:
            conn.close()
            return
    if os.path.exists(SQL_FILE):
        with open(SQL_FILE, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    conn.close()

init_db()

# ── Load data ───────────────────────────────────────────────────
conn = sqlite3.connect(db_path)
cur  = conn.cursor()
cur.execute("""
    SELECT irradiation_solaire, vitesse_vent, disponibilite_eau,
           disponibilite_biomasse, disponibilite_terrain,
           temperature_moyenne, energie_recommandee
    FROM donnees_entrainement ORDER BY id
""")
rows = cur.fetchall()
conn.close()

if not rows:
    print(json.dumps({"error": "Base de donnees vide. Verifiez donnees_energie.sql"}))
    sys.exit(1)

# ── Encoders ────────────────────────────────────────────────────
eau_enc      = LabelEncoder().fit(["Non", "Oui"])
biomasse_enc = LabelEncoder().fit(["Faible", "Moyen", "Eleve"])
terrain_enc  = LabelEncoder().fit(["Petit", "Moyen", "Grand"])

COULEURS = {
    "Energie Solaire PV": "#f59e0b",
    "Energie Eolienne":   "#38bdf8",
    "Hydroelectricite":   "#3b82f6",
    "Energie Biomasse":   "#22c55e",
}
ICONES = {
    "Energie Solaire PV": "sun",
    "Energie Eolienne":   "wind",
    "Hydroelectricite":   "droplets",
    "Energie Biomasse":   "leaf",
}

def encode(row):
    s, v, e, b, t, tmp, _ = row
    return [float(s), float(v),
            float(eau_enc.transform([e])[0]),
            float(biomasse_enc.transform([b])[0]),
            float(terrain_enc.transform([t])[0]),
            float(tmp)]

X = np.array([encode(r) for r in rows])
y = np.array([r[6] for r in rows])

# ── Train ───────────────────────────────────────────────────────
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=3,
                              min_samples_leaf=2, random_state=42)
clf.fit(X, y)

# ── Validate input values ───────────────────────────────────────
if eau not in ["Oui", "Non"]:
    print(json.dumps({"error": f"Valeur eau invalide: '{eau}'. Doit etre Oui ou Non"}))
    sys.exit(1)
if biomasse not in ["Faible", "Moyen", "Eleve"]:
    print(json.dumps({"error": f"Valeur biomasse invalide: '{biomasse}'"}))
    sys.exit(1)
if terrain not in ["Petit", "Moyen", "Grand"]:
    print(json.dumps({"error": f"Valeur terrain invalide: '{terrain}'"}))
    sys.exit(1)

# ── Predict ─────────────────────────────────────────────────────
x_in = np.array([[
    solaire, vent,
    eau_enc.transform([eau])[0],
    biomasse_enc.transform([biomasse])[0],
    terrain_enc.transform([terrain])[0],
    temp
]])

rec       = clf.predict(x_in)[0]
proba_arr = clf.predict_proba(x_in)[0]
probs     = {k: round(float(v)*100, 1) for k, v in zip(clf.classes_, proba_arr)}
confiance = probs[rec]

# ── Save prediction ─────────────────────────────────────────────
try:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_prediction TEXT DEFAULT (datetime('now','localtime')),
        nom_site TEXT, irradiation_solaire REAL, vitesse_vent REAL,
        disponibilite_eau TEXT, disponibilite_biomasse TEXT,
        disponibilite_terrain TEXT, temperature_moyenne REAL,
        energie_recommandee TEXT, confiance_pct REAL)""")
    cur.execute("""INSERT INTO predictions
        (nom_site,irradiation_solaire,vitesse_vent,disponibilite_eau,
         disponibilite_biomasse,disponibilite_terrain,temperature_moyenne,
         energie_recommandee,confiance_pct)
        VALUES(?,?,?,?,?,?,?,?,?)""",
        (nom_site,solaire,vent,eau,biomasse,terrain,temp,rec,confiance))
    conn.commit()
    conn.close()
except Exception:
    pass

# ── Decision path ───────────────────────────────────────────────
FEAT_NAMES = ["Irradiation Solaire","Vitesse du Vent","Eau","Biomasse","Terrain","Temperature"]
def get_path(tree, x):
    node, path = 0, []
    while tree.feature[node] != -2:
        feat  = FEAT_NAMES[tree.feature[node]]
        thresh = round(float(tree.threshold[node]), 2)
        val    = float(x[0][tree.feature[node]])
        left   = val <= tree.threshold[node]
        path.append({"feature": feat, "threshold": thresh,
                     "value": round(val,2), "direction": "<=" if left else ">"})
        node = tree.children_left[node] if left else tree.children_right[node]
    return path

# ── Output ──────────────────────────────────────────────────────
result = {
    "recommandation": rec,
    "confiance":      confiance,
    "couleur":        COULEURS.get(rec, "#ffffff"),
    "icone":          ICONES.get(rec, "zap"),
    "probabilites": [
        {"energie": k, "prob": v, "couleur": COULEURS.get(k, "#aaa")}
        for k, v in sorted(probs.items(), key=lambda x: -x[1])
    ],
    "decisionPath": get_path(clf.tree_, x_in),
    "params": {
        "nomSite": nom_site, "solaire": solaire, "vent": vent,
        "eau": eau, "biomasse": biomasse, "terrain": terrain,
        "temperature": temp,
    }
}

print(json.dumps(result, ensure_ascii=False))
