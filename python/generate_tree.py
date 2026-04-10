"""
generate_tree.py - Genere le graphique de l'arbre de decision
Usage: python generate_tree.py <db_path> <output_png_path>
Retourne: JSON {"success": true, "path": "..."}
"""
import sys, json, os, sqlite3

db_path    = sys.argv[1] if len(sys.argv) > 1 else "energie_renouvelable.db"
output_png = sys.argv[2] if len(sys.argv) > 2 else "arbre_decision.png"

# Optional: last prediction params for highlighting
params_json = sys.argv[3] if len(sys.argv) > 3 else None

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    print(json.dumps({"success": False, "error": f"Module manquant: {e}"}))
    sys.exit(1)

SQL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donnees_energie.sql")

# ── Init DB ────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='donnees_entrainement'")
    if not cur.fetchone() or conn.execute("SELECT COUNT(*) FROM donnees_entrainement").fetchone()[0] == 0:
        if os.path.exists(SQL_FILE):
            with open(SQL_FILE, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            conn.commit()
    conn.close()

init_db()

# ── Load data ──────────────────────────────────────────────────
conn = sqlite3.connect(db_path)
rows = conn.execute("""
    SELECT irradiation_solaire, vitesse_vent, disponibilite_eau,
           disponibilite_biomasse, disponibilite_terrain,
           temperature_moyenne, energie_recommandee
    FROM donnees_entrainement ORDER BY id
""").fetchall()
conn.close()

eau_enc      = LabelEncoder().fit(["Non","Oui"])
biomasse_enc = LabelEncoder().fit(["Faible","Moyen","Eleve"])
terrain_enc  = LabelEncoder().fit(["Petit","Moyen","Grand"])

COULEURS = {
    "Energie Solaire PV": "#f59e0b",
    "Energie Eolienne":   "#38bdf8",
    "Hydroelectricite":   "#3b82f6",
    "Energie Biomasse":   "#22c55e",
}
COULEURS_LIST = ["#f59e0b","#38bdf8","#3b82f6","#22c55e"]

def encode(row):
    s,v,e,b,t,tmp,_ = row
    return [float(s), float(v),
            float(eau_enc.transform([e])[0]),
            float(biomasse_enc.transform([b])[0]),
            float(terrain_enc.transform([t])[0]),
            float(tmp)]

X = np.array([encode(r) for r in rows])
y = np.array([r[6] for r in rows])

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=3,
                              min_samples_leaf=2, random_state=42)
clf.fit(X, y)

FEAT_NAMES = ["Irradiation\nSolaire (kWh/m²/an)", "Vitesse\ndu Vent (m/s)",
              "Eau\n(0=Non,1=Oui)", "Biomasse\n(0=Faible,2=Eleve)",
              "Terrain\n(0=Petit,2=Grand)", "Temperature\n(°C)"]
CLASS_NAMES = clf.classes_.tolist()
NODE_COLORS = [COULEURS.get(c, "#8ba3c0") for c in CLASS_NAMES]

# ── Parse last prediction ──────────────────────────────────────
last_params = None
if params_json:
    try:
        last_params = json.loads(params_json)
    except:
        pass

# ── Figure ─────────────────────────────────────────────────────
BG   = "#060b14"
CARD = "#0f1929"
ACCENT = "#00d4ff"

fig = plt.figure(figsize=(20, 13), facecolor=BG)
fig.patch.set_facecolor(BG)

# Title
fig.text(0.5, 0.97, "Arbre de Decision — Classification des Sources d'Energie Renouvelable",
         ha='center', va='top', fontsize=16, fontweight='bold', color='white',
         fontfamily='monospace')
fig.text(0.5, 0.945, "Donnees entrainement : donnees_energie.sql  |  Profondeur max : 5  |  40 echantillons",
         ha='center', va='top', fontsize=9, color='#8ba3c0')

# ── Tree plot ──────────────────────────────────────────────────
ax_tree = fig.add_axes([0.01, 0.22, 0.98, 0.70])
ax_tree.set_facecolor(BG)

# Map class → color for nodes
class_to_color = {c: COULEURS.get(c, "#666") for c in CLASS_NAMES}
node_colors_tree = []
tree = clf.tree_
for i in range(tree.node_count):
    if tree.feature[i] == -2:  # leaf
        majority = CLASS_NAMES[np.argmax(tree.value[i][0])]
        c = class_to_color[majority]
        node_colors_tree.append(c + "55")
    else:
        node_colors_tree.append("#1a2840")

plot_tree(
    clf,
    feature_names=FEAT_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    ax=ax_tree,
    fontsize=7,
    node_ids=False,
)

# Re-color nodes and text
for collection in ax_tree.collections:
    collection.set_edgecolor(ACCENT + "80")
for i, patch in enumerate(ax_tree.patches):
    if i < len(node_colors_tree):
        patch.set_facecolor(node_colors_tree[i])
        patch.set_edgecolor(ACCENT + "60")
        patch.set_linewidth(0.8)

for text in ax_tree.texts:
    text.set_color("white")
    text.set_fontsize(6.5)

for line in ax_tree.lines:
    line.set_color("#263d5f")
    line.set_linewidth(0.8)

ax_tree.set_title("Structure de l'Arbre de Decision", color='#8ba3c0',
                   fontsize=10, pad=6, loc='center')
ax_tree.axis('off')

# ── Bottom panels ──────────────────────────────────────────────
# Panel 1: Confidence bars
ax_bar = fig.add_axes([0.03, 0.01, 0.35, 0.18])
ax_bar.set_facecolor(CARD)
ax_bar.spines[:].set_color("#1e3050")
ax_bar.tick_params(colors='#8ba3c0', labelsize=8)
ax_bar.xaxis.label.set_color('#8ba3c0')
ax_bar.yaxis.label.set_color('#8ba3c0')
ax_bar.set_title("Frequence des Classes (entrainement)", color='#8ba3c0', fontsize=8, pad=4)

class_counts = {c: list(y).count(c) for c in CLASS_NAMES}
bars = ax_bar.barh(list(class_counts.keys()), list(class_counts.values()),
                    color=[COULEURS.get(c, '#666') for c in class_counts.keys()],
                    edgecolor='none', height=0.5)
for bar, val in zip(bars, class_counts.values()):
    ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='white', fontsize=8)
ax_bar.set_xlabel("Echantillons", fontsize=8)
ax_bar.set_xlim(0, max(class_counts.values()) + 3)
for label in ax_bar.get_yticklabels():
    label.set_color(COULEURS.get(label.get_text(), '#8ba3c0'))

# Panel 2: Feature importances
ax_imp = fig.add_axes([0.41, 0.01, 0.35, 0.18])
ax_imp.set_facecolor(CARD)
ax_imp.spines[:].set_color("#1e3050")
ax_imp.tick_params(colors='#8ba3c0', labelsize=7)
ax_imp.set_title("Importance des Variables", color='#8ba3c0', fontsize=8, pad=4)

short_names = ["Irradiation", "Vent", "Eau", "Biomasse", "Terrain", "Temperature"]
importances = clf.feature_importances_
sorted_idx  = np.argsort(importances)
colors_imp  = [f"#{int(0+i*30):02x}{int(100+i*20):02x}{int(150+i*15):02x}" for i in range(len(importances))]

bars2 = ax_imp.barh([short_names[i] for i in sorted_idx],
                     importances[sorted_idx],
                     color=[ACCENT if importances[i] == max(importances) else "#263d5f"
                            for i in sorted_idx],
                     edgecolor='none', height=0.5)
for bar, val in zip(bars2, importances[sorted_idx]):
    ax_imp.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', color='white', fontsize=7)
ax_imp.set_xlabel("Importance (Gini)", fontsize=8, color='#8ba3c0')
ax_imp.set_xlim(0, max(importances) + 0.12)
ax_imp.xaxis.label.set_color('#8ba3c0')

# Panel 3: Legend
ax_leg = fig.add_axes([0.79, 0.01, 0.20, 0.18])
ax_leg.set_facecolor(CARD)
ax_leg.spines[:].set_color("#1e3050")
ax_leg.axis('off')
ax_leg.set_title("Legende", color='#8ba3c0', fontsize=8, pad=4)

icons = {"Energie Solaire PV": "☀", "Energie Eolienne": "~",
         "Hydroelectricite": "≈", "Energie Biomasse": "✿"}
for i, (cls, color) in enumerate(COULEURS.items()):
    y_pos = 0.82 - i * 0.22
    rect = FancyBboxPatch((0.05, y_pos - 0.08), 0.12, 0.14,
                           boxstyle="round,pad=0.02",
                           facecolor=color + "44", edgecolor=color, linewidth=1.2,
                           transform=ax_leg.transAxes)
    ax_leg.add_patch(rect)
    ax_leg.text(0.11, y_pos, icons.get(cls, "●"), transform=ax_leg.transAxes,
                fontsize=12, ha='center', va='center', color=color)
    short = cls.replace("Energie ", "").replace("Hydroelectricite", "Hydro")
    ax_leg.text(0.25, y_pos, short, transform=ax_leg.transAxes,
                fontsize=8, va='center', color=color, fontweight='bold')

plt.savefig(output_png, dpi=130, bbox_inches='tight',
            facecolor=BG, edgecolor='none', pad_inches=0.15)
plt.close()

print(json.dumps({"success": True, "path": output_png}))
