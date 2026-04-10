"""
history.py  — Retourne l'historique des predictions en JSON
Usage: python history.py <db_path>
"""
import sys
import json
import sqlite3
import os

db_path = sys.argv[1] if len(sys.argv) > 1 else "energie_renouvelable.db"

if not os.path.exists(db_path):
    print("[]")
    sys.exit(0)

try:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, date_prediction, nom_site, irradiation_solaire,
               vitesse_vent, disponibilite_eau, disponibilite_biomasse,
               disponibilite_terrain, temperature_moyenne,
               energie_recommandee, confiance_pct
        FROM predictions ORDER BY id DESC LIMIT 50
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    print(json.dumps(rows, ensure_ascii=False))
except Exception as e:
    print("[]")
