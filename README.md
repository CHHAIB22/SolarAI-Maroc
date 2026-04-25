# ⚡ Énergie Renouvelable IA

<div align="center">

![Electron](https://img.shields.io/badge/Electron-2B2E3A?style=for-the-badge&logo=electron&logoColor=9FEAF9)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

**Application desktop intelligente pour la classification et le dimensionnement des énergies renouvelables au Maroc**

*Projet de Fin d'Études — DUT Énergies Renouvelables et Efficacité Énergétique*  
*École Supérieure de Technologie de Guelmim — 2025/2026*

</div>

---

## 📸 Aperçu

> Interface dark moderne avec classification IA, historique SQLite, arbre de décision interactif et calculateur de dimensionnement solaire.

---

## ✨ Fonctionnalités

### 🔍 Analyse IA
- Sélecteur de **43 villes marocaines** avec données climatiques réelles (irradiation, vent, température)
- Sliders interactifs pour les paramètres du site
- Classification automatique par **DecisionTreeClassifier (scikit-learn)**
- Affichage **RadarChart + BarChart** (Recharts) avec probabilités par classe
- Confiance de prédiction en %
- Sauvegarde automatique dans **SQLite**

### 📋 Historique
- Tableau de toutes les prédictions sauvegardées
- Affichage ville, région, paramètres et résultat coloré par énergie
- Actualisation en temps réel

### 🌳 Arbre de Décision Interactif
- Visualisation **SVG pure** du modèle entraîné (aucune image, 100% code)
- **Zoom**, **drag & drop** pour naviguer
- Clic sur chaque nœud → détails (gini, échantillons, seuil, pureté)
- **Chemin de décision surligné** après chaque prédiction

### 🔆 Calculateur de Dimensionnement
- Sélection de ville → irradiation réelle automatique
- Liste d'appareils personnalisable (puissance + heures/jour)
- Paramètres système : tension (12/24/48V), autonomie, profondeur de décharge...
- Calcul automatique de :
  - ☀ **Nombre de panneaux** (méthode Peak Sun Hours)
  - 🔋 **Nombre de batteries** (capacité Ah)
  - ⚡ **Régulateur** (courant A, type PWM/MPPT)
  - 🔌 **Onduleur** (puissance W, type sinusoïdal)
  - 🔗 **Section des câbles** (mm², formule IEC ΔU=3%)
- **Schéma de câblage SVG** généré dynamiquement

### 📚 Guide Énergies
- Référence sur les 4 sources : Solaire PV, Éolien, Hydroélectricité, Biomasse

---

## 🗺️ Villes Marocaines Couvertes

43 villes avec données réelles NASA POWER / IRENA couvrant toutes les régions :

| Région | Exemples |
|--------|---------|
| Guelmim-Oued Noun | Guelmim, Tan-Tan, Assa |
| Drâa-Tafilalet | Ouarzazate, Zagora, Errachidia, Merzouga |
| Souss-Massa | Agadir, Tiznit, Taroudant |
| Marrakech-Safi | Marrakech, Essaouira, Safi |
| Oriental | Oujda, Nador, Figuig |
| Nord / Méditerranée | Tanger, Tétouan, Al Hoceïma |
| Et plus... | Fès, Meknès, Rabat, Casablanca... |

---

## 🧮 Algorithmes & Modèles

### Classification IA
```
Algorithme  : DecisionTreeClassifier (scikit-learn)
Critère     : Gini impurity
Profondeur  : max_depth = 5
Données     : 40 échantillons (donnees_energie.sql)
Classes     : Solaire PV · Éolien · Hydroélectricité · Biomasse
```

### Dimensionnement (formules IEC)
```
Panneaux  : N = ⌈(Conso × (1+pertes)) / (Irr × Ppanneau)⌉
Batteries : N = ⌈(Conso × Autonomie × k) / (V × DoD × Cap)⌉
Régulateur: I = ⌈Ppv / V × 1.25⌉  →  MPPT si I > 30A
Câbles    : S = (I × L × 2) / (σ × ΔU)   σ=56 S/m, ΔU=3%
```

---

## 🛠️ Stack Technique

| Couche | Technologie |
|--------|------------|
| Desktop shell | Electron |
| Interface | React 18 |
| Graphiques | Recharts, SVG natif |
| IA / ML | Python 3, scikit-learn, NumPy |
| Base de données | SQLite 3 |
| Style | CSS Variables (thème dark) |
| Fonts | Syne, Space Grotesk, JetBrains Mono |

---

## 🚀 Installation

### Prérequis
- **Node.js** ≥ 16
- **Python** 3.9 → 3.12 (recommandé)
- **pip** packages : `scikit-learn numpy`

### Windows
```bat
git clone https://github.com/CHHAIB22/energie-renouvelable-ia.git
cd energie-renouvelable-ia
install-and-run.bat
```

### Linux / macOS
```bash
git clone https://github.com/CHHAIB22/energie-renouvelable-ia.git
cd energie-renouvelable-ia
pip install scikit-learn numpy --break-system-packages
npm install
npm run dev
```

### Installer les dépendances Python manuellement
```bash
pip install scikit-learn numpy
```

---

## 📁 Structure du Projet

```
energie-renouvelable-ia/
├── electron/
│   ├── main.js              # Processus principal Electron + IPC
│   └── preload.js           # Bridge contextBridge sécurisé
├── python/
│   ├── predict.py           # Modèle scikit-learn + prédiction
│   ├── history.py           # Lecture historique SQLite
│   └── donnees_energie.sql  # Données d'entraînement (40 samples)
├── src/
│   ├── App.js               # Routage des 5 vues
│   ├── components/
│   │   ├── FormPanel        # Saisie paramètres + sélecteur villes
│   │   ├── ResultPanel      # Résultats RadarChart + BarChart
│   │   ├── HistoryPanel     # Tableau SQLite
│   │   ├── TreePanel        # Arbre SVG interactif
│   │   ├── DimensionnementPanel  # Calculateur solaire
│   │   └── Sidebar          # Navigation 5 onglets
│   └── data/
│       └── villesMaroc.js   # 43 villes avec données climatiques
├── install-and-run.bat      # Script installation Windows
├── install-and-run.sh       # Script installation Linux/macOS
└── package.json
```

---

## 👥 Équipe

| Nom | Rôle |
|-----|------|
| **Mohamed-Amine CHHAIB** | Développement full-stack, modèle IA, dimensionnement |

**Encadrant :** Pr. Nour-Eddine AIT ABDELLAH  
**Établissement :** École Supérieure de Technologie — Guelmim  
**Filière :** DUT Énergies Renouvelables et Efficacité Énergétique (EREE)

---

## 📄 Licence

Ce projet est développé dans le cadre d'un Projet de Fin d'Études académique.  
© 2026 CHHAIB22 — EST Guelmim

---

<div align="center">
  <sub>Fait avec ☀ à Guelmim, Maroc 🇲🇦</sub>
</div>
