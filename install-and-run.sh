#!/bin/bash
echo "============================================"
echo " Energie Renouvelable IA - Installation"
echo "============================================"
echo

echo "[1/3] Installation des modules Python..."
pip install scikit-learn numpy --quiet || pip3 install scikit-learn numpy --quiet
echo " OK"

echo
echo "[2/3] Installation des modules Node.js..."
npm install
echo " OK"

echo
echo "[3/3] Lancement de l'application..."
npm run dev
