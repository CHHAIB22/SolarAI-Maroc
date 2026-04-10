@echo off
setlocal enabledelayedexpansion
title Diagnostic Python

echo.
echo  =========================================
echo   Diagnostic Python - Energie IA
echo  =========================================
echo.
echo  Recherche de toutes les installations Python...
echo.

set FOUND=0

:: py launcher
py --version >nul 2>&1
if !errorlevel! == 0 (
    set FOUND=1
    for /f "tokens=*" %%i in ('py --version 2^>^&1') do echo  [OK] py launcher : %%i
    py -m pip --version >nul 2>&1
    if !errorlevel! == 0 (
        for /f "tokens=*" %%i in ('py -m pip --version 2^>^&1') do echo       pip : %%i
        echo       Commande recommandee : py -m pip install scikit-learn numpy
    ) else (
        echo       [!] pip non disponible pour cette installation
    )
    echo.
)

:: python in PATH
python --version >nul 2>&1
if !errorlevel! == 0 (
    set FOUND=1
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo  [OK] python : %%i
    for /f "tokens=*" %%i in ('where python 2^>^&1') do echo       Chemin : %%i
    python -m pip --version >nul 2>&1
    if !errorlevel! == 0 (
        for /f "tokens=*" %%i in ('python -m pip --version 2^>^&1') do echo       pip : %%i
        echo       Commande recommandee : python -m pip install scikit-learn numpy
    ) else (
        echo       [!] pip non disponible pour cette installation
    )
    echo.
)

:: Scan AppData
for /d %%d in ("C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python3*") do (
    if exist "%%d\python.exe" (
        "%%d\python.exe" --version >nul 2>&1
        if !errorlevel! == 0 (
            set FOUND=1
            for /f "tokens=*" %%i in ('"%%d\python.exe" --version 2^>^&1') do echo  [OK] %%d\python.exe : %%i
            "%%d\python.exe" -m pip --version >nul 2>&1
            if !errorlevel! == 0 (
                echo       pip disponible
                echo       Commande : "%%d\python.exe" -m pip install scikit-learn numpy
            ) else (
                echo       [!] pip non disponible
            )
            echo.
        ) else (
            echo  [--] %%d\python.exe  INTROUVABLE ou corrompu
        )
    )
)

if !FOUND! == 0 (
    echo  [ERREUR] Aucun Python trouve sur ce systeme.
    echo.
    echo  Telechargez Python 3.11 ou 3.12 sur :
    echo  https://www.python.org/downloads/
    echo  Cochez "Add Python to PATH" lors de l'installation.
)

echo.
echo  =========================================
echo.
echo  Une fois Python identifie, lancez :
echo    install-and-run.bat
echo.
pause
