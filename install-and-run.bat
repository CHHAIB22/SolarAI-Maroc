@echo off
setlocal enabledelayedexpansion
title Energie Renouvelable IA

echo.
echo  ============================================
echo   Energie Renouvelable IA - Installation
echo  ============================================
echo.

:: STEP 1 - Python
echo  ETAPE 1/4 - Python...
set PY=
for %%v in (312 311 310 313 314) do (
    if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python%%v\python.exe" (
        set PY="C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python%%v\python.exe"
        goto :pyok
    )
)
for %%v in (3.12 3.11 3.10 3.13 3.14) do (
    py -%%v --version >nul 2>&1
    if !errorlevel!==0 ( set PY=py -%%v & goto :pyok )
)
py --version >nul 2>&1
if !errorlevel!==0 ( set PY=py & goto :pyok )
python --version >nul 2>&1
if !errorlevel!==0 ( set PY=python & goto :pyok )
echo  ERREUR: Python introuvable.
pause & exit /b 1
:pyok
for /f "tokens=*" %%i in ('%PY% --version 2^>^&1') do echo  Python: %%i
echo  OK

:: STEP 2 - pip
echo.
echo  ETAPE 2/4 - scikit-learn + numpy...
%PY% -m pip install scikit-learn numpy --quiet 2>nul
if !errorlevel! neq 0 %PY% -m pip install scikit-learn numpy --user --quiet 2>nul
echo  OK
if not exist electron mkdir electron
echo %PY%> electron\python_path.txt

:: STEP 3 - Node.js
echo.
echo  ETAPE 3/4 - Node.js...
set ND=
set NM=
where node >nul 2>&1
if !errorlevel!==0 ( set ND=node & set NM=npm & for /f "tokens=*" %%i in ('node --version') do echo  Node: %%i & goto :nodeok )
if exist "C:\Program Files\nodejs\node.exe" ( set ND="C:\Program Files\nodejs\node.exe" & set NM="C:\Program Files\nodejs\npm.cmd" & goto :nodeok )
if exist "C:\Program Files (x86)\nodejs\node.exe" ( set ND="C:\Program Files (x86)\nodejs\node.exe" & set NM="C:\Program Files (x86)\nodejs\npm.cmd" & goto :nodeok )
for /r "C:\Program Files" %%f in (node.exe) do ( if not defined ND set ND="%%f" & set NM="%%~dpfnpm.cmd" & goto :nodeok )
echo  NODE.JS INTROUVABLE
start "" "https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi"
pause & exit /b 1
:nodeok
for /f "tokens=*" %%i in ('%ND% --version 2^>^&1') do echo  Node: %%i
echo  OK

:: STEP 4 - npm install
echo.
echo  ETAPE 4/4 - npm install (2-5 min)...
echo.
call %NM% install
if !errorlevel! neq 0 ( echo  ERREUR npm install. & pause & exit /b 1 )
echo  OK

:: LAUNCH
echo.
echo  ============================================
echo   Lancement...
echo  ============================================
echo.
call %NM% run dev
pause
