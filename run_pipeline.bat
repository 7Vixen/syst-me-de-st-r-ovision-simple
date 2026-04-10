@echo off
chcp 65001 >nul
echo ========================================
echo    Pipeline Stereovision
echo ========================================
echo.

echo [1/2] Calibration de la camera...
python calibrate.py
if %errorlevel% neq 0 (
    echo ERREUR: La calibration a echoue !
    pause
    exit /b 1
)
echo Calibration terminee.
echo.

echo [2/2] Detection SIFT + Reconstruction 3D...
python "LoacteSift&3dpoint.py"
if %errorlevel% neq 0 (
    echo ERREUR: La reconstruction 3D a echoue !
    pause
    exit /b 1
)
echo Reconstruction 3D terminee.
echo.

echo ========================================
echo    Pipeline complet termine !
echo ========================================
echo Fichiers generes:
echo   - nuage_points.xyz
echo   - sift_matches.png
echo   - nuage_3d.png
pause
