# Makefile - Pipeline Stéréovision
# Usage: make all    (calibration + SIFT + 3D)
#        make calib  (calibration seule)
#        make sift   (SIFT + reconstruction 3D seule)
#        make clean  (supprime les fichiers générés)

PYTHON = python

all: calib sift
	@echo ✅ Pipeline complet terminé !

calib:
	@echo 📷 Étape 1 : Calibration de la caméra...
	$(PYTHON) calibrate.py
	@echo ✅ Calibration terminée.

sift:
	@echo 🔍 Étape 2 : Détection SIFT + Reconstruction 3D...
	$(PYTHON) "LoacteSift&3dpoint.py"
	@echo ✅ Reconstruction 3D terminée.

.PHONY: all calib sift clean
