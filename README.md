# 🎥 Stéréovision Simple


---

## 📋 Description

Ce projet implémente un système de **stéréovision simple** à partir d'une seule caméra (téléphone) effectuant une translation horizontale. L'objectif est de reconstruire un nuage de points 3D à partir de deux images stéréo en utilisant la détection SIFT et la triangulation.

### Scène utilisée
Trois objets de dimensions connues posés sur une table :
- 📦 Une boîte noire (à ~55 cm de la caméra)
- 🗿 Une statuette/tour (à ~57 cm de la caméra)  
- 🍵 Une boîte de thé El Mrouan (à ~64 cm de la caméra)

![Scène stéréo](image%20copy.png)

---

## ⚙️ Pipeline

```
Calibration caméra
      ↓
Acquisition 2 images (translation horizontale, baseline = 8.78 cm)
      ↓
Détection SIFT + Matching FLANN
      ↓
Undistort des points (undistortPoints)
      ↓
Filtrage RANSAC (matrice fondamentale)
      ↓
Filtre sur disparité (50px < d < 800px)
      ↓
Triangulation 3D (cv2.triangulatePoints)
      ↓
Filtrage Z + Visualisation nuage de points
```

---

## 📷 Calibration de la caméra

La calibration a été effectuée avec un damier (chessboard) via OpenCV. Les paramètres obtenus :

```
Matrice intrinsèque :
[[1548.73    0      433.96]
 [   0    1548.37  978.33]
 [   0       0       1   ]]

Coefficients de distorsion :
[0.315, -1.053, 0.012, -0.001, 0.339]
```

**✅ Points positifs :**
- `fx ≈ fy` (1548) → caméra bien symétrique
- `cx ≈ 434`, `cy ≈ 978` → cohérent avec la résolution 1728×3840

**⚠️ Distorsion forte** (k1=0.315, k2=-1.053) → problème majeur rencontré lors de l'undistortion des images complètes (voir section Difficultés).

---

## 🔍 Détection SIFT & Matching

```python
sift = cv2.SIFT_create(
    nfeatures=8000,
    contrastThreshold=0.02,   # abaissé pour mieux détecter
    edgeThreshold=15,
    sigma=1.2
)
```

Le matching est réalisé avec **FLANN** suivi du **test de ratio de Lowe** (seuil 0.75), puis filtré par **RANSAC** sur la matrice fondamentale.

**Résultats obtenus :**
| Étape | Nombre de points |
|-------|-----------------|
| Keypoints détectés | ~8000 par image |
| Bons matches (ratio test) | ~1596 |
| Inliers après RANSAC | ~528 |
| Après filtre disparité | ~519 |
| Points 3D finaux | ~468 |

---

## 📐 Reconstruction 3D

La reconstruction utilise la triangulation directe avec les matrices de projection :

```
P_gauche = K · [I | 0]
P_droite = K · [I | -baseline]
```

**Valeurs Z reconstruites :** entre 0.35m et 0.80m — cohérent avec les distances mesurées (55–64 cm).

### Nuage de points obtenu

![Nuage 3D](nuage_3d.png)

> La forme allongée/inclinée correspond à la **statuette** — l'objet le plus texturé de la scène. Les axes représentent X (latéral), Z (profondeur) et Y (vertical).

---

## ⚠️ Difficultés rencontrées

### 1. 🌀 Distorsion forte → images déformées en "boule"
**Problème :** L'appel à `cv2.undistort()` sur l'image entière créait un effet fisheye inversé très prononcé à cause des coefficients de distorsion élevés (k1=0.315).

**Solution :** Ne **pas** undistorter les images — undistorter uniquement les **coordonnées des keypoints** avec `cv2.undistortPoints()`. Cela corrige la distorsion mathématiquement sans déformer l'image visuellement.

```python
# ❌ Mauvaise approche :
img_L = cv2.undistort(img_L_raw, mtx, dist)

# ✅ Bonne approche :
pts_L_ud = cv2.undistortPoints(pts_L.reshape(-1,1,2), mtx, dist, P=mtx)
```

---

### 2. ↔️ Disparités négatives et points aberrants
**Problème :** Après RANSAC, certains matches avaient des disparités négatives (jusqu'à -857px) ou excessives (+900px), produisant des points 3D derrière la caméra (Z < 0).

**Solution :** Filtre explicite sur la disparité avant triangulation :
```python
disparites = pts_L_in[:, 0] - pts_R_in[:, 0]
mask_disp = (disparites > 50) & (disparites < 800)
```

---

### 3. 📦 Boîte noire et boîte El Mrouan non détectées par SIFT

**Problème :** SIFT détecte des points d'intérêt uniquement sur les régions à **forte variation de gradient local**. La boîte noire (surface uniforme) et la boîte El Mrouan (surface lisse avec texte imprimé peu contrasté) génèrent très peu de descripteurs distinctifs, contrairement à la statuette qui est très riche en détails.

**Résultat :** La quasi-totalité des 468 points reconstruits appartient à la statuette.

**Pistes d'amélioration testées / envisagées :**
- Appliquer **CLAHE** (égalisation d'histogramme adaptative) avant la détection pour améliorer le contraste local
- Utiliser **ORB** à la place de SIFT, plus sensible aux surfaces peu texturées
- Utiliser **Harris corners** pour détecter les arêtes des boîtes

> Cette limitation est une **contrainte connue de SIFT** : il est optimisé pour les textures riches et n'est pas adapté aux surfaces homogènes ou uniformément colorées.

---

### 4. 📐 Baseline effective vs baseline mesurée
**Problème :** La baseline mesurée physiquement (8.78 cm) donnait des profondeurs Z légèrement sous-estimées par rapport aux distances réelles.

**Solution :** Ajustement à 11.7 cm après calibration empirique basée sur les distances réelles mesurées au mètre.

---

## 🗂️ Structure du projet

```
steriovision-simple/
├── ProjectImages/
│   ├── IMG_20260404_223421_508.jpg   # Image gauche
│   └── IMG_20260404_223447_839.jpg   # Image droite
├── camera_params_2/
│   ├── mtx.npy                        # Matrice intrinsèque
│   └── dist.npy                       # Coefficients distorsion
├── calibrate.py                       # Script de calibration
├── LoacteSift&3dpoint.py              # Script principal SIFT + 3D
├── visualise_Blender.py               # Visualisation dans Blender
├── nuage_points.xyz                   # Nuage de points (sortie)
├── sift_matches.png                   # Visualisation des matches
├── nuage_3d.png                       # Visualisation 3D matplotlib
└── README.md
```

---

## 🚀 Lancement

```bash
# 1. Calibration (si pas encore fait)
python calibrate.py

# 2. Détection SIFT + Reconstruction 3D
python LoacteSift&3dpoint.py

# 3. Visualisation dans Blender (optionnel)
python visualise_Blender.py
```

**Dépendances :**
```bash
pip install opencv-python numpy matplotlib
```

---

## 📊 Résultats finaux

| Paramètre | Valeur |
|-----------|--------|
| Baseline | 11.08 cm (effectif : ~11.7 cm) |
| Distance objets | 55 – 64 cm |
| Z reconstruit | 0.35 – 0.80 m |
| Points 3D valides | 468 |
| Objet principal détecté | Statuette (très texturée) |

---


