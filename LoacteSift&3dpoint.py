import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================
# PARAMÈTRES
# ==============================================================
IMG_GAUCHE = r'ProjectImages/IMG_20260410_230157_126.jpg'
IMG_DROITE = r'ProjectImages/IMG_20260410_230207_597.jpg'
CALIB_MTX  = 'camera_params_2/mtx.npy'
CALIB_DIST = 'camera_params_2/dist.npy'
BASELINE   = 0.05  # en mètres (5 cm)

# Distances réelles des boîtes (en mètres) — ground truth
GT_DISTANCES = {
    'Rose':    0.385,
    'Blanche': 0.495,
    'Noire':   0.650,
}

# ==============================================================
# 1. CHARGEMENT + CORRECTION ÉCHELLE CALIBRATION
# ==============================================================
mtx  = np.load(CALIB_MTX)
dist = np.load(CALIB_DIST)

img_L = cv2.imread(IMG_GAUCHE)
img_R = cv2.imread(IMG_DROITE)

if img_L is None or img_R is None:
    raise FileNotFoundError("Impossible de charger les images. Vérifiez les chemins.")

gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

print("Images chargées:", img_L.shape, img_R.shape)

# --- FIX #1 : Mise à l'échelle de la matrice intrinsèque ---
# La calibration a été faite sur des images redimensionnées à 50%,
# mais les images stéréo sont en pleine résolution → facteur ×2
CALIB_SCALE = 2.0
mtx_scaled = mtx.copy()
mtx_scaled[0, 0] *= CALIB_SCALE  # fx
mtx_scaled[1, 1] *= CALIB_SCALE  # fy
mtx_scaled[0, 2] *= CALIB_SCALE  # cx
mtx_scaled[1, 2] *= CALIB_SCALE  # cy

print(f"\nMatrice intrinsèque originale (calibration 50%):")
print(f"  fx={mtx[0,0]:.1f}, fy={mtx[1,1]:.1f}, cx={mtx[0,2]:.1f}, cy={mtx[1,2]:.1f}")
print(f"Matrice corrigée (pleine résolution, ×{CALIB_SCALE}):")
print(f"  fx={mtx_scaled[0,0]:.1f}, fy={mtx_scaled[1,1]:.1f}, cx={mtx_scaled[0,2]:.1f}, cy={mtx_scaled[1,2]:.1f}")

# ==============================================================
# 2. MASQUE ROI — cibler les boîtes, exclure le fond
# ==============================================================
h, w = gray_L.shape

# Masque image gauche : zone où se trouvent les boîtes
mask_roi_L = np.zeros((h, w), dtype=np.uint8)
mask_roi_L[int(h*0.35):int(h*0.68), int(w*0.08):int(w*0.92)] = 255

# Masque image droite : décalé vers la gauche (caméra déplacée à droite)
mask_roi_R = np.zeros((h, w), dtype=np.uint8)
mask_roi_R[int(h*0.33):int(h*0.68), int(w*0.02):int(w*0.85)] = 255

# Sauvegarde du masque pour vérification visuelle
debug_mask = img_L.copy()
debug_mask[mask_roi_L == 0] = (debug_mask[mask_roi_L == 0] * 0.3).astype(np.uint8)
cv2.imwrite('debug_masque.png', debug_mask)
print("\nMasque ROI sauvegardé dans debug_masque.png")

# ==============================================================
# 3. AMÉLIORATION CONTRASTE + DETECTION SIFT + MATCHING
# ==============================================================
# CLAHE pour améliorer les features sur les objets
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
gray_L_eq = clahe.apply(gray_L)
gray_R_eq = clahe.apply(gray_R)

# --- FIX #3 : Paramètres SIFT optimisés ---
sift = cv2.SIFT_create(
    nfeatures=8000,
    contrastThreshold=0.03,   # ↑ de 0.02 : filtre le bruit sur fond blanc
    edgeThreshold=10,          # ↓ de 15 : rejette les bords flous
    sigma=1.6                  # Standard Lowe (au lieu de 1.2)
)

# Détection avec masques ROI et images améliorées (CLAHE)
kp_L, des_L = sift.detectAndCompute(gray_L_eq, mask_roi_L)
kp_R, des_R = sift.detectAndCompute(gray_R_eq, mask_roi_R)

print(f"\nKeypoints (avec masque ROI + CLAHE): Gauche={len(kp_L)}, Droite={len(kp_R)}")

FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)  # ↑ de 50 : meilleure qualité de matching
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches_raw = flann.knnMatch(des_L, des_R, k=2)

# --- FIX #3 : Ratio test plus strict (0.70 au lieu de 0.75) ---
good_matches = []
for m, n in matches_raw:
    if m.distance < 0.70 * n.distance:
        good_matches.append(m)

print(f"Bons matches après ratio test (0.70): {len(good_matches)}")

pts_L = np.float32([kp_L[m.queryIdx].pt for m in good_matches])
pts_R = np.float32([kp_R[m.trainIdx].pt for m in good_matches])

# ==============================================================
# 4. UNDISTORT SEULEMENT LES POINTS (pas les images !)
# ==============================================================
# Utiliser mtx_scaled (corrigée) pour undistort
pts_L_ud = cv2.undistortPoints(pts_L.reshape(-1, 1, 2), mtx_scaled, dist, P=mtx_scaled)
pts_R_ud = cv2.undistortPoints(pts_R.reshape(-1, 1, 2), mtx_scaled, dist, P=mtx_scaled)

pts_L_ud = pts_L_ud.reshape(-1, 2)
pts_R_ud = pts_R_ud.reshape(-1, 2)

# ==============================================================
# 5. FILTRAGE GÉOMÉTRIQUE RANSAC
# ==============================================================
F, mask = cv2.findFundamentalMat(pts_L_ud, pts_R_ud, cv2.FM_RANSAC, 2.0, 0.99)
mask = mask.ravel().astype(bool)

pts_L_in = pts_L_ud[mask]
pts_R_in = pts_R_ud[mask]
good_inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

print(f"Inliers après RANSAC: {len(pts_L_in)}")

# ==============================================================
# 6. FILTRE DISPARITÉ (avant triangulation)
# ==============================================================
disparites = pts_L_in[:, 0] - pts_R_in[:, 0]
print(f"\nDisparité moyenne: {disparites.mean():.2f} px")
print(f"Disparité min/max: {disparites.min():.2f} / {disparites.max():.2f} px")

# --- FIX #4 : Filtre disparité ajusté ---
# Avec fx≈3097, B=0.05m :
#   Z=0.385m → disp≈402px, Z=0.65m → disp≈238px
# Marge de sécurité : [150, 550]
DISP_MIN, DISP_MAX = 150, 550
mask_disp = (disparites > DISP_MIN) & (disparites < DISP_MAX)
pts_L_in  = pts_L_in[mask_disp]
pts_R_in  = pts_R_in[mask_disp]

# Mettre à jour good_inliers pour la visualisation
indices_inliers = np.where(mask)[0]
indices_disp    = indices_inliers[mask_disp]
good_inliers_final = [good_matches[i] for i in indices_disp]

print(f"Points apres filtre disparite [{DISP_MIN}, {DISP_MAX}]: {len(pts_L_in)}")

# Visualisation sur images originales — echantillon representatif
# Trier les matches par coordonnee X pour couvrir toutes les boites
n_show = min(80, len(good_inliers_final))
if len(good_inliers_final) > n_show:
    # Echantillonner uniformement sur X pour montrer toutes les boites
    x_coords = np.array([kp_L[m.queryIdx].pt[0] for m in good_inliers_final])
    sorted_indices = np.argsort(x_coords)
    step = len(sorted_indices) // n_show
    selected_indices = sorted_indices[::step][:n_show]
    matches_to_show = [good_inliers_final[i] for i in selected_indices]
else:
    matches_to_show = good_inliers_final

img_matches = cv2.drawMatches(
    img_L, kp_L, img_R, kp_R,
    matches_to_show, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(20, 7))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Matches filtres (affiches: {len(matches_to_show)}/{len(pts_L_in)} - echantillon representatif)')
plt.axis('off')
plt.tight_layout()
plt.savefig('sift_matches.png', dpi=150)
plt.show()

# ==============================================================
# 7. TRIANGULATION 3D (avec matrice corrigée)
# ==============================================================
P_L = mtx_scaled @ np.hstack([np.eye(3), np.zeros((3, 1))])
T   = np.array([[-BASELINE], [0.0], [0.0]])
P_R = mtx_scaled @ np.hstack([np.eye(3), T])

print("\nP_L:\n", P_L)
print("P_R:\n", P_R)

points_4D = cv2.triangulatePoints(P_L, P_R, pts_L_in.T, pts_R_in.T)
points_3D = (points_4D[:3] / points_4D[3]).T

print(f"\nPoints 3D reconstruits: {len(points_3D)}")
print(f"X: [{points_3D[:,0].min():.4f}, {points_3D[:,0].max():.4f}] m")
print(f"Y: [{points_3D[:,1].min():.4f}, {points_3D[:,1].max():.4f}] m")
print(f"Z: [{points_3D[:,2].min():.4f}, {points_3D[:,2].max():.4f}] m")

# ==============================================================
# 8. FILTRAGE Z — garder les profondeurs physiquement cohérentes
# ==============================================================
Z_FILTER_MIN, Z_FILTER_MAX = 0.25, 0.90
mask_z = (points_3D[:, 2] > Z_FILTER_MIN) & (points_3D[:, 2] < Z_FILTER_MAX)
points_3D_clean = points_3D[mask_z]

# Récupérer coords originales pour les couleurs
pts_L_orig_filtered = pts_L[mask][mask_disp][mask_z]

print(f"Points après filtrage Z [{Z_FILTER_MIN}, {Z_FILTER_MAX}]m: {len(points_3D_clean)}")

# ==============================================================
# 9. COULEURS
# ==============================================================
colors = []
for pt in pts_L_orig_filtered:
    x = int(np.clip(pt[0], 0, img_L.shape[1] - 1))
    y = int(np.clip(pt[1], 0, img_L.shape[0] - 1))
    b, g, r = img_L[y, x]
    colors.append([r/255, g/255, b/255])
colors = np.array(colors)

# ==============================================================
# 10. ANALYSE PAR BOÎTE — Vérification ground truth
# ==============================================================
print("\n" + "="*60)
print("ANALYSE PAR BOÎTE — Comparaison avec distances réelles")
print("="*60)

# Convertir les couleurs des points en HSV pour la classification
hsv_img = cv2.cvtColor(img_L, cv2.COLOR_BGR2HSV)

pts_hsv = []
for pt in pts_L_orig_filtered:
    x = int(np.clip(pt[0], 0, img_L.shape[1] - 1))
    y = int(np.clip(pt[1], 0, img_L.shape[0] - 1))
    pts_hsv.append(hsv_img[y, x])
pts_hsv = np.array(pts_hsv)

# Classification par couleur HSV
# Rose : teinte ~140-175 (en OpenCV: 0-180), saturation moyenne-haute
mask_rose = (
    ((pts_hsv[:, 0] >= 140) | (pts_hsv[:, 0] <= 10)) &  # Teinte rose/magenta
    (pts_hsv[:, 1] > 50) &   # Saturation non nulle
    (pts_hsv[:, 2] > 100)     # Pas trop sombre
)

# Blanche : saturation basse, valeur haute
mask_blanche = (
    (pts_hsv[:, 1] < 60) &    # Saturation très basse
    (pts_hsv[:, 2] > 180)      # Très lumineux
)

# Noire : valeur basse (sombre)
mask_noire = (
    (pts_hsv[:, 2] < 80) &    # Très sombre
    (pts_hsv[:, 1] < 100)      # Saturation pas trop haute
)

box_masks = {
    'Rose': mask_rose,
    'Blanche': mask_blanche,
    'Noire': mask_noire,
}

results_table = []
for box_name, box_mask in box_masks.items():
    n_pts = box_mask.sum()
    gt_z = GT_DISTANCES[box_name]
    if n_pts > 0:
        z_vals = points_3D_clean[box_mask, 2]
        z_mean = z_vals.mean()
        z_median = np.median(z_vals)
        z_std = z_vals.std()
        erreur = abs(z_median - gt_z)
        erreur_pct = erreur / gt_z * 100
        print(f"\n[BOITE] {box_name}:")
        print(f"   Points detectes : {n_pts}")
        print(f"   Z moyen  = {z_mean:.4f} m | Z median = {z_median:.4f} m | sigma = {z_std:.4f} m")
        print(f"   Ground truth    = {gt_z:.3f} m")
        print(f"   Erreur (median) = {erreur:.4f} m ({erreur_pct:.1f}%)")
        results_table.append((box_name, n_pts, z_median, gt_z, erreur, erreur_pct))
    else:
        print(f"\n[BOITE] {box_name}: ATTENTION - 0 points detectes")
        results_table.append((box_name, 0, None, gt_z, None, None))

# Résumé tableau
print("\n" + "-"*70)
print(f"{'Boîte':<10} {'Pts':>5} {'Z médian':>10} {'GT':>10} {'Erreur':>10} {'%':>8}")
print("-"*70)
for name, n, z_med, gt, err, pct in results_table:
    if z_med is not None:
        print(f"{name:<10} {n:>5} {z_med:>10.4f} {gt:>10.3f} {err:>10.4f} {pct:>7.1f}%")
    else:
        print(f"{name:<10} {n:>5} {'N/A':>10} {gt:>10.3f} {'N/A':>10} {'N/A':>8}")
print("-"*70)

# ==============================================================
# 11. SAUVEGARDE .xyz (pour Blender / CloudCompare)
# ==============================================================
with open('nuage_points.xyz', 'w') as f:
    for i, pt in enumerate(points_3D_clean):
        r = int(colors[i][0] * 255)
        g = int(colors[i][1] * 255)
        b = int(colors[i][2] * 255)
        f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")

print(f"\nFichier nuage_points.xyz sauvegarde ({len(points_3D_clean)} points) !")

# ==============================================================
# 12. VISUALISATION 3D + HISTOGRAMME PROFONDEUR
# ==============================================================
fig = plt.figure(figsize=(18, 8))

# --- Subplot 1 : Nuage 3D ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(
    points_3D_clean[:, 0],
    points_3D_clean[:, 2],
    -points_3D_clean[:, 1],
    c=colors,
    s=8,
    alpha=0.8
)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Profondeur Z (m)')
ax1.set_zlabel('Y (m)')
ax1.set_title(f'Nuage 3D - {len(points_3D_clean)} points')

# --- Subplot 2 : Histogramme Z avec ground truth ---
ax2 = fig.add_subplot(122)
z_vals = points_3D_clean[:, 2]
ax2.hist(z_vals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)

gt_colors_map = {'Rose': 'magenta', 'Blanche': 'gray', 'Noire': 'black'}
for box_name, gt_z in GT_DISTANCES.items():
    ax2.axvline(x=gt_z, color=gt_colors_map[box_name], linestyle='--', linewidth=2,
                label=f'{box_name} GT={gt_z*100:.1f}cm')

ax2.set_xlabel('Profondeur Z (m)')
ax2.set_ylabel('Nombre de points')
ax2.set_title('Distribution des profondeurs')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Stereovision - Baseline={BASELINE*100:.0f}cm, fx={mtx_scaled[0,0]:.0f}', fontsize=13)
plt.tight_layout()
plt.savefig('nuage_3d.png', dpi=150)
plt.show()

print("\n=== TERMINE ===")
print("Fichiers sauvegardes: debug_masque.png, sift_matches.png, nuage_3d.png, nuage_points.xyz")