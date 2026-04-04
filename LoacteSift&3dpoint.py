import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================
# PARAMÈTRES
# ==============================================================
IMG_GAUCHE = r'ProjectImages/IMG_20260404_223421_508.jpg'
IMG_DROITE = r'ProjectImages/IMG_20260404_223447_839.jpg'
CALIB_MTX  = 'camera_params_2/mtx.npy'
CALIB_DIST = 'camera_params_2/dist.npy'
BASELINE   = 0.117  # en mètres (corrigé)

# ==============================================================
# 1. CHARGEMENT (sans undistort image)
# ==============================================================
mtx  = np.load(CALIB_MTX)
dist = np.load(CALIB_DIST)

img_L = cv2.imread(IMG_GAUCHE)
img_R = cv2.imread(IMG_DROITE)

gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

print("Images chargées:", img_L.shape, img_R.shape)











# ==============================================================
# 2. DETECTION SIFT + MATCHING
# ==============================================================
sift = cv2.SIFT_create(
    nfeatures=8000,
    contrastThreshold=0.02,
    edgeThreshold=15,
    sigma=1.2
)

kp_L, des_L = sift.detectAndCompute(gray_L, None)
kp_R, des_R = sift.detectAndCompute(gray_R, None)

print(f"Keypoints: Gauche={len(kp_L)}, Droite={len(kp_R)}")

FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches_raw = flann.knnMatch(des_L, des_R, k=2)

good_matches = []
for m, n in matches_raw:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Bons matches après ratio test: {len(good_matches)}")

pts_L = np.float32([kp_L[m.queryIdx].pt for m in good_matches])
pts_R = np.float32([kp_R[m.trainIdx].pt for m in good_matches])

# ==============================================================
# 3. UNDISTORT SEULEMENT LES POINTS (pas les images !)
# ==============================================================
pts_L_ud = cv2.undistortPoints(pts_L.reshape(-1, 1, 2), mtx, dist, P=mtx)
pts_R_ud = cv2.undistortPoints(pts_R.reshape(-1, 1, 2), mtx, dist, P=mtx)

pts_L_ud = pts_L_ud.reshape(-1, 2)
pts_R_ud = pts_R_ud.reshape(-1, 2)

# ==============================================================
# 4. FILTRAGE GÉOMÉTRIQUE RANSAC
# ==============================================================
F, mask = cv2.findFundamentalMat(pts_L_ud, pts_R_ud, cv2.FM_RANSAC, 3.0, 0.99)
mask = mask.ravel().astype(bool)

pts_L_in = pts_L_ud[mask]
pts_R_in = pts_R_ud[mask]
good_inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

print(f"Inliers après RANSAC: {len(pts_L_in)}")

# ==============================================================
# 5. FILTRE DISPARITÉ (avant triangulation)
# ==============================================================
disparites = pts_L_in[:, 0] - pts_R_in[:, 0]
print(f"Disparité moyenne: {disparites.mean():.2f} px")
print(f"Disparité min/max: {disparites.min():.2f} / {disparites.max():.2f} px")

# Garder seulement disparités positives et cohérentes
mask_disp = (disparites > 50) & (disparites < 800)
pts_L_in  = pts_L_in[mask_disp]
pts_R_in  = pts_R_in[mask_disp]

# Mettre à jour good_inliers pour la visualisation
indices_inliers = np.where(mask)[0]
indices_disp    = indices_inliers[mask_disp]
good_inliers_final = [good_matches[i] for i in indices_disp]

print(f"Points après filtre disparité: {len(pts_L_in)}")

# Visualisation sur images originales
img_matches = cv2.drawMatches(
    img_L, kp_L, img_R, kp_R,
    good_inliers_final[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(18, 6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Matches filtrés (affichés: 50/{len(pts_L_in)})')
plt.axis('off')
plt.tight_layout()
plt.savefig('sift_matches.png', dpi=150)
plt.show()

# ==============================================================
# 6. TRIANGULATION 3D
# ==============================================================
P_L = mtx @ np.hstack([np.eye(3), np.zeros((3, 1))])
T   = np.array([[-BASELINE], [0.0], [0.0]])
P_R = mtx @ np.hstack([np.eye(3), T])

print("\nP_L:\n", P_L)
print("P_R:\n", P_R)

points_4D = cv2.triangulatePoints(P_L, P_R, pts_L_in.T, pts_R_in.T)
points_3D = (points_4D[:3] / points_4D[3]).T

print(f"\nPoints 3D reconstruits: {len(points_3D)}")
print(f"X: [{points_3D[:,0].min():.3f}, {points_3D[:,0].max():.3f}] m")
print(f"Y: [{points_3D[:,1].min():.3f}, {points_3D[:,1].max():.3f}] m")
print(f"Z: [{points_3D[:,2].min():.3f}, {points_3D[:,2].max():.3f}] m")

# ==============================================================
# 7. FILTRAGE Z
# ==============================================================
mask_z = (points_3D[:, 2] > 0.35) & (points_3D[:, 2] < 0.80)
points_3D_clean = points_3D[mask_z]

# Récupérer coords originales pour les couleurs
pts_L_orig_filtered = pts_L[mask][mask_disp][mask_z]

print(f"Points après filtrage Z [0.35, 0.80]m: {len(points_3D_clean)}")

# ==============================================================
# 8. COULEURS
# ==============================================================
colors = []
for pt in pts_L_orig_filtered:
    x = int(np.clip(pt[0], 0, img_L.shape[1] - 1))
    y = int(np.clip(pt[1], 0, img_L.shape[0] - 1))
    b, g, r = img_L[y, x]
    colors.append([r/255, g/255, b/255])
colors = np.array(colors)

# ==============================================================
# 9. SAUVEGARDE .xyz (pour Blender / CloudCompare)
# ==============================================================
with open('nuage_points.xyz', 'w') as f:
    for i, pt in enumerate(points_3D_clean):
        r = int(colors[i][0] * 255)
        g = int(colors[i][1] * 255)
        b = int(colors[i][2] * 255)
        f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")

print("Fichier nuage_points.xyz sauvegardé !")

# ==============================================================
# 10. VISUALISATION 3D
# ==============================================================
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')

ax.scatter(
    points_3D_clean[:, 0],
    points_3D_clean[:, 2],
    -points_3D_clean[:, 1],
    c=colors,
    s=5,
    alpha=0.8
)

ax.set_xlabel('X (m)')
ax.set_ylabel('Profondeur Z (m)')
ax.set_zlabel('Y (m)')
ax.set_title(f'Nuage de points 3D — {len(points_3D_clean)} points')
plt.tight_layout()
plt.savefig('nuage_3d.png', dpi=150)
plt.show()

print("\n✅ Terminé !")
print("Fichiers sauvegardés: sift_matches.png, nuage_3d.png, nuage_points.xyz")