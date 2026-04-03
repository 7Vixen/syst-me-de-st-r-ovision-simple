import cv2
import numpy as np

# --- ÉTAPE 1 : Chargement et correction de la matrice ---
mtx  = np.load('camera_params_2/mtx.npy')
dist = np.load('camera_params_2/dist.npy')

img_gauche = cv2.imread(r'ProjectImages/IMG_20260325_212518_670.jpg')
img_droite = cv2.imread(r'ProjectImages/IMG_20260325_212558_085.jpg')

h, w = img_gauche.shape[:2]
print(f"Images projet : {w}x{h} (paysage)")

# Résolution du calibrage (après resize 50% sur portrait)
calib_w, calib_h = 864, 1920

# ✅ Facteurs d'échelle EN TENANT COMPTE DE LA ROTATION 90°
# Le 'w' du projet correspond au 'h' du calibrage (et vice versa)
scale_x = w / calib_h   # 3840 / 1920 = 2.0
scale_y = h / calib_w   # 1728 / 864  = 2.0

print(f"Scale x={scale_x:.3f}, y={scale_y:.3f}")

# ✅ Matrice corrigée avec échange fx↔fy et Ox↔Oy (rotation 90°)
fx_calib = mtx[0, 0]   # 1548.73  → devient fy
fy_calib = mtx[1, 1]   # 1548.37  → devient fx
Ox_calib = mtx[0, 2]   # 433.96   → devient Oy
Oy_calib = mtx[1, 2]   # 978.33   → devient Ox

mtx_landscape = np.array([
    [fy_calib * scale_x,  0,                  Oy_calib * scale_x],
    [0,                   fx_calib * scale_y,  Ox_calib * scale_y],
    [0,                   0,                   1                 ]
], dtype=np.float64)

print(f"\nMatrice caméra corrigée pour paysage :\n{mtx_landscape}")
print(f"\nVérification centre image :")
print(f"  Ox estimé = {mtx_landscape[0,2]:.1f}  (théorique = {w/2:.1f})")
print(f"  Oy estimé = {mtx_landscape[1,2]:.1f}  (théorique = {h/2:.1f})")

# ✅ Undistort correct
mtx_new, _ = cv2.getOptimalNewCameraMatrix(
    mtx_landscape, dist, (w, h), alpha=0.5, newImgSize=(w, h)
)

img_gauche_rect = cv2.undistort(img_gauche, mtx_landscape, dist, None, mtx_new)
img_droite_rect = cv2.undistort(img_droite, mtx_landscape, dist, None, mtx_new)

# Sauvegarde debug
cv2.imwrite("debug_gauche_rect.jpg", img_gauche_rect)
cv2.imwrite("debug_droite_rect.jpg", img_droite_rect)
print("\n✅ Images rectifiées sauvegardées — vérifie qu'elles ont l'air normales !")

gray_gauche = cv2.cvtColor(img_gauche_rect, cv2.COLOR_BGR2GRAY)
gray_droite = cv2.cvtColor(img_droite_rect, cv2.COLOR_BGR2GRAY)

# --- ÉTAPE 2 : SIFT ---
sift = cv2.SIFT_create(nfeatures=0)
kp1, des1 = sift.detectAndCompute(gray_gauche, None)
kp2, des2 = sift.detectAndCompute(gray_droite, None)
print(f"\nKeypoints : {len(kp1)} gauche | {len(kp2)} droite")

# FLANN matcher
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
    dict(checks=50)
)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
print(f"Après ratio test : {len(good_matches)} matches")

# --- ÉTAPE 3 : RANSAC ---
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

F, mask = cv2.findFundamentalMat(
    pts1, pts2, cv2.FM_RANSAC,
    ransacReprojThreshold=2.0, confidence=0.99
)
mask = mask.ravel().astype(bool)
pts1_f = pts1[mask]
pts2_f = pts2[mask]
print(f"Après RANSAC : {len(pts1_f)} matches")

# --- ÉTAPE 4 : Reconstruction 3D ---
b  = 0.14
fx = mtx_new[0, 0]
Ox = mtx_new[0, 2]
Oy = mtx_new[1, 2]

points_3D = []
stats = {"vertical": 0, "disparite": 0, "distance": 0}

for pt1, pt2 in zip(pts1_f, pts2_f):
    xl, yl = pt1
    xr, yr = pt2

    if abs(yl - yr) > 100:
        stats["vertical"] += 1
        continue

    d = abs(xl - xr)
    if d < 2.0 or d > 2000:
        stats["disparite"] += 1
        continue

    z = (b * fx) / d
    x =  b * (xl - Ox) / d
    y =  b * (yl - Oy) / d

    if not (0.05 < z < 8.0):
        stats["distance"] += 1
        continue

    points_3D.append([x, y, z])

print(f"\n--- Diagnostic ---")
for k, v in stats.items():
    print(f"  Rejetés ({k:10s}) : {v}")
print(f"  Points 3D gardés    : {len(points_3D)}")

if len(points_3D) > 0:
    pts = np.array(points_3D)
    print(f"\n  X : [{pts[:,0].min():.3f} → {pts[:,0].max():.3f}]  étendue={pts[:,0].max()-pts[:,0].min():.3f}m")
    print(f"  Y : [{pts[:,1].min():.3f} → {pts[:,1].max():.3f}]  étendue={pts[:,1].max()-pts[:,1].min():.3f}m")
    print(f"  Z : [{pts[:,2].min():.3f} → {pts[:,2].max():.3f}]  étendue={pts[:,2].max()-pts[:,2].min():.3f}m")
    np.savetxt("nuage_points.xyz", pts, fmt="%.5f")
    print("\n✅ nuage_points.xyz sauvegardé !")

img_matches = cv2.drawMatches(
    img_gauche_rect, kp1, img_droite_rect, kp2,
    [cv2.DMatch(i, i, 0) for i in range(min(100, len(pts1_f)))],
    None, matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("matches_valides.jpg", img_matches)
print("✅ matches_valides.jpg sauvegardé")