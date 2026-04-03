
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Dimensions du damier : nombre de coins INTÉRIEURS (colonnes, lignes)
# Doit correspondre exactement à la grille utilisée lors de la prise de vue
nx = 7  # coins intérieurs en x (colonnes)
ny = 9  # coins intérieurs en y (lignes)
CHECKERBOARD = (nx, ny)  # tuple unique utilisé partout pour la cohérence
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
error =[]

# Définition des coordonnées 3D réelles des coins du damier (plan Z=0)
# objp a exactement nx*ny points, cohérent avec findChessboardCorners
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# Extracting path of individual image stored in a given directory
images = glob.glob('./images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    
    # --- AJOUT: Redimensionnement des images ---
    # Option 1: Redimensionnement proportionnel (ex: 50% de la taille d'origine)
    scale_percent = 50 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # L'interpolation cv2.INTER_AREA est recommandée pour réduire la taille
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print(img.shape)
    print(img.size)
    width2= width/2
    height2= height/2
    print(width2)
    print(height2)
    
    # Option 2: Redimensionnement par taille fixe (décommenter pour l'utiliser)
    # img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
    # -------------------------------------------

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        # CHECKERBOARD = (nx, ny) : maintenant cohérent avec findChessboardCorners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    # --- MODIFICATION: Affichage avec Matplotlib ---
    # OpenCV utilise le format BGR, Matplotlib attend du RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8)) # Contrôle de la taille de la fenêtre d'affichage
    plt.imshow(img_rgb)
    plt.title(f"Image: {os.path.basename(fname)}")
    plt.axis('off') # Masquer les axes pour un affichage propre
    plt.show(block=True) 
    # -----------------------------------------------

# Attention: cv2.destroyAllWindows() supprimé car nous n'utilisons plus cv2.imshow

h, w = img.shape[:2]

"""
Performing camera calibration by passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n", mtx)
print("dist : \n", dist)
print("rvecs : \n", rvecs)
print("tvecs : \n", tvecs)

# Save parameters into numpy file
if not os.path.exists("./camera_params_2"):
    os.makedirs("./camera_params_2")
np.save("./camera_params_2/ret", ret)
np.save("./camera_params_2/mtx", mtx)
np.save("./camera_params_2/dist", dist)
np.save("./camera_params_2/rvecs", rvecs)
np.save("./camera_params_2/tvecs", tvecs)

tot_error = 0
for i in range(len(objpoints)):
    reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    reprojected_points = reprojected_points.reshape(-1,2)
    
i=0
for j in range(5):
    error.append(np.sum(np.abs(imgpoints[i][j]-reprojected_points[j]), axis=0))

if len(error) >= 3:
    print(error[0])
    print(error[1])
    print(error[2])
