import os
import cv2
import numpy as np
from scipy.interpolate import griddata
import time
import yaml

def interp2linear(z, xi, yi, extrapval=np.nan):
    x = xi.copy()
    y = yi.copy()
    nrows, ncols, nchannels = z.shape

    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")

    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")


    # find x values out of range
    x_bad = ( (x < 0) | (x > ncols - 1))
    if x_bad.any():
        x[x_bad] = 0

    # find y values out of range
    y_bad = ((y < 0) | (y > nrows - 1))
    if y_bad.any():
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x)
    ndx = ndx.astype('int32')

    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - np.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols

    # interpolate
    one_minus_t = 1 - y
    z_ravel0 = z[:,:,0].ravel()
    z_ravel1 = z[:,:,1].ravel()
    z_ravel2 = z[:,:,2].ravel()
    f0 = (z_ravel0[ndx] * one_minus_t + z_ravel0[ndx + ncols] * y ) * (1 - x) + (
            z_ravel0[ndx + 1] * one_minus_t + z_ravel0[ndx + ncols + 1] * y) * x
    f1 = (z_ravel1[ndx] * one_minus_t + z_ravel1[ndx + ncols] * y) * (1 - x) + (
            z_ravel1[ndx + 1] * one_minus_t + z_ravel1[ndx + ncols + 1] * y) * x
    f2 = (z_ravel2[ndx] * one_minus_t + z_ravel2[ndx + ncols] * y) * (1 - x) + (
            z_ravel2[ndx + 1] * one_minus_t + z_ravel2[ndx + ncols + 1] * y) * x
    f = np.stack([f0,f1,f2], axis=-1)
    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval

    return f

def undistSphIm(Idis, Paramsd, Paramsund, resize_val=3):
    Paramsund['W'] = Paramsd['W'] * resize_val  # size of output (undist)
    Paramsund['H'] = Paramsd['H'] * resize_val

    # Parameters of the camera to generate
    f_dist = Paramsd['f']
    u0_dist = Paramsd['W'] / 2  
    v0_dist = Paramsd['H'] / 2
    
    f_undist = Paramsund['f']
    u0_undist = Paramsund['W'] / 2  
    v0_undist = Paramsund['H'] / 2
    xi = Paramsd['xi']  # distortion parameter (spherical model)
    

    # 1. Projection on the image
    grid_x, grid_y = np.meshgrid(np.arange(Paramsund['W']), np.arange(Paramsund['H']))
    X_Cam = np.divide(grid_x,f_undist) - u0_undist / f_undist
    Y_Cam = np.divide(grid_y, f_undist) - v0_undist / f_undist
    Z_Cam = np.ones((Paramsund['H'], Paramsund['W']))

    # 2. Image to sphere cart
    xi1 = 0
    
    alpha_cam_1 = (xi1*Z_Cam + np.sqrt(Z_Cam*Z_Cam + ((1-xi1*xi1)*(X_Cam*X_Cam + Y_Cam*Y_Cam))))
    alpha_cam_2 = (X_Cam*X_Cam+Y_Cam*Y_Cam+Z_Cam*Z_Cam)
    alpha_cam = alpha_cam_1 / alpha_cam_2

    X_Sph = X_Cam * alpha_cam
    Y_Sph = Y_Cam * alpha_cam
    Z_Sph = Z_Cam * alpha_cam - xi1

    # 3. Reprojection on distorted
    den = xi * (np.sqrt(X_Sph*X_Sph + Y_Sph*Y_Sph + Z_Sph*Z_Sph)) + Z_Sph
    X_d = ((X_Sph * f_dist) / den) + u0_dist
    Y_d = ((Y_Sph * f_dist) / den) + v0_dist

    # 4. Final step: interpolation and mapping
    Image_und = np.zeros((Paramsund['H'], Paramsund['W'], 3), dtype=np.float32)
    
    return interp2linear(Idis, X_d, Y_d)

# Main script

dist_folder = '/DeepCalib/weights/real_data/undistortion'
os.makedirs(dist_folder, exist_ok=True)

with open('/DeepCalib/weights/real_data/real_data.txt', 'r') as file:
    lines = file.readlines()

paths = []
focal = []
distortion = []

for line in lines:
    parts = line.strip().split()
    paths.append(parts[0])
    focal.append(float(parts[2]))  # Extract the focal length after 'prediction_focal'
    distortion.append(float(parts[4]))  # Extract the distortion after 'prediction_dist'

print(f"{paths}")
print(f"{focal}")
print(f"{distortion}")

for i, path in enumerate(paths):
    Idis = cv2.imread(path)
    if Idis is None:
        print(f"Failed to load image: {path}")
        continue

    xi = distortion[i]  # distortion
    ImH, ImW, _ = Idis.shape
    f_dist = focal[i] * (ImW / ImH) * (ImH / 299)  # focal length

    Paramsd = {'f': f_dist, 'W': ImW, 'H': ImH, 'xi': xi}
    Paramsund = {'f': f_dist, 'W': ImW, 'H': ImH}
    
    print (Paramsd)
    print (Paramsund)

    start_time = time.time()
    resize_val = 3
    if path == "/DeepCalib/weights/real_data/distorted/stitching.jpg":
        resize_val = 4
    Image_und = undistSphIm(Idis, Paramsd, Paramsund, resize_val)
    print(f"Undistortion time for image {i + 1}: {time.time() - start_time:.2f} seconds")

    # Save the undistorted image
    filename = os.path.basename(path).split('.')[0]
    out_name = f"{filename}_ud.jpg"
    fullname = os.path.join(dist_folder, out_name)
    cv2.imwrite(fullname, Image_und)
