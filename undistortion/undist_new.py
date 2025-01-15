import os
import cv2
import numpy as np
from scipy.interpolate import griddata
import time
import yaml

def compute_undistorted_bbox(
    W_dist, H_dist, f_dist, xi_dist, 
    f_und,   # new focal length (for the undistorted camera)
    step=1   # sampling step: increase for speed, 1 for max accuracy
):
    """
    Computes the bounding box of the undistorted points (x_und, y_und)
    when mapping all distorted pixels (x_d, y_d) in the range [0..W_dist] x [0..H_dist].
    
    Returns:
        (min_x, max_x, min_y, max_y)
    """
    # Distorted camera principal point
    u0_dist = W_dist / 2.0
    v0_dist = H_dist / 2.0

    # We will gather the results of x_und, y_und here
    all_und = []

    # Loop over every 'step'-th pixel in the original image
    # If the image is large, you can set step > 1 for speed
    for y_d in range(0, H_dist, step):
        for x_d in range(0, W_dist, step):
            # 1) Distorted pixel -> normalized camera coordinates
            X_cam = (x_d - u0_dist) / f_dist
            Y_cam = (y_d - v0_dist) / f_dist
            Z_cam = 1.0

            # 2) To sphere
            #    alpha_cam_1 = xi*Z + sqrt(Z^2 + (1 - xi^2)(X^2 + Y^2))
            #    alpha_cam_2 = X^2 + Y^2 + Z^2
            alpha_cam_1 = xi_dist*Z_cam + np.sqrt(
                Z_cam**2 + (1.0 - xi_dist**2) * (X_cam**2 + Y_cam**2)
            )
            alpha_cam_2 = (X_cam**2 + Y_cam**2 + Z_cam**2)

            # avoid numerical issues if alpha_cam_2 is very small
            if np.isclose(alpha_cam_2, 0.0):
                continue

            alpha_cam = alpha_cam_1 / alpha_cam_2

            X_sph = alpha_cam * X_cam
            Y_sph = alpha_cam * Y_cam
            Z_sph = alpha_cam * Z_cam - xi_dist

            # 3) Sphere -> undistorted camera (xi1 = 0)
            #    x_und = (X_sph * f_und) / Z_sph
            #    y_und = (Y_sph * f_und) / Z_sph
            # only valid if Z_sph != 0
            if np.abs(Z_sph) < 1e-12:
                continue

            x_und = (X_sph * f_und) / Z_sph
            y_und = (Y_sph * f_und) / Z_sph

            all_und.append((x_und, y_und))

    if len(all_und) == 0:
        # Edge case: if for some reason everything was invalid
        return (0, 0, 0, 0)

    # Convert to array
    all_und = np.array(all_und)
    min_x = np.min(all_und[:, 0])
    max_x = np.max(all_und[:, 0])
    min_y = np.min(all_und[:, 1])
    max_y = np.max(all_und[:, 1])

    return (min_x, max_x, min_y, max_y)

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
    # Paramsund['W'] = int(Paramsd['W'] * resize_val)  # size of output (undist)
    # Paramsund['H'] = int(Paramsd['H'] * resize_val)

    # Parameters of the camera to generate
    f_dist = Paramsd['f']
    u0_dist = Paramsd['W'] / 2  
    v0_dist = Paramsd['H'] / 2
    xi = Paramsd['xi']  # distortion parameter (spherical model)
    
    f_undist = Paramsund['f']
    # u0_undist = Paramsund['W'] / 2  
    # v0_undist = Paramsund['H'] / 2
    u0_undist = Paramsund['u0']
    v0_undist = Paramsund['v0']
    xi1 = 0
    

    # 1. Projection on the image
    grid_x, grid_y = np.meshgrid(np.arange(Paramsund['W']), np.arange(Paramsund['H']))
    # X_Cam = np.divide(grid_x,f_undist) - u0_undist / f_undist
    # Y_Cam = np.divide(grid_y, f_undist) - v0_undist / f_undist
    X_Cam = (grid_x - u0_undist) / f_undist
    Y_Cam = (grid_y - v0_undist) / f_undist
    Z_Cam = np.ones((Paramsund['H'], Paramsund['W']))

    # 2. Image to sphere cart
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
    print(f"img: {path}")
    if Idis is None:
        print(f"Failed to load image: {path}")
        continue

    xi = distortion[i]  # distortion
    ImH, ImW, _ = Idis.shape
    f_dist = focal[i] * (ImW / ImH) * (ImH / 299)  # focal length
    
    start_time = time.time()
    min_x, max_x, min_y, max_y = compute_undistorted_bbox(
        W_dist=ImW,
        H_dist=ImH,
        f_dist=f_dist,
        xi_dist=xi,
        f_und=f_dist,
        step=4  # sample every 4th pixel for speed
    )
    print (f"test: {min_x}, {max_x}, {min_y}, {max_y}")
    # 4) Convert bounding box to integer resolution
    if int(np.floor(max_x - min_x)) > ImW * 3 : 
        W_und = ImW * 3
        H_und = int(np.floor(max_y - min_y) * ImW * 3 / np.floor(max_x - min_x))
        u0_und = int(np.floor(W_und/2))
        v0_und = int(np.floor(H_und/2))
    else :
        W_und = int(np.floor(max_x - min_x))
        H_und = int(np.floor(max_y - min_y))
        u0_und = -min_x
        v0_und = -min_y
    print (f"test: {W_und} x {H_und}")

    # 5) Define the principal point so that the bounding box fits from (0..W_und, 0..H_und).
    
    
    # resize_val = 3

    Paramsd = {'f': f_dist, 'W': ImW, 'H': ImH, 'xi': xi}
    Paramsund = {'f': f_dist, 'W': W_und, 'H': H_und, 'u0': u0_und, 'v0': v0_und}
    
    print (Paramsd)
    print (Paramsund)
    
    Image_und = undistSphIm(Idis, Paramsd, Paramsund, 1)
    print(f"Undistortion time for image {i + 1}: {time.time() - start_time:.2f} seconds")
    # Save the undistorted image
    filename = os.path.basename(path).split('.')[0]
    out_name = f"{filename}_ud.jpg"
    fullname = os.path.join(dist_folder, out_name)
    cv2.imwrite(fullname, Image_und)
