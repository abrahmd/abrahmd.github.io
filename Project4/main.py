import cv2
import numpy as np
import os
import skimage.io as skio
import viser
import time
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from nerf import NeRF
from RaysData import RaysData
import torch.nn.functional as F

import imageio
from natsort import natsorted


def visualise1(images_train, K, c2ws_train):
    # --- You Need to Implement These ------
    dataset = RaysData(images_train, K, c2ws_train)
    rays_o, rays_d, idx = dataset.sample_rays(100) # Should expect (B, 3)
    points, t_vals = dataset.sample_points_along_rays(rays_o, rays_d, perturb=True)
    H, W = images_train.shape[1:3]

    K = K.detach().cpu().numpy()
    rays_o = rays_o.detach().cpu().numpy()
    rays_d = rays_d.detach().cpu().numpy()
    points = points.detach().cpu().numpy()
    images_train = images_train.detach().cpu().numpy()

    c2ws_train = c2ws_train.detach().cpu().numpy()
    # ---------------------------------------

    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
        )
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run

def visualise2(images_train, K, c2ws_train):
    # --- You Need to Implement These ------
    dataset = RaysData(images_train, K, c2ws_train)
    H, W = images_train[0].shape[:2]

    images_train = images_train.detach().cpu().numpy()
    c2ws_train = c2ws_train.detach().cpu().numpy()
    K = K.detach().cpu().numpy()

    # This will check that your uvs aren't flipped
    uvs_start = 0
    uvs_end = 40_000
    sample_uvs = dataset.uvs[uvs_start:uvs_end] # These are integer coordinates of widths / heights (xy not yx) of all the pixels in an image
    # uvs are array of xy coordinates, so we need to index into the 0th image tensor with [0, height, width], so we need to index with uv[:,1] and then uv[:,0]

    sample_uvs = sample_uvs.detach().cpu().numpy()
    pixels = dataset.pixels.detach().cpu().numpy()
    assert np.all(images_train[0, sample_uvs[:,1], sample_uvs[:,0]] == pixels[uvs_start:uvs_end])

    # # Uncoment this to display random rays from the first image
    #indices = np.random.randint(low=0, high=40_000, size=100)

    # # Uncomment this to display random rays from the top left corner of the image
    indices_x = np.random.randint(low=100, high=200, size=100)
    indices_y = np.random.randint(low=0, high=100, size=100)
    indices = indices_x + (indices_y * 200)

    data = {"rays_o": dataset.ray_origins[indices], "rays_d": dataset.ray_directions[indices]}
    points, t = dataset.sample_points_along_rays(data["rays_o"], data["rays_d"])
    points = points.detach().cpu().numpy()
    # ---------------------------------------

    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    for i, (o, d) in enumerate(zip(data["rays_o"], data["rays_d"])):
        positions = np.stack((o, o + d * 6.0))
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=positions,
        )
        server.add_point_cloud(
            f"/samples",
            colors=np.zeros_like(points).reshape(-1, 3),
            points=points.reshape(-1, 3),
            point_size=0.03,
        )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run


def show_images(images, titles=None, cols=3, same_scale=False):
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axs = np.atleast_1d(axs).ravel()

    vmin = vmax = None
    if same_scale:
        vals = [im for im in images if im.ndim == 2]
        if vals:
            vmin = min(float(x.min()) for x in vals)
            vmax = max(float(x.max()) for x in vals)

    for i, im in enumerate(images):
        if im.ndim == 2:
            axs[i].imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            axs[i].imshow(im)  # RGB
        if titles: axs[i].set_title(titles[i])
        axs[i].axis('off')

    # Hide any empty slots
    for j in range(i+1, rows*cols):
        axs[j].axis('off')

    plt.tight_layout(); plt.show()

# Create ArUco dictionary and detector parameters (4x4 tags)
def get_matching_points(images_path):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    id_to_world_coords = dict()
    id_to_world_coords[0] = np.array([[0, 0, 0], [0.06, 0, 0], [0.06, 0.06, 0], [0, 0.06, 0]], dtype=np.float32)
    id_to_world_coords[1] = np.array([[0.09, 0, 0], [.15, 0, 0], [.15, 0.06, 0], [0.09, 0.06, 0]], dtype=np.float32)
    id_to_world_coords[2] = np.array([[0, .07567, 0], [0.06, 0.07567, 0], [0.06, .13567, 0], [0, .13567, 0]], dtype=np.float32)
    id_to_world_coords[3] = np.array([[0.09, 0.07567, 0], [.15, 0.07567, 0], [.15, .13567, 0], [0.09, .13567, 0]], dtype=np.float32)
    id_to_world_coords[4] = np.array([[0, .15134, 0], [0.06, .15134, 0], [0.06, .21134, 0], [0, .21134, 0]], dtype=np.float32)
    id_to_world_coords[5] = np.array([[0.09, .15134, 0], [.15, .15134, 0], [.15, .21134, 0], [0.09, .21134, 0]], dtype=np.float32)

    img_pts_list = []
    obj_pts_list = []
    images = []
    for filename in os.listdir(images_path):
        if not filename.lower().endswith('.jpg'):
            continue


        im_path = images_path + "/" + filename
        im = skio.imread(im_path)
        
        # Detect ArUco markers in an image
        # Returns: corners (list of numpy arrays), ids (numpy array)
        corners, ids, _ = cv2.aruco.detectMarkers(im, aruco_dict, parameters=aruco_params)

        # Check if any markers were detected
        if ids is None:
            continue
            
        # Process the detected corners
            # corners: list of length N (number of detected tags)
            #   - each element is a numpy array of shape (1, 4, 2) containing the 4 corner coordinates (x, y)
            # ids: numpy array of shape (N, 1) containing the tag IDs for each detected marker
            # Example: if 3 tags detected, corners will be a list of 3 arrays, ids will be shape (3, 1)
        img_pts = []
        obj_pts = []
        for i in range(len(ids)):
            tag_id = ids[i][0]
            if tag_id in id_to_world_coords:
                c = corners[i].reshape(-1, 2)
                w = id_to_world_coords[tag_id]

                img_pts.append(c)
                obj_pts.append(w)
        if len(img_pts) == 0:
            continue


        img_pts = np.vstack(img_pts)
        obj_pts = np.vstack(obj_pts)

        images.append(im)
        img_pts_list.append(img_pts)
        obj_pts_list.append(obj_pts)
            

    return img_pts_list, obj_pts_list, images

def visualize_cameras(images, c2ws, K):
    H = images[0].shape[0]
    W = images[0].shape[1]
    c2ws = np.array(c2ws)
    server = viser.ViserServer(host="127.0.0.1", port=8080)
    # Example of visualizing a camera frustum (in practice loop over all images)
    # c2w is the camera-to-world transformation matrix (3x4), and K is the camera intrinsic matrix (3x3)
    for i in range(len(images)):
        c2w = c2ws[i, :3, :]
        img = images[i]
        server.scene.add_camera_frustum(
            f"/cameras/{i}", # give it a name
            fov=2 * np.arctan2(H / 2, K[1, 1]), # field of view
            aspect=W / H, # aspect ratio
            scale=0.02, # scale of the camera frustum change if too small/big
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz, # orientation in quaternion format
            position=c2w[:3, 3], # position of the camera
            image=img # image to visualize
        )

    while True:
        time.sleep(0.1) 
################################################### 0.3 ##################################################################

def estimate_camera_pose(images, obj_pts_list, img_pts_list, K, distCoeffs):
    c2ws = []
    #extrinsics = []
    for i in range(len(images)):
        success, rvec, tvec = cv2.solvePnP(obj_pts_list[i], img_pts_list[i], K, distCoeffs)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3,1)
        Rt = np.hstack((R, t))
        Rth = np.vstack((Rt, [0,0,0,1]))
        #extrinsics.append(Rth)
        c2ws.append(np.linalg.inv(Rth))
    
    return c2ws

def split_and_save_data(images, c2ws, K, distCoeffs):
    undisorted_images = []
    for i in range(len(images)):
        im = images[i]
        im = cv2.undistort(im, K, distCoeffs)
        undisorted_images.append(im)

    idx = np.arange(len(images))
    idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

    images_train = np.array([images[i] for i in idx_train])
    c2w_train = np.array([c2ws[i] for i in idx_train])

    images_val = np.array([images[i] for i in idx_val])
    c2w_val = np.array([c2ws[i] for i in idx_val])

    images_test = np.array([images[i] for i in idx_test])
    c2w_test = np.array([c2ws[i] for i in idx_test])

    np.savez(
        'data.npz',
        images_train = images_train,
        c2ws_train = c2w_train,
        images_val = images_val,
        c2ws_val = c2w_val,
        c2ws_test = c2w_test,
        focal = K[0,0]
    )

def save_dataset():
    img_pts_list, obj_pts_list, images = get_matching_points("data/calibration2") #object points in 3d world coord system, image points appear in 2d plane
    h, w = images[0].shape[:2]
    _, K, distCoeffs, _, _ = cv2.calibrateCamera(obj_pts_list, img_pts_list, (w, h), None, None)

    img_pts_list, obj_pts_list, kitty_images = get_matching_points("data/teapot")
    c2ws = estimate_camera_pose(kitty_images, obj_pts_list, img_pts_list, K, distCoeffs)#, images)

    split_and_save_data(kitty_images, c2ws, K, distCoeffs)
    #visualize_cameras(kitty_images, c2ws, K)

####################################### PART 2 #############################################


def PSNR(mse_loss):
    return 10 * torch.log10(1 / mse_loss)


def PE(x, L):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()

    x = x.float()

    pe = [x]
    for i in range(L):
        pe.append(torch.sin(2**i * math.pi * x))
        pe.append(torch.cos(2**i * math.pi * x))

    return torch.cat(pe, dim=-1)

def create_network(L):
    model = nn.Sequential(
        nn.Linear(42, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 3),
        nn.Sigmoid()
    )
    return model


def dataloader(image, N):
    h, w = image.shape[:2]
    idx = torch.randint(0, h*w, (N,))

    y, x = idx // w, idx % w

    colors = image[y, x].to(torch.float32)

    # normalize coordinates in [-1, 1]
    x = (x.float() + 0.5) / w * 2 - 1
    y = (y.float() + 0.5) / h * 2 - 1
    coordinates = torch.stack([x, y], dim=-1)
    coordinates = coordinates.to(torch.float32)
    
    return coordinates, colors

@torch.no_grad()
def render_full(model, H, W, L=10, device="cpu"):
    ys = torch.linspace(0.5, H-0.5, H, device=device)
    xs = torch.linspace(0.5, W-0.5, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([2*(xx/W)-1, 2*(yy/H)-1], dim=-1).reshape(-1,2)
    out = []
    for s in range(0, coords.shape[0], 131072): # LARGE NUMBER
        out.append(model(PE(coords[s:s+131072], L=L)))

    return torch.cat(out).reshape(H, W, 3).clamp(0,1)


def plot_psnr(psnr_logs):
    plt.figure(figsize=(6,4))
    plt.plot(psnr_logs)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title("Training PSNR Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/psnr_curve.png", dpi=300)
    plt.show()

def train_neural_field(img, iters=2000, L=10, minibatch_size=10000):
    #img is np uint8
    H, W = img.shape[:2]
    device = torch.device("cpu")
    img = torch.from_numpy(img).float() / 255.0
    im_flat = img.reshape(-1, img.shape[-1])

    model = create_network(L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    loss_function = nn.MSELoss()

    psnr_logs = []

    model.train()
    for i in range(iters):
        coords, colors = dataloader(img, minibatch_size)

        x_enc = PE(coords, L)

        pred = model(x_enc)
        mse = loss_function(pred, colors)
        psnr_logs.append(PSNR(mse).detach().cpu().item())

        opt.zero_grad(set_to_none=True)
        mse.backward()
        opt.step()
        scheduler.step()

    plot_psnr(psnr_logs)

    return model 


def parse_data(path=f"data/lego_200x200.npz"):
    data = np.load(path)

    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0

    # Cameras for the training images 
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]

    # Validation images: 
    images_val = data["images_val"] / 255.0

    # Cameras for the validation images: [10, 4, 4]
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = data["c2ws_val"]

    # Test cameras for novel-view video rendering: 
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = data["c2ws_test"]

    # Camera focal length
    focal = data["focal"]  # float

    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal

def render_full_image(nerf, dataset, K, c2w, H, W, device):
    nerf.eval()
    with torch.no_grad():
        i, j = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing='xy'
        )  # (H, W)

        dirs = torch.stack([
            (i - K[0, 2]) / K[0, 0],
            (j - K[1, 2]) / K[1, 1],
            torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)

        R = c2w[:3, :3].to(device=device, dtype=torch.float32)
        t = c2w[:3, 3].to(device=device, dtype=torch.float32)

        dirs = dirs.to(device=device, dtype=torch.float32)
        rays_d = dirs @ R.T
        rays_o = t.view(1, 1, 3).expand_as(rays_d)

        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)

        sample_points, t_vals = dataset.sample_points_along_rays(rays_o_flat, rays_d_flat, perturb=False)
        sample_points = PE(sample_points.to(device), 10)
        density, rgb = nerf(sample_points)

        colors, depths = render_volume(rgb, density, t_vals)  # (H*W, 3)

        img = colors.reshape(H, W, 3).cpu().clamp(0, 1)
        depth_img = depths.reshape(H, W).cpu()
    nerf.train()
    return img, depth_img

def render_volume(rgb, density, t_vals):
    t_vals = t_vals.view(-1, t_vals.shape[-1]) 

    dist = t_vals[:, 1:] - t_vals[:, :-1]
    dist = torch.cat([dist, torch.full_like(dist[:, :1], 1e10)], dim=1)

    num_samples = dist.shape[1]
    num_rays = rgb.shape[0] // num_samples

    rgb = rgb.reshape((num_rays, num_samples, 3))
    density = density.reshape((num_rays, num_samples))

    dist = dist.float()
    exp = -(density * dist)
    opacity = 1 - torch.exp(exp)

    transmitence = 1.0 - opacity

    T = torch.cumprod(torch.cat([torch.ones_like(transmitence[:, :1]), transmitence], dim=-1), dim=-1)[:, :-1] 

    prod = T * opacity
    prod = prod[..., None] * rgb

    C = torch.sum(prod, dim=1)

    t_vals = t_vals.reshape(num_rays, num_samples)
    depths = (T * opacity * t_vals).sum(dim=1)

    return C, depths

def plot_losses(mse_train, mse_val, psnr_train, psnr_val, out_dir="results"):
    train_mse_x  = [t[1] for t in mse_train]
    train_mse_y  = [t[0] for t in mse_train]

    val_mse_x    = [t[1] for t in mse_val]
    val_mse_y    = [t[0] for t in mse_val]

    train_psnr_x = [t[1] for t in psnr_train]
    train_psnr_y = [t[0] for t in psnr_train]

    val_psnr_x   = [t[1] for t in psnr_val]
    val_psnr_y   = [t[0] for t in psnr_val]


    plt.figure(figsize=(6,4))
    plt.plot(train_mse_x, train_mse_y, label="train MSE")
    plt.plot(val_mse_x,   val_mse_y,   label="val MSE")
    plt.xlabel("iteration")
    plt.ylabel("MSE")
    plt.title("Training vs Validation MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mse_plot_kitty.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(train_psnr_x, train_psnr_y, label="train PSNR")
    plt.plot(val_psnr_x,   val_psnr_y,   label="val PSNR")
    plt.xlabel("iteration")
    plt.ylabel("PSNR (dB)")
    plt.title("Training vs Validation PSNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/psnr_plot_kitty.png")
    plt.close()

def render_lego_nerf(iters=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = parse_data()

    h, w = images_train[0].shape[:2]

    c2ws_train = torch.from_numpy(c2ws_train).float().to(device)
    images_train = torch.from_numpy(images_train).float().to(device)
    focal = float(focal)
    K = torch.tensor([[focal, 0, w/2],
                      [0, focal, h/2],
                      [0, 0, 1]], dtype=torch.float32, device=device)
    
    c2ws_val = torch.from_numpy(c2ws_val).float().to(device)
    images_val = torch.from_numpy(images_val).float().to(device)
    
    loss_function = nn.MSELoss()
    
    dataset_train = RaysData(images_train, K, c2ws_train)
    dataset_val = RaysData(images_val, K, c2ws_val)
    
    nerf = NeRF().float().to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=5e-4)
    nerf.train()

    mse_train_list = []
    psnr_train_list = []
    mse_val_list = []
    psnr_val_list = []


    for i in range(iters):
        ray_origin_samples, ray_direction_samples, idx = dataset_train.sample_rays(100)
        ray_origin_samples = ray_origin_samples.float().to(device)
        ray_direction_samples = ray_direction_samples.float().to(device)
        
        sample_points, t_vals = dataset_train.sample_points_along_rays(ray_origin_samples, ray_direction_samples, perturb=True)
        sample_points = sample_points.to(device).float()
        t_vals = t_vals.to(device).float()
        
        colors_target = dataset_train.pixels[idx].to(device).float()
        sample_points = PE(sample_points, 10)
        density, rgb = nerf(sample_points)

        dist_between_pts = t_vals[:, 1:] - t_vals[:, :-1]
        dist_between_pts = torch.cat([dist_between_pts, torch.full_like(dist_between_pts[:, :1], 1e10)], dim=-1)

        colors_pred = render_volume(rgb, density, dist_between_pts)
        
        mse_train = loss_function(colors_pred, colors_target)
        psnr_train = PSNR(mse_train)

        if i % 50 == 0:
            mse_train_list.append((mse_train.item(), i))
            psnr_train_list.append((psnr_train.item(), i))

        optimizer.zero_grad()
        mse_train.backward()
        optimizer.step()


        if i % 500 == 0:
            print(f"Iteration {i} loss: {mse_train}")
            with torch.no_grad():
            
                H, W = images_train[0].shape[:2]
                img_pred = render_full_image(nerf, dataset_val, K, c2ws_val[0], H, W, device)
                img_target = images_val[0].float().to(device)

                val_mse = F.mse_loss(img_pred, img_target)
                mse_val_list.append((val_mse.item(), i))
                psnr_val_list.append((PSNR(val_mse).item(), i))

                plt.imsave(f"results/rendered_image_{i}.jpg", img_pred.detach().cpu().numpy())
    
    torch.save(nerf.state_dict(), "nerf_weights.pth")
    plot_losses(mse_train_list, mse_val_list, psnr_train_list, psnr_val_list)


def render_test_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nerf = NeRF().to(device)
    nerf.load_state_dict(torch.load('nerf_weights.pth'))
    nerf.eval()

    with torch.no_grad():
        images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = parse_data()
        H, W = images_train[0].shape[:2]

        c2ws_test = torch.from_numpy(c2ws_test).float().to(device)
        images_val = torch.from_numpy(images_train).float().to(device)
        c2ws_val = torch.from_numpy(c2ws_train).float().to(device)
    
        focal = float(focal)
        K = torch.tensor([[focal, 0, W/2],
                        [0, focal, H/2],
                        [0, 0, 1]], dtype=torch.float32, device=device)

        dataset_val = RaysData(images_val, K, c2ws_val)

        for i in range(len(c2ws_test)):
            c2w = c2ws_test[i]
            img_pred, depths = render_full_image(nerf, dataset_val, K, c2w, H, W, device)

            plt.imsave(f"results/depth_{i}.jpg", depths.detach().cpu().numpy(), cmap="gray")
            print(f"Saved image {i}")


### GIF function from ChatGPT
def make_gif_from_results(
        directory="results",
        prefix="depth",
        output="depth.gif",
        fps=10
    ):
    # Collect matching images
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(files) == 0:
        raise ValueError("No images found with prefix '{}' in '{}'".format(prefix, directory))

    files = natsorted(files)
    # Load images
    frames = [imageio.imread(f) for f in files]

    # Save GIF
    imageio.mimsave(output, frames, fps=fps, loop=0)

def look_at_origin(pos):
  # Camera looks towards the origin
  forward = -pos / np.linalg.norm(pos)  # Normalize the direction vector

  # Define up vector (assuming y-up)
  up = np.array([0, 1, 0])

  # Compute right vector using cross product
  right = np.cross(up, forward)
  right = right / np.linalg.norm(right)

  # Recompute up vector to ensure orthogonality
  up = np.cross(forward, right)

  # Create the camera-to-world matrix
  c2w = np.eye(4)
  c2w[:3, 0] = right
  c2w[:3, 1] = up
  c2w[:3, 2] = forward
  c2w[:3, 3] = pos

  return c2w

def rot_x(phi):
    return np.array([
        [math.cos(phi), -math.sin(phi), 0, 0],
        [math.sin(phi), math.cos(phi), 0, 0],
        [0,0,1,0],
        [0,0,0,1],
    ])

def render_teapot(iters=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = parse_data(path=f"data.npz")
    h, w = images_train[0].shape[:2]

    c2ws_train = torch.from_numpy(c2ws_train).float().to(device)
    images_train = torch.from_numpy(images_train).float().to(device)
    focal = float(focal)
    K = torch.tensor([[focal, 0, w/2],
                      [0, focal, h/2],
                      [0, 0, 1]], dtype=torch.float32, device=device)
    
    c2ws_val = torch.from_numpy(c2ws_val).float().to(device)
    images_val = torch.from_numpy(images_val).float().to(device)

    loss_function = nn.MSELoss()
    
    dataset_train = RaysData(images_train, K, c2ws_train)
    dataset_val = RaysData(images_val, K, c2ws_val)
    
    nerf = NeRF().float().to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=5e-4)
    nerf.train()

    mse_train_list = []
    psnr_train_list = []
    mse_val_list = []
    psnr_val_list = []

    for i in range(iters):
        ray_origin_samples, ray_direction_samples, idx = dataset_train.sample_rays(100)
        ray_origin_samples = ray_origin_samples.float().to(device)
        ray_direction_samples = ray_direction_samples.float().to(device)
        
        sample_points, t_vals = dataset_train.sample_points_along_rays(ray_origin_samples, ray_direction_samples, perturb=True)
        sample_points = sample_points.to(device).float()
        t_vals = t_vals.to(device).float()
        
        colors_target = dataset_train.pixels[idx].to(device).float()
        sample_points = PE(sample_points, 10)
        density, rgb = nerf(sample_points)

        dist_between_pts = t_vals[:, 1:] - t_vals[:, :-1]
        dist_between_pts = torch.cat([dist_between_pts, torch.full_like(dist_between_pts[:, :1], 1e10)], dim=-1)

        colors_pred, depths_pred = render_volume(rgb, density, dist_between_pts)
        
        mse_train = loss_function(colors_pred, colors_target)
        psnr_train = PSNR(mse_train)

        if i % 50 == 0:
            mse_train_list.append((mse_train.item(), i))
            psnr_train_list.append((psnr_train.item(), i))

        optimizer.zero_grad()
        mse_train.backward()
        optimizer.step()

        if i % 400 == 0:
            print(f"Iteration {i} loss: {mse_train}")
            with torch.no_grad():
            
                H, W = images_train[0].shape[:2]
                img_pred, depth = render_full_image(nerf, dataset_val, K, c2ws_val[3], H, W, device)
                img_target = images_val[3].float().to(device)

                val_mse = F.mse_loss(img_pred, img_target)
                mse_val_list.append((val_mse.item(), i))
                psnr_val_list.append((PSNR(val_mse).item(), i))

                #plt.imsave(f"results/rendered_teapot_{i}.jpg", img_pred.detach().cpu().numpy())
    
    torch.save(nerf.state_dict(), "teapot_nerf_weights2.pth")
    plot_losses(mse_train_list, mse_val_list, psnr_train_list, psnr_val_list)

def show_depths_gif():
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = parse_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerf = NeRF().to(device)
    nerf.load_state_dict(torch.load("nerf_weights.pth", map_location=device))
    nerf.eval()

    h, w = images_val[0].shape[:2]

    c2ws_test = torch.from_numpy(c2ws_val).float().to(device)
    images_val = torch.from_numpy(images_val).float().to(device)
    focal = float(focal)

    K = torch.tensor([[focal, 0, w/2],
                      [0, focal, h/2],
                      [0, 0, 1]], dtype=torch.float32, device=device)
    
    dataset = RaysData(images_val, K, c2ws_val)
    
    # TODO: Change start position to a good position for your scene such as 
    # the translation vector of one of your training camera extrinsics
    START_POS = np.array(c2ws_test[4][:3, 3])
    NUM_SAMPLES = 60

    frames = []
    for phi in np.linspace(360., 0., NUM_SAMPLES, endpoint=False):
        c2w = look_at_origin(START_POS)
        extrinsic = rot_x(phi/180.*np.pi) @ c2w

        extrinsic = torch.from_numpy(extrinsic)
        
        # Generate view for this camera pose
        # TODO: Add code for generating a view with your model from the current extrinsic
        img = render_full_image(nerf, dataset, K, extrinsic, h, w, device)
        frame = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
    
    imageio.mimsave("teapot2.gif", frames, fps=20, loop=0)

def show_teapot_gif():
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = parse_data(path=f"data.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerf = NeRF().to(device)
    nerf.load_state_dict(torch.load("teapot_nerf_weights.pth", map_location=device))
    nerf.eval()

    h, w = images_val[0].shape[:2]

    c2ws_val = torch.from_numpy(c2ws_val).float().to(device)
    images_val = torch.from_numpy(images_val).float().to(device)
    focal = float(focal)

    K = torch.tensor([[focal, 0, w/2],
                      [0, focal, h/2],
                      [0, 0, 1]], dtype=torch.float32, device=device)
    
    dataset = RaysData(images_val, K, c2ws_val)
    
    # TODO: Change start position to a good position for your scene such as 
    # the translation vector of one of your training camera extrinsics
    START_POS = np.array(c2ws_test[4][:3, 3])
    NUM_SAMPLES = 60

    frames = []
    for phi in np.linspace(360., 0., NUM_SAMPLES, endpoint=False):
        c2w = look_at_origin(START_POS)
        extrinsic = rot_x(phi/180.*np.pi) @ c2w

        extrinsic = torch.from_numpy(extrinsic)
        
        # Generate view for this camera pose
        # TODO: Add code for generating a view with your model from the current extrinsic
        img = render_full_image(nerf, dataset, K, extrinsic, h, w, device)
        frame = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
    
    imageio.mimsave("teapot2.gif", frames, fps=20, loop=0)


def main():
    #part 0
    #save_dataset()

    #part 1
    #im_path = 'data/lazaro.jpeg'
    #im = plt.imread(im_path)
    #model = train_neural_field(im)

    #part 2
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = parse_data()

    h, w = images_train[0].shape[:2]

    c2ws_train = torch.from_numpy(c2ws_train).float().to(device)
    images_train = torch.from_numpy(images_train).float().to(device)
    focal = float(focal)
    K = torch.tensor([[focal, 0, w/2],
                      [0, focal, h/2],
                      [0, 0, 1]], dtype=torch.float32, device=device) """
    
    #visualise1(images_train, K, c2ws_train)
    #visualise2(images_train, K, c2ws_train)

    #render_lego_nerf()

    #render_test_set()

    #render_babushka()
    #render_teapot()
    #render_test_set()
    #show_teapot_gif()
    
    
    #show_teapot_gif()

    render_test_set()
    #make_gif_from_results()

    
    
    
    


if __name__ == "__main__":
    main()



