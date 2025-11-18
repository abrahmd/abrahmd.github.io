import numpy as np
import torch

class RaysData():

    def __init__(self, images, K, c2ws):
        self.images = images
        self.K = K
        self.c2ws = c2ws
        self.h = images[0].shape[0]
        self.w = images[0].shape[1]
        self.n_images = len(images)
        self.uvs = torch.tensor([[i % self.w, i // self.w] for i in range(self.w * self.h)])

        pixels = []
        for i in range(self.n_images):
            img = images[i]
            for u, v in self.uvs:
                pixels.append(img[v, u])
        self.pixels = torch.stack(pixels)

        self.ray_origins, self.ray_directions = self.uvs_to_rays(self.uvs, self.c2ws)
        

    """
        c2ws is (N, 4,4)
        x_cs is (H*W, 3)
        Converts batch camera coordinates to world coordinates in shape (N*H*W, 3) 
    """
    @staticmethod
    def __transform_c2w(c2ws, x_cs):
        n = x_cs.shape[0]
        ones = torch.ones((n, 1), device=x_cs.device, dtype=torch.float32)
        x_cs = x_cs.to(torch.float32)

        x_c_hs = torch.cat([x_cs, ones], dim=1)

        # nij = shape c2ws (N, 4, 4)
        # mj  = shape x_c_hs (H*W, 4)
        # nmi = shape output (N, HW, 4) - world coordinates for each pixel in each image
        x_w_hs = torch.einsum('nij,mj->nmi', c2ws, x_c_hs) 
        x_w_hs = x_w_hs.reshape(-1,x_w_hs.shape[-1])

        return x_w_hs[:, :-1] # output (N*HW, 3)
    
    """
    uv is (N*H*W, 2)
    s = zc, the depth of uv along the optical axis
    """
    def transform_pixel_to_camera(self, uv, s):
        n = uv.shape[0]
        uv_h = np.concatenate([uv, np.ones((n, 1))], axis=1) # convert to homogeneous coords
        uv_h = s * uv_h 
        K_inv = np.linalg.inv(self.K)

        cam_coords = (K_inv @ uv_h.T).T
        
        return cam_coords

    """
    
    """
    def uvs_to_rays(self, uvs, c2ws):
        t = self.c2ws[:, :-1, -1]
        ray_origins = t # (N, 3) is world coordinates
        ray_origins = np.repeat(ray_origins, self.h * self.w, axis=0)

        uvs_c = self.transform_pixel_to_camera(uvs, 1)
        uvs_w = RaysData.__transform_c2w(c2ws, uvs_c) # (N*HW, 3)

        diff = uvs_w - ray_origins
        norms = np.linalg.norm(diff, axis = 1, keepdims=True)
        ray_directions = diff / np.maximum(norms, 1e-8)
        
        return ray_origins, ray_directions


    """
        Sample N rays
    """
    def sample_rays(self, N=10):
        #idx = np.random.choice(self.n_images * self.h * self.w, N, replace=False)
        #m = idx // (self.h * self.w) #which image
        #remainder = idx % (self.h * self.w)
        #x = remainder % self.w
        #y = remainder // self.w

        #u = x.astype(np.float64) + 0.5
        #v = y.astype(np.float64) + 0.5
        #uv = np.stack([u, v], axis = 1)
        #c2ws_selected = self.c2ws[m]
        #rays_o, rays_d = self.uvs_to_rays(c2ws_selected, uv)

        idx = np.random.choice(self.ray_origins.shape[0], N)

        ray_origins_selected = self.ray_origins[idx]
        ray_directions_selected = self.ray_directions[idx]

        return ray_origins_selected, ray_directions_selected, idx

    """
        Returns points in (n_samples*N, 3)
    """
    @staticmethod
    def sample_points_along_rays(rays_o, rays_d, perturb=True, n_samples=64, near=0.1, far=1.0):
        N = rays_o.shape[0]
        t = torch.linspace(near, far, n_samples)

        ### adding for density
        t = t.unsqueeze(0).expand(N, -1)

        if perturb:
            bin_width = (far-near) / n_samples # 
            noise = np.random.rand(N, n_samples) * bin_width # change bin_width by factor 0-1
            t = t[None, :] + noise
        else:
            t = t[None, :] 

        
        # result of below expression is pts.shape = (N, n_samples, 3) N rays with n_samples of (x,y,z) points sampled per
        # Syntax creates new axis
        pts = rays_o[:, None, :] + t[..., None] * rays_d[:, None, :]
        pts = pts.reshape(-1, pts.shape[-1])

        return pts, t