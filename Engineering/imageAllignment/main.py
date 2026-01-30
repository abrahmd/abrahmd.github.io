# CS180 (CS280A): Project 1 starter Python code

# Sources:
# Guassian pyramid: https://www.youtube.com/watch?v=1GFQ4V8cV0o
# Laplacian pyramid: https://www.youtube.com/watch?v=QNxJoglVS1Q
# Image filtering: https://www.youtube.com/watch?v=6v8dNtknOSM


# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import cv2
import skimage as sk
import skimage.io as skio
from skimage.util import img_as_ubyte
from skimage.transform import rescale, resize
#from IPython.display import display, clear_output
from PIL import Image

#delete
import matplotlib.pyplot as plt

#may try to replace
from skimage.registration import phase_cross_correlation

def NCC_score(im1, im2) -> float:
    #im1 = im1.astype(np.float32).ravel()
    #im2 = im2.astype(np.float32).ravel()

    im1 = im1 - im1.mean()
    im2 = im2 - im2.mean()
    
    dot = np.dot(im1.ravel(), im2.ravel())
    den = np.linalg.norm(im1.ravel()) * np.linalg.norm(im2.ravel()) + 1e-8

    return float(dot/den)
# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
"""
def local_search(im1, im2, delta, radius) -> int:
    print("delta: ", delta)
    dx_i, dy_i = delta
    best_score = -float('inf')

    print(f"im2 shape: {im2.shape}")
    
    for dy in range(dy_i - radius, dy_i + radius + 1):
        for dx in range(dx_i - radius, dx_i + radius + 1):
            im2_shifted = np.roll(im2, shift=(dy, dx), axis=(0,1))
            score = NCC_score(im1, im2_shifted)
            if score > best_score:
                delta = (dy, dx)
                best_score = score
    return delta
"""
            
# RETURNS DY, DX
def local_search(im1, im2, radius) -> tuple:

    max_score = -float('inf')
    best_displacement = (0,0)
    for ax0 in range(-radius, radius):
        for ax1 in range(-radius, radius):
            im2_shifted = np.roll(im2, shift=(ax0, ax1), axis=(0,1))
            score = NCC_score(im1, im2_shifted)
            
            if score > max_score:
                #print("New best score: ", score)
                best_displacement = (ax0, ax1)
                max_score = score

    return best_displacement

def align(g_pyramid1, g_pyramid2, radius=8) -> tuple:
    levels = len(g_pyramid1)
    dy, dx = local_search(g_pyramid1[-1], g_pyramid2[-1], radius)

    for l in range(levels-2, -1, -1):
        dy *= 2
        dx *= 2
        im2 =  np.roll(g_pyramid2[l], shift=(dy, dx), axis=(0,1))

        shift = local_search(g_pyramid1[l], im2, radius)
        dy += shift[0]
        dx += shift[1]    
    
    return (dy, dx)


def laplacian_pyramid(g_pyramid) -> list:
    pyramid = []
    for i in range(len(g_pyramid)-1):
        l1 = g_pyramid[i]
        l2 = g_pyramid[i+1]
        l2_upscaled = cv2.resize(l2, (l1.shape[1], l1.shape[0]))
        pyramid.append(l1 - l2_upscaled)
    pyramid.append(g_pyramid[-1])

    return pyramid

def gaussian_pyramid(im, K) -> list:
    pyramid = [im]
    for i in range(K-1):
        im2 = cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2)))
        pyramid.append(im2)
        im = im2
    return pyramid

def auto_contrast(im, low=1, high=99):
    #if np.issubdtype(img.dtype, np.integer):
    #    x /= np.iinfo(img.dtype).max

    
    for c in range(im.shape[2]):
        lo, hi = np.percentile(im[...,c], (low, high))
        im[...,c] = np.clip((im[...,c]-lo)/(hi-lo+1e-8), 0, 1)

    return im


#im_names = ['church.tif', 'emir.tif', 'harvesters.tif', 'icon.tif', 'italil.tif', 'lastochikino.tif', 'lugano.tif', 'melons.tif', 'self_portrait.tif', 'siren.tif', 'three_generations.tif']

im_names = ['cathedral.jpg', 'monastery.jpg', 'tobolsk.jpg']
for name in im_names:
    imname = f'data/{name}'

    im = skio.imread(imname) 
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int64)

    b = im[:height] 
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # Cropping
    y, x = b.shape
    b = b[int(0.05*y):int(0.95*y), int(0.05*x):int(0.95*x)]
    g = g[int(0.05*y):int(0.95*y), int(0.05*x):int(0.95*x)]
    r = r[int(0.05*y):int(0.95*y), int(0.05*x):int(0.95*x)]


    N_levels = 1 + int(np.log2(height/64)) #final image in pyramid should be ~ 64 x 64
    r_g_pyramid = gaussian_pyramid(r, N_levels)
    g_g_pyramid = gaussian_pyramid(g, N_levels)
    b_g_pyramid = gaussian_pyramid(b, N_levels)

    r_LP = laplacian_pyramid(r_g_pyramid)
    g_LP = laplacian_pyramid(g_g_pyramid)
    b_LP = laplacian_pyramid(b_g_pyramid)

    #dy_r, dx_r = align(g_g_pyramid, r_g_pyramid) 
    #dy_b, dx_b = align(g_g_pyramid, b_g_pyramid) 
    dy_r, dx_r = align(g_LP, r_LP) 
    dy_b, dx_b = align(g_LP, b_LP) 


    r = np.roll(r, shift=(dx_r, dy_r), axis=(1,0))
    b = np.roll(b, shift=(dx_b, dy_b), axis=(1,0))
    #fake_color_channel = np.zeros((r.shape[0], r.shape[1]))

    rgb = np.dstack([r, g, b])

    #rgb = auto_contrast(rgb)

    rgb8 = (rgb * 255).astype(np.uint8)

    #rgb_GP = gaussian_pyramid(rgb8, 6)
    #rgb_LP = laplacian_pyramid(rgb_GP)

    #skio.imshow(rgb_LP[-1])
    #skio.show()
    


    skio.imsave(f'out_path/{name}', rgb8)

    dst = f"out_path_jpgs/{name.removesuffix(".tif")}.jpg"
    with Image.open(f'out_path/{name}') as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        im.save(dst, "JPEG", quality=92, optimize=True, progressive=True)

    skio.imshow(rgb8)
    skio.show() 


"""
dx_r, dy_r = align(g_g_pyramid, r_g_pyramid, 5)
dx_b, dy_b = align(g_g_pyramid, b_g_pyramid, 5)

print(f"dx_r: {dx_r}, dy_r: {dy_r}")
print(f"dx_r: {dx_b}, dy_r: {dy_b}")


r = np.roll(r, (dx_r, dy_r))
b = np.roll(g, (dx_b, dy_b))

fake_color_channel = np.zeros((r.shape[0], r.shape[1]))

rgb = np.dstack([r, g, fake_color_channel])
rgb8 = (rgb * 255).astype(np.uint8)

#
# display the image
skio.imshow(rgb8)
skio.show() 
"""

################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################



"""
def show_images(images, titles=None, cols=3):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            if img.ndim == 2:   # grayscale
                ax.imshow(img, cmap='gray')
            else:               # RGB/RGBA
                ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def lucas_kanade(r, g, b) -> np.array:
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)

    #use g as reference
    r_shift, _, _ = phase_cross_correlation(g, r, upsample_factor=10)
    b_shift, _, _ = phase_cross_correlation(g, b, upsample_factor=10)


    print(f"Shift red: {r_shift}")
    print(f"Shift blue: {b_shift}")

    r = np.roll(r, (round(r_shift[0], round(r_shift[1]))))
    b = np.roll(b, (round(b_shift[0], round(b_shift[1]))))


    r = (r * 255).astype(np.uint8)
    g = (g * 255).astype(np.uint8)
    b = (b * 255).astype(np.uint8)
    

    #skio.imshow(r)
    #skio.show() 

    rgb = np.dstack([r, g, b])
    rgb8 = (rgb * 255).astype(np.uint8)
    return rgb8
"""
# Here we blur our image before downsampling
# Blurring is done to avoid aliasing
# Possible issues: Kernel should be flipped? Research this?
#                  Losing edges of original, no padding
def blur_convolution(im, kernel) -> np.array:
    im2 = np.zeros([im.shape[0], im.shape[1]]).astype(np.uint8)
    k_size = kernel.shape[0]

    im = np.pad(im, ((1, 1), (1,1)), mode='reflect')
    
    for i in range(1, im.shape[0]-2): # Assumes 3x3 kernel, OPTIMIZE
        for j in range(1, im.shape[1]-2):
            im2[i-1, j-1] = np.sum(im[i:i+k_size, j:j+k_size] * kernel).astype(np.uint8)

    return im2

def gaussian_blur(im) -> np.array:
    gaussianKernel = np.array([[1, 2, 1],[ 2, 4, 2], [1, 2, 1]]) / 16.0 # allegedly this should be flipped
    return blur_convolution(im, gaussianKernel)


def image_matching_scoreL2(im1, im2) -> int:
    return np.sqrt(np.sum((im1- im2)**2))