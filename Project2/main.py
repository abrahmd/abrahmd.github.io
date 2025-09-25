import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2
import math
from hybrid_python.align_image_code import align_images

######################################################### 1.1 ################################################################
def convolve_four_loops(im, kernel) -> np.array:
    ### TODO: FLIP kernel
    im_convolved = np.zeros((im.shape[0], im.shape[1]))
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            sum = 0.0
            y_lo = y - kernel.shape[0]//2
            y_hi = y + kernel.shape[0]//2 + 1
            x_lo = x - kernel.shape[1] // 2
            x_hi = x + kernel.shape[1] // 2 + 1
            if y_lo >= 0 and y_hi <= im.shape[0] and x_lo >= 0 and x_hi <= im.shape[1]: #If the kernel is within the image
                for yk in range(-kernel.shape[0]//2, kernel.shape[0]//2 + 1):
                    for xk in range(-kernel.shape[1] // 2, kernel.shape[1] // 2 + 1):
                        sum += im[y+yk][x+xk] * kernel[yk][xk]
            sum = np.clip(sum, 0, 255).astype(np.uint8)
            im_convolved[y][x] = sum
    
    return im_convolved

def convolve_two_loops(im, kernel) -> np.array:
    ### TODO: FLIP kernel
    im_convolved = np.zeros((im.shape[0], im.shape[1]))
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            y_lo = y - kernel.shape[0]//2
            y_hi = y + kernel.shape[0]//2 + 1
            x_lo = x - kernel.shape[1] // 2
            x_hi = x + kernel.shape[1] // 2 + 1
            if y_lo >= 0 and y_hi <= im.shape[0] and x_lo >= 0 and x_hi <= im.shape[1]:
                im_convolved[y][x] = np.sum(kernel * im[y_lo: y_hi, x_lo:x_hi])

    return im_convolved


def part1_1():
    imname = 'data/self.jpg'
    im = skio.imread(imname) 
    im = im[:,:,0] #black and white
    im = np.pad(im, (1,1), mode='edge')

    box_filter = 1.0/9 * np.ones((9,9))
    dx = np.array([-1, 0, 1]).reshape(1,-1) / 2
    dy = np.array([-1, 0, 1]).reshape(-1,1) / 2

    im_boxf = convolve2d(im, box_filter)
    im_dx = convolve2d(im, dx)
    im_dy = convolve2d(im, dy)

    plt.imsave(f'data/BW_self_boxfiltered.jpg', im_boxf, cmap='gray')
    plt.imshow(im_boxf, cmap='gray')
    plt.show()

    plt.imsave(f'data/BW_selfDX.jpg', im_dx, cmap='gray')
    plt.imshow(im_dx, cmap='gray')
    plt.show()

    plt.imsave(f'data/BW_selfDY.jpg', im_dy, cmap='gray')
    plt.imshow(im_dy, cmap='gray')
    plt.show()


################################################## Part 1.2 ##########################################################

def part1_2():
    imname = 'data/cameraman.jpg'
    im = skio.imread(imname) 
    im = im[:,:,0] #black and white
    im = np.pad(im, (1,1), mode='edge')

    dx = np.array([[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]]) / 2
    dy = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [0,-1, 0]]) / 2
    

    im_dx = convolve2d(im, dx)
    im_dy = convolve2d(im, dy)

    im_gradmag = np.sqrt(im_dx**2 + im_dy**2)

    plt.imshow(im_gradmag, cmap='gray')
    plt.show()

    threshold = 45
    mask = (im_gradmag > threshold)
    im_gradmag_bin = np.zeros_like(im_gradmag)
    im_gradmag_bin[mask] = 255

    plt.imshow(im_gradmag_bin, cmap='gray')
    plt.show()


"""
What difference do you see? -> Less noise = lower threshold
"""
def part1_3():
    imname = 'data/cameraman.jpg'
    im = skio.imread(imname) 
    im = im[:,:,0] #black and white
    im = np.pad(im, (1,1), mode='edge')

    G = cv2.getGaussianKernel(ksize=5, sigma=2/3)
    G = G @ G.T
    
    
    dx = np.array([[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]]) / 2
    dy = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [0,-1, 0]]) / 2
    
    im_G = convolve2d(im, G)
    im_Gdx = convolve2d(im_G, dx)
    im_Gdy = convolve2d(im_G, dy)

    im_gradmag = np.sqrt(im_Gdx**2 + im_Gdy**2)

    plt.imshow(im_gradmag, cmap='gray')
    plt.show()

    threshold = 35
    mask = (im_gradmag > threshold)
    im_gradmag_bin = np.zeros_like(im_gradmag)
    im_gradmag_bin[mask] = 255

    plt.imshow(im_gradmag_bin, cmap='gray')
    plt.show()

    ########################### 
    
    ## Aproach 1
    G_dx = convolve2d(G, dx) ### This seems to work
    G_dy = convolve2d(G, dy)
    
    im_Gdx = convolve2d(im, G_dx)
    im_Gdy = convolve2d(im, G_dy)

    im_gradmag = np.sqrt(im_Gdx**2 + im_Gdy**2)
    
    threshold = 35
    mask = (im_gradmag > threshold)
    im_Gdxdy_bin = np.zeros_like(im_gradmag)
    im_Gdxdy_bin[mask] = 255    

    plt.imshow(im_Gdxdy_bin, cmap='gray')
    plt.show()

def show_images(images, titles=None, cols=3, same_scale=False):
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axs = np.atleast_1d(axs).ravel()

    # Optional: consistent grayscale scaling across all images
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
    
### Oberserve that in the final part, we get the original image, mathematically they are equivalent
def part2_1():
    imname = 'data/taj.jpg'
    im = skio.imread(imname) 
    im = np.pad(im, (1,1), mode='edge')
    
    G = cv2.getGaussianKernel(ksize=5, sigma=2/3)
    G = G @ G.T
        
    im_c = im[:,:,1:4] # RGB channels
    
    im_blur_r = convolve2d(im_c[:,:,0], G, mode='same').astype(np.uint8)
    im_blur_g = convolve2d(im_c[:,:,1], G, mode='same').astype(np.uint8)
    im_blur_b = convolve2d(im_c[:,:,2], G, mode='same').astype(np.uint8)

    im_blur = np.stack([im_blur_r, im_blur_g, im_blur_b], axis=2)

    im_highfreq = im_c.astype(np.int32) - im_blur.astype(np.int32)
    im_highfreq = np.clip(im_highfreq, 0, 255).astype(np.uint8)

    im_sharp = im_c.astype(np.int32) + im_highfreq.astype(np.int32)
    im_sharp = np.clip(im_sharp, 0, 255).round().astype(np.uint8)
    
    plt.imsave(f'data/taj_highfrequencies.jpg', im_highfreq)
    plt.imsave(f'data/taj_tajsharpened.jpg', im_sharp)

    ### blur an image, and try to sharpen it
    im_self = 'data/BW_self.jpg'
    ims = skio.imread(im_self)
    ims = np.pad(ims, (1,1), mode='edge')

    print(f'ims.shape: ', ims.shape)
    print(f'G.shape: ', G.shape)
    ims_blur = convolve2d(ims, G, mode='same')
    ims_highfreq = ims.astype(np.int32) - ims_blur.astype(np.int32)
    ims_highfreq = np.clip(ims_highfreq, 0, 255).astype(np.uint8)
    ims_blursharp = ims_blur.astype(np.int32) + ims_highfreq.astype(np.int32)
    ims_blursharp = np.clip(ims_blursharp, 0, 255).astype(np.uint8)

    show_images([im_c, im_blur, im_highfreq, im_sharp, ims, ims_blur, ims_highfreq, ims_blursharp])

    
def part2_2x():
    im1 = 'hybrid_python/DerekPicture.jpg'
    im2 = 'hybrid_python/nutmeg.jpg'

    im1 = skio.imread(im1) 
    im2 = skio.imread(im2)

    

    im3, im4 = align_images(im2, im1)

    im3 = (im3 * 255).astype(np.uint8)
    #im4 = (im4 * 255).astype(np.uint8)

    G1 = cv2.getGaussianKernel(ksize=10, sigma=5)
    G1 = G1 @ G1.T

    G2 = cv2.getGaussianKernel(ksize=10, sigma=5)
    G2 = G2 @ G2.T


    im3_blur_r = convolve2d(im3[:,:,0], G1, mode='same').astype(np.uint8)
    im3_blur_g = convolve2d(im3[:,:,1], G1, mode='same').astype(np.uint8)
    im3_blur_b = convolve2d(im3[:,:,2], G1, mode='same').astype(np.uint8)
    im3_blur = np.stack([im3_blur_r, im3_blur_g, im3_blur_b], axis=2)

    show_images([im3, im3_blur_r, im3_blur_g, im3_blur_b, im3_blur])

    im4_blur_r = convolve2d(im4[:,:,0], G2, mode='same').astype(np.uint8)
    im4_blur_g = convolve2d(im4[:,:,1], G2, mode='same').astype(np.uint8)
    im4_blur_b = convolve2d(im4[:,:,2], G2, mode='same').astype(np.uint8)
    im4_blur = np.stack([im4_blur_r, im4_blur_g, im4_blur_b], axis=2)

    im4_highfreq = im4.astype(np.int32) - im4_blur.astype(np.int32)
    im4_highfreq = 4 * np.clip(im4_highfreq, 0, 255).astype(np.uint8)

    im5 = im3_blur.astype(np.int32) + im4_highfreq.astype(np.int32)
    im5 = np.clip(im5, 0, 255).astype(np.uint8)


    show_images([im3, im4, im3_blur, im4_highfreq, im5])

"""
Align images
"""
def part2_21():
    im1 = 'hybrid_python/DerekPicture.jpg'
    im2 = 'hybrid_python/nutmeg.jpg'

    im1 = skio.imread(im1) 
    im2 = skio.imread(im2)

    

    im3, im4 = align_images(im2, im1)

    im3 = (im3 * 255).astype(np.uint8) 
    
    im3 = cv2.cvtColor(im3, cv2.COLOR_RGB2BGR) #im3 is CAT
    im3 = im3[:,:,0]
    im4 = cv2.cvtColor(im4, cv2.COLOR_RGB2BGR) # im4 is MAN
    im4 = im4[:,:,0]

    G1 = cv2.getGaussianKernel(ksize=73, sigma=12) ## for the cat, LOW PASS ONLY
    G1 = G1 @ G1.T

    G2 = cv2.getGaussianKernel(ksize=72, sigma=12) ## for the man
    G2 = G2 @ G2.T


    im3_blur = convolve2d(im3, G1, mode='same').astype(np.uint8)
    im4_blur = convolve2d(im4, G2, mode='same').astype(np.uint8)

    im3_highfreq = im3.astype(np.int32) - im3_blur.astype(np.int32)
    im3_highfreq = np.clip(im3_highfreq, 0, 255).astype(np.uint8)

    im5 = im4_blur.astype(np.int32) + im3_highfreq.astype(np.int32)
    im5 = np.clip(im5, 0, 255).astype(np.uint8)

    show_images([im3, im4, im3_blur, im3_highfreq, im5])

"""
Frequency analysis - show Fourier transform of input images, filtered images, hybrid image
"""
def part2_22():
    im1 = 'hybrid_python/DerekPicture.jpg'
    im2 = 'hybrid_python/nutmeg.jpg'

    im1 = skio.imread(im1) 
    im2 = skio.imread(im2)

    im1_aligned, im2_aligned = align_images(im2, im1)

    im1_aligned = (im1_aligned * 255).astype(np.uint8) 
    
    im1_aligned_bw = cv2.cvtColor(im1_aligned, cv2.COLOR_RGB2BGR) #im3 is CAT
    im1_aligned_bw = im1_aligned_bw[:,:,0]
    im2_aligned_bw = cv2.cvtColor(im2_aligned, cv2.COLOR_RGB2BGR) # im4 is MAN
    im2_aligned_bw = im2_aligned_bw[:,:,0]

    G1 = cv2.getGaussianKernel(ksize=73, sigma=12) ## for the cat, LOW PASS ONLY
    G1 = G1 @ G1.T

    G2 = cv2.getGaussianKernel(ksize=72, sigma=12) ## for the man
    G2 = G2 @ G2.T


    im1_al_bw_blur = convolve2d(im1_aligned_bw, G1, mode='same').astype(np.uint8)
    im2_al_bw_blur = convolve2d(im2_aligned_bw, G2, mode='same').astype(np.uint8)

    im1_al_bw_blur_highfq = im1_aligned_bw.astype(np.int32) - im1_al_bw_blur.astype(np.int32)
    im1_al_bw_blur_highfq = np.clip(im1_al_bw_blur_highfq, 0, 255).astype(np.uint8)

    im_hybrid = im2_al_bw_blur.astype(np.int32) + im1_al_bw_blur_highfq.astype(np.int32)
    im_hybrid = np.clip(im_hybrid, 0, 255).astype(np.uint8)

    im1_fourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)[:,:,0]))))
    im2_fourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)[:,:,0]))))

    im1_filter_fourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_al_bw_blur_highfq))))
    im2_filter_fourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_al_bw_blur))))

    im_hybrid_fourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im_hybrid))))

    show_images([im1_fourier, im2_fourier, im1_filter_fourier, im2_filter_fourier, im_hybrid_fourier])


def create_gaussian_stack(im, n_levels):
    G = cv2.getGaussianKernel(ksize=19, sigma=3).astype(np.float32)
    G = G @ G.T
    stack = [im]
    for i in range(n_levels-1):
        im_blur_r = convolve2d(im[:,:,0], G, mode='same')#.astype(np.uint8)
        im_blur_g = convolve2d(im[:,:,1], G, mode='same')#.astype(np.uint8)
        im_blur_b = convolve2d(im[:,:,2], G, mode='same')#.astype(np.uint8)
        im_blur = np.stack([im_blur_r, im_blur_g, im_blur_b], axis=2)
        stack.append(im_blur)
        im = im_blur
    return stack

def create_laplacian_stack(gaussian_stack):
    stack = []
    for i in range(len(gaussian_stack)-1):
        dif = gaussian_stack[i] - gaussian_stack[i+1]
        #normalization ???
        #dif = dif - np.min(dif)
        #dif = dif / np.max(dif)

        stack.append(dif)
    stack.append(gaussian_stack[-1])#.astype(np.float32))
    return stack

def part2_3():
    #create gaussian stack
    im1 = 'data/apple.jpeg'
    im2 = 'data/orange.jpeg'

    im1 = skio.imread(im1) 
    im2 = skio.imread(im2)

    n_levels = 4
    im1_gaussian_stack = create_gaussian_stack(im1, n_levels)
    im1_laplacian_stack = create_laplacian_stack(im1_gaussian_stack)

    show_images(im1_gaussian_stack + im1_laplacian_stack)
    
def floats_to_255(stack):
    for i in range(len(stack)): #scaling
        maxv = np.max(np.abs(stack[i])) + 1e-8
        stack[i] = (np.clip(0.5 + 0.5*(stack[i]/maxv), 0, 1) * 255).astype(np.uint8)
    """
    for i in range(len(stack)):
        im = stack[i]
        im += np.min(im)
        im /= (np.max(im) + np.min(im))
        im *= 255
        stack[i] = im.astype(np.uint8)
    """
    return stack

def part2_4():
    im1 = 'data/apple.jpeg'
    im2 = 'data/orange.jpeg'

    im1 = skio.imread(im1).astype(np.float32) / 255.0
    im2 = skio.imread(im2).astype(np.float32) / 255.0

    n_levels = 4
    im1_gstack = create_gaussian_stack(im1, n_levels)
    im1_lstack = create_laplacian_stack(im1_gstack)

    im2_gstack = create_gaussian_stack(im2, n_levels)
    im2_lstack = create_laplacian_stack(im2_gstack)

    mask = np.zeros_like(im1).astype(np.float32)
    mask[:, :mask.shape[1]//2] = 1.0## left half is 1s, use on apple (or image where you want left half)
    mask_gstack = create_gaussian_stack(mask, 12)
    mask_gstack = mask_gstack[-n_levels:]

    show_images(mask_gstack)

    hybrid_lstack = []
    for i in range(n_levels-1, -1, -1): 
        res = mask_gstack[i]*im1_lstack[i] + (1-mask_gstack[i])*im2_lstack[i]
        hybrid_lstack.append(res)
    #hybrid_lstack.append(mask_gstack[-1]*im1_lstack[-1] + (1-mask_gstack[-1])*im2_lstack[-1])


    im = np.zeros_like(im1).astype(np.float32)
    for i in range(len(hybrid_lstack)):
        im += hybrid_lstack[i]
    
    
    #for i in range(len(hybrid_lstack)): #scaling
    #    maxv = np.max(np.abs(hybrid_lstack[i])) + 1e-8
    #    hybrid_lstack[i] = (np.clip(0.5 + 0.5*(hybrid_lstack[i]/maxv), 0, 1) * 255).astype(np.uint8)
    
    im1_gstack = floats_to_255(im1_gstack)
    im1_lstack = floats_to_255(im1_lstack)

    im2_gstack = floats_to_255(im2_gstack)
    im2_lstack = floats_to_255(im2_lstack)
    hybrid_lstack = floats_to_255(hybrid_lstack)

    im += np.min(im)
    im /= (np.max(im) + np.min(im))
    im *= 255
    im = im.astype(np.uint8)
    #maxv = np.max(np.abs(im)) + 1e-8
    #im = (np.clip(0.5 + 0.5*(im/maxv), 0, 1) * 255).astype(np.uint8)

    #show_images(im1_gstack + im1_lstack)
    #show_images(im2_gstack + im2_lstack)
    show_images(im1_gstack + im1_lstack)
    show_images(im2_gstack + im2_lstack)
    show_images([im] + hybrid_lstack)
    
def part2_4_irregularmask():
    pass



    

#part1_1()
#part1_2()
#part1_3()

#part2_1()
#part2_22()
#part2_3()
part2_4()
