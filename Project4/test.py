import numpy as np
import os
import skimage.io as skio

def warp(I, output_height, output_width):
    T = np.array([[1, 0, 40],
                [0, 1, 50],
                [0, 0, 1]])

    #T = np.array([[1, 0, 50],
    #          [0, 1, 30],
    #          [0, 0, 1]], dtype=float)
    
    O = np.zeros((output_height, output_width), dtype=I.dtype)

    for y_in in range(I.shape[0]):
        for x_in in range(I.shape[1]):
            out_coord = T @ np.array([x_in, y_in, 1.0])

            # Homogeneous division
            x_out = int(out_coord[0] / out_coord[2])
            y_out = int(out_coord[1] / out_coord[2])

            if (0 <= x_out < output_width) and (0 <= y_out < output_height):
                O[y_out, x_out] = I[y_in, x_in]

    skio.imshow(O)
    skio.show()


im_path = "data/calibration-images/IMG_5199.JPG"
im = skio.imread(im_path)
             
warp(im, im.shape[0], im.shape[1])