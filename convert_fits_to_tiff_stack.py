################### Import Necessary Modules ###################

import glob as glob
import os

import numpy as np
import tqdm
from astropy.io import fits
from natsort import natsorted
from tifffile import imsave

################### Directory Information ###################


path_to_assoc_files = "/home/oana/Desktop/swamis/swamis/01-mask/"

if not os.path.isdir(path_to_assoc_files):
    print("Directory doesn't exist")


assoc_files = natsorted(glob.glob(path_to_assoc_files + "*.fits"))
main_files = natsorted(
    glob.glob("/home/oana/Desktop/swamis/swamis/00-data/" + "*.fits")
)

temp_img = fits.getdata(assoc_files[0])

im_array = np.zeros([len(assoc_files), temp_img.shape[0], temp_img.shape[1]])
im_array_main = np.zeros([len(assoc_files), temp_img.shape[0], temp_img.shape[1]])

temp_img = 0


for cnt, fname in tqdm.tqdm(enumerate(assoc_files), total=len(assoc_files)):
    assoc_img = fits.getdata(fname)
    im_array[cnt, :, :] = assoc_img

    main_img = fits.getdata(main_files[cnt])
    im_array_main[cnt, :, :] = main_img


imsave("/home/oana/Desktop/swamis/swamis/11272018.GBAND.Event26.main.tif", im_array_main)


image_hdu = fits.ImageHDU(im_array * im_array_main)
image_hdu.writeto(
    "/home/oana/Desktop/swamis/swamis/11272018.GBAND.Event26.both.fits", overwrite=True
)
