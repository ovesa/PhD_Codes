# This program creates quick gifs
# It assumes that the 1st extension of the fits
# files is the primary header that holds no images
# and all the images are in the 2nd extension
# It uses the python library celluloid and imagemagick
# FPS determines the animated gif speed

import matplotlib.pyplot as plt
from celluloid import Camera
from tqdm import tqdm
from astropy.io import fits
import os

# get current working directory
input_path = os.path.abspath(os.getcwd())
print("Your current working directory is", input_path)

# inputs
print('Hello. This program creates quick gifs for a chosen fits file')
filename = input('Enter directory and name of fits file:   ')
outputname = input('Enter output filename of gif. Please enclose the extension of .gif:  ')
chosen_fps = input('Enter desired FPS. Determines the speed of the gif:  ')

# make inputs be strings
filename = str(filename)
outputname = str(outputname)
chosen_fps = int(chosen_fps)

# ensure that selected file exists
# if not, error is raised
if not os.path.isfile(filename):
    raise FileNotFoundError

# grab path of chosen file and filename
path, absolute_filename = os.path.split(filename)

# opening the fits file
# Images are stored in the 2nd fits extension
# First extension contains header information
hdul = fits.open(filename)[1].data
timedim = hdul.shape[0]
thalf = int(timedim//2)

# Body that generates the animation
fig = plt.figure(figsize=[8, 8],frameon=False)
camera = Camera(fig)
for ext in tqdm(range(0,timedim)):
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(hdul[ext,:,:], cmap='gray')
    plt.axis('off')
    plt.margins(0,0)
    camera.snap()
animation = camera.animate()
print("Saving fits file with the name", outputname)
print("Saved to this directory: ", path )
animation.save(str(path) + "/" + outputname, writer = 'imagemagick', fps=chosen_fps)
print("Gif is created.")

