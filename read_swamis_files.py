import multiprocessing

from astropy.utils.data import get_pkg_data_filename
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()
from astropy.io import fits

backend = "threading"
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


dt = 2  # seconds
dx = 150  # km
minlifetime = 5

path_assoc = "/home/oana/Desktop/swamis/swamis/03-assoc/"
path_main = "/home/oana/Desktop/swamis/swamis/00-data/"

assoc_files = sorted(glob.glob(path_assoc + "*.fits"))
main_files = sorted(glob.glob(path_main + "*.fits"))


def velocity(vx):
    vv = np.gradient((vx)) * dx / dt
    vv = vv - np.mean(vv)
    return vv


def add_new_element(i_elem):
    global df

    mask = img == i_elem

    if mask.sum() > 0:
        Bm = B * mask
        Area = mask.sum()
        Flux = Bm.sum() / Area

        X = ((mask * x_1) * Bm).sum() / Bm.sum()
        Y = ((mask * y_1) * Bm).sum() / Bm.sum()

        # print(X,Y)

        # inserimento in dataframe elemento esistente
        chk = i_elem in df.index.values
        if chk == True:
            Area0 = df.loc[i_elem]["Area"]
            ti = df.loc[i_elem]["ti"]
            Area0.append(Area)
            Flux0 = df.loc[i_elem]["Flux"]
            Flux0.append(Flux)
            X0 = df.loc[i_elem]["X"]
            X0.append(X)
            Y0 = df.loc[i_elem]["Y"]
            Y0.append(Y)
            valtf = df.at[i_elem, "tf"]
            df.at[i_elem, "Area"] = Area0
            df.at[i_elem, "Flux"] = Flux0
            df.at[i_elem, "X"] = X0
            df.at[i_elem, "Y"] = Y0
            df.at[i_elem, "Lifetime"] = valtf - ti

        # inserimento in dataframe elemento nuovo
        if chk == False:
            new_row = {
                "label": i_elem,
                "ti": count,
                "tf": count,
                "Area": [Area],
                "Flux": [Flux],
                "X": [X],
                "Y": [Y],
            }
            new = pd.Series(data=new_row, name=i_elem)
            df = df.append(new, ignore_index=False)
    return


# step 0.

count = 0
assoc_image_file = get_pkg_data_filename(assoc_files[35])
assoc_img = fits.getdata(assoc_image_file)

main_image_file = get_pkg_data_filename(main_files[35])
main_imag = fits.getdata(main_image_file)

count = count + 1

img_size = np.shape(assoc_img)
im_x, im_y = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))

max_img_size = np.int64(assoc_img.max())
min_img_size = np.int64(assoc_img.min())


df = pd.DataFrame(
    columns=[
        "label",
        "ti",
        "tf",
        "Lifetime",
        "Area",
        "Flux",
        "X",
        "Y",
        "Intensity",
        "Vel_x",
        "Vel_y",
        "stdx",
        "stdy",
    ],
    index=list(np.int64(np.arange(min_img_size, max_img_size))),
)

count = 0
for fname in assoc_files:
    assoc_image_file = get_pkg_data_filename(fname)
    assoc_img = fits.getdata(assoc_image_file)

    main_image_file = get_pkg_data_filename(main_files[0])
    main_imag = fits.getdata(main_image_file)
    print(count)

    for i_elem in range(min_img_size, max_img_size):
        indx = np.where(assoc_img == i_elem)
        value_x = indx[0][:]
        value_y = indx[1][:]

        if np.size(value_x) > 0:
            mask = np.zeros([img_size[0], img_size[1]])
            mask[indx] = 1
            Bm = main_imag * mask
            Area = mask.sum()
            wh = np.where(np.abs(Bm) > 0)
            Flux = Bm[indx].mean()
            valsY = (indx[0][:] * Bm[indx[0][:], indx[1][:]]).sum() / np.sum(Bm)
            valsX = (indx[1][:] * Bm[indx[0][:], indx[1][:]]).sum() / np.sum(Bm)

            # inserimento in dataframe elemento esistente
            chk = i_elem in df.index.values
            if chk == True:
                df.at[i_elem, "ti"] = count
                ti = df.loc[i_elem]["ti"]
                df.at[i_elem, "label"] = i_elem
                df.at[i_elem, "Area"] = Area
                df.at[i_elem, "Flux"] = Flux
                df.at[i_elem, "X"] = valsX
                df.at[i_elem, "Y"] = valsY
                df.at[i_elem, "tf"] = count
                df.at[i_elem, "Lifetime"] = count - ti

                # df.at[i_elem, "label"] = i_elem
                # ti = df.loc[i_elem]["ti"]
                # df.at[i_elem, "ti"] = count
                # df.at[i_elem, "tf"] = count
                # df.at[i_elem, "Area"] = Area
                # df.at[i_elem, "Flux"] = Flux
                # df.at[i_elem, "X"] = valsX
                # df.at[i_elem, "Y"] = valsY
                # df.at[i_elem, "Lifetime"] = 1

            # inserimento in dataframe elemento nuovo
            if chk == False:
                new_row = {
                    "label": i_elem,
                    "ti": count,
                    "tf": count,
                    "Area": Area,
                    "Flux": Flux,
                    "X": valsX,
                    "Y": valsY,
                }
                new = pd.Series(data=new_row, name=i_elem)
                df = df.append(new, ignore_index=False)

    count = count + 1


df = df.dropna(axis=0, how="all")

# for i in range(1,np.size(assoc_files)):
#         print(count)
#         count=count+1
#         image_file = get_pkg_data_filename(files[i])
#         img = fits.getdata(image_file, ext=0)
#         image_fileB = get_pkg_data_filename(filesB[i])
#         B = fits.getdata(image_fileB, ext=0)
#         maxv=np.int64(img.max())
#         minv=np.int64(img.min())


#         Parallel(n_jobs=num_cores, backend=backend)(delayed(add_new_element)(i_elem) for i_elem in np.int64(np.unique(img)))

df = df[df["Lifetime"] < 1]
df.loc[:, "Vel_x"] = df.loc[:, "X"]
df.loc[:, "Vel_y"] = df.loc[:, "Y"]


df["Vel_x"] = velocity(df["Vel_x"].to_numpy())
df["Vel_y"] = velocity(df["Vel_y"].to_numpy())


df["stdx"] = df["Vel_x"].apply(np.std)
df["stdy"] = df["Vel_y"].apply(np.std)
# df.to_pickle(swamis_path + "/dataframe.pkl")

plt.figure(figsize=[10, 7])
plt.hist(
    df.loc[:, "stdx"],
    bins="auto",
    density=True,
    label="$v_{x}$",
    alpha=0.5,
    color="red",
    range=(1, 6),
)
plt.hist(
    df.loc[:, "stdy"],
    bins="auto",
    density=True,
    label="$v_{y}$",
    alpha=0.5,
    range=(1, 6),
)
plt.xlabel("$\sigma_{v}$ (km/s)")
plt.ylabel("PDF")
# plt.xlim([0, 10])
plt.legend()
plt.show()


plt.figure(figsize=[10, 7])
plt.hist(
    df.loc[:, "Vel_x"],
    bins="auto",
    label="$v_{x}$",
    alpha=0.5,
)

plt.hist(df.loc[:, "Vel_y"], bins="auto", label="$v_{y}$", alpha=0.5)

plt.xlabel("$\sigma_{v}$ (km/s)")
plt.ylabel("PDF")
# plt.xlim([0, 10])
plt.legend()
plt.tight_layout()
plt.show()


### ---------------------------------


folders = natsorted(glob.glob("/home/oana/Desktop/swamis/swamis/"))

for folder in folders:
    swamis_path = folder
    print(swamis_path)

    def velocity(vx):
        vv = np.gradient((vx)) * dx / dt
        vv = vv - np.mean(vv)
        return vv

    def add_new_element(i_elem):
        global df

        mask = img == i_elem

        if mask.sum() > 0:
            Bm = B * mask
            Area = mask.sum()
            Flux = Bm.sum() / Area

            X = ((mask * x_1) * Bm).sum() / Bm.sum()
            Y = ((mask * y_1) * Bm).sum() / Bm.sum()

            # print(X, Y)

            # inserimento in dataframe elemento esistente
            chk = i_elem in df.index.values
            # print(chk)

            if chk == True:
                Area0 = df.loc[i_elem, "Area"]
                ti = df.loc[i_elem, "ti"]
                Area0.append(Area)
                Flux0 = df.loc[i_elem, "Flux"]
                Flux0.append(Flux)
                X0 = df.loc[i_elem, "X"]
                X0.append(X)
                Y0 = df.loc[i_elem, "Y"]
                Y0.append(Y)

                df.at[i_elem, "tf"] = i

                df.at[i_elem, "Area"] = Area0
                df.at[i_elem, "Flux"] = Flux0
                df.at[i_elem, "X"] = X0
                df.at[i_elem, "Y"] = Y0
                df.at[i_elem, "Lifetime"] = i - ti
                # print(df.at[i_elem, "Lifetime"], valtf, ti)

                # Area0 = df.loc[i_elem]["Area"]
                # ti = df.loc[i_elem]["ti"]
                # Area0.append(Area)
                # Flux0 = df.loc[i_elem]["Flux"]
                # Flux0.append(Flux)
                # X0 = df.loc[i_elem]["X"]
                # X0.append(X)
                # Y0 = df.loc[i_elem]["Y"]
                # Y0.append(Y)
                # df.at[i_elem, "tf"] = i
                # df.at[i_elem, "Area"] = Area0
                # df.at[i_elem, "Flux"] = Flux0
                # df.at[i_elem, "X"] = X0
                # df.at[i_elem, "Y"] = Y0
                # df.at[i_elem, "Lifetime"] = i - ti

            # inserimento in dataframe elemento nuovo
            if chk == False:
                new_row = {
                    "label": i_elem,
                    "ti": count,
                    "tf": count,
                    "Area": [Area],
                    "Flux": [Flux],
                    "X": [X],
                    "Y": [Y],
                }
                new = pd.Series(data=new_row, name=i_elem)
                df = df.append(new, ignore_index=False)
                # new = pd.DataFrame(new).T
                # df = pd.concat([df, new.to_frame().T], ignore_index=True)
        return

    files = natsorted(glob.glob(swamis_path + "/03-assoc/*.fits"))
    filesB = natsorted(glob.glob(swamis_path + "/00-data/*.fits"))

    count = 0
    image_file = get_pkg_data_filename(files[35])
    img = fits.getdata(image_file)  # [64:73, 53:70]
    image_fileB = get_pkg_data_filename(filesB[35])
    B = fits.getdata(image_fileB)  # [64:73, 53:70]
    count = count + 1
    size = np.shape(img)
    x_1, y_1 = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    maxv = np.int64(img.max())
    minv = np.int64(img.min())

    df = pd.DataFrame(
        columns=[
            "label",
            "ti",
            "tf",
            "Lifetime",
            "Area",
            "Flux",
            "X",
            "Y",
            "Intensity",
            "Vel_x",
            "Vel_y",
            "stdx",
            "stdy",
        ],
        index=list(np.int64(np.arange(minv, maxv))),
    )

    for i_elem in range(minv, maxv):
        p = np.where(img == i_elem)
        x = p[0][:]
        y = p[1][:]

        if np.size(x) > 0:
            mask = np.zeros([size[0], size[1]])
            mask[p] = 1
            Bm = B * mask
            Area = mask.sum()
            wh = np.where(np.abs(Bm) > 0)
            Flux = Bm[p].mean()
            Y = (p[0][:] * Bm[p[0][:], p[1][:]]).sum() / np.sum(Bm)
            X = (p[1][:] * Bm[p[0][:], p[1][:]]).sum() / np.sum(Bm)
            plt.scatter(X + 1, Y + 1)

            print(i_elem, X, Y, Area, Flux)

            # inserimento in dataframe elemento esistente
            chk = i_elem in df.index.values
            if chk == True:
                df.at[i_elem, "label"] = i_elem
                df.at[i_elem, "ti"] = count
                df.at[i_elem, "tf"] = count
                df.at[i_elem, "Area"] = [Area]
                df.at[i_elem, "Flux"] = [Flux]
                df.at[i_elem, "X"] = [X]
                df.at[i_elem, "Y"] = [Y]
                df.at[i_elem, "Lifetime"] = 1

                # df.at[i_elem, "label"] = i_elem
                # df.at[i_elem, "ti"] = count
                # df.at[i_elem, "tf"] = count
                # df.at[i_elem, "Area"] = [Area]
                # df.at[i_elem, "Flux"] = [Flux]
                # df.at[i_elem, "X"] = [X]
                # df.at[i_elem, "Y"] = [Y]
                # df.at[i_elem, "Lifetime"] = 1

            # inserimento in dataframe elemento nuovo
            if chk == False:
                new_row = {
                    "label": i_elem,
                    "ti": count,
                    "tf": count,
                    "Area": [Area],
                    "Flux": [Flux],
                    "X": [X],
                    "Y": [Y],
                }
                new = pd.Series(data=new_row, name=i_elem)
                df = df.append(new, ignore_index=False)
                # new = pd.DataFrame(new).T
                # df = pd.concat([df, new.to_frame().T], ignore_index=True)

    df = df.dropna(axis=0, how="all")
    count = 1

    for i in range(1, len(files[36:163])):
        print(count)
        image_file = get_pkg_data_filename(files[i])
        img = fits.getdata(image_file)  # [64:73, 53:70]
        image_fileB = get_pkg_data_filename(filesB[i])
        B = fits.getdata(image_fileB)  # [64:73, 53:70]
        maxv = np.int64(img.max())
        minv = np.int64(img.min())

        try:
            # block raising an exception
            Parallel(n_jobs=num_cores, backend=backend)(
                delayed(add_new_element)(i_elem) for i_elem in np.int64(np.unique(img))
            )
        except:
            pass  # doing nothing on exception
        count = count + 1

    df.loc[:, "Vel_x"] = df.loc[:, "X"]
    df.loc[:, "Vel_y"] = df.loc[:, "Y"]
    # df["Vel_x"] = velocity(df["Vel_x"])
    # df["Vel_y"] = velocity(df["Vel_y"])
    df["stdy"] = df["Vel_y"].apply(np.std)
    df["stdx"] = df["Vel_x"].apply(np.std)
    df = df[df["stdx"] > 0]
    # df = df[df["Lifetime"] > 5]


df.loc[72, :]


#############

from matplotlib.pyplot import cm

color = cm.rainbow(np.linspace(0, 1, 2557))
image_fileB = get_pkg_data_filename(filesB[71])
B = fits.getdata(image_fileB)

plt.figure()
plt.pcolormesh(B)
plt.scatter(df["X"][0], df["Y"][0], color="red")
plt.scatter(df["X"][72], df["Y"][72], color="red")

plt.show()


chosen_frame = df.loc[df["tf"] == 1]

image_fileB = get_pkg_data_filename(filesB[72])
B = fits.getdata(image_fileB)

color = cm.rainbow(np.linspace(0, 1, df["X"].index.shape[0]))
plt.figure()
plt.pcolormesh(img)
for i, c in enumerate(color):
    # plt.scatter(chosen_frame["X"][i], chosen_frame["Y"][i], c=c)
    try:
        plt.scatter(df["X"][i][0], df["Y"][i][0], c=c)
    except:
        pass
plt.show()


plt.scatter(chosen_frame["X"][4892], chosen_frame["Y"][4892])


image_file = get_pkg_data_filename(files[72])
img = fits.getdata(image_file)[64:73, 53:70]
plt.pcolormesh(img)


7724


import skimage
import matplotlib.patches as mpatches

img_example = img

# Label elements on the picture
white = 0.0
label_image, number_of_labels = skimage.measure.label(
    img_example, background=white, return_num=True
)
print("Found %d features" % (number_of_labels))
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
ax.pcolormesh(img_example)
for region in skimage.measure.regionprops(label_image, intensity_image=img_example):
    # Everywhere, skip small and large areas
    if region.area < 3 or region.area > 30:
        continue
    # Only black areas
    if region.mean_intensity < 0.5:
        continue

    # Draw rectangle which survived to the criterions
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle(
        (minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor="red", linewidth=1
    )

    ax.add_patch(rect)
