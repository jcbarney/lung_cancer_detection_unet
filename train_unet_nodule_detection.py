import SimpleITK as sitk
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import roberts
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from sklearn.preprocessing import minmax_scale
from skimage.transform import resize
import numpy as np
import pandas as pd
import os
import tensorflow as tf

annotations = pd.read_csv("annotations.csv")

##################################################################################################################################
###### function given by Luna16 challenge administrators to assist in converting coordinates from the  candidates data set #######
##################################################################################################################################


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


##############################################################################################################################
######                                       end code from Luna16                                                     #######
############################################################################################################################


def pixel_diameter(spacing, diameter):
    px_diam = np.absolute(diameter) / spacing
    px_diam = [int(np.floor(i)) for i in px_diam]
    return px_diam


def preprocess(image):
    # get image and pull origin and spacing for translating nodule coordinates from file
    ct_scan = sitk.ReadImage(image, sitk.sitkFloat32)
    numpyOrigin = np.array(list(reversed(ct_scan.GetOrigin())))
    numpySpacing = np.array(list(reversed(ct_scan.GetSpacing())))
    ct_scan = sitk.GetArrayFromImage(ct_scan)

    # blur image
    ct_scan_thresh_mask = gaussian_filter(ct_scan, sigma=(5, 5, 0))

    # already in hounsfield units
    # range of tissue and bronchioles to keep: study doesn't give exact number (close to -1000 or above -320 masked)
    min_hu = -900
    max_hu = -320
    ct_scan_thresh_mask = np.clip(ct_scan_thresh_mask, min_hu, max_hu)
    ct_scan_thresh_mask = ct_scan_thresh_mask < -400

    # remove peices of scan that aren't part of lung tissues
    for i in range(ct_scan_thresh_mask.shape[0]):
        ct_scan_thresh_mask[i] = clear_border(ct_scan_thresh_mask[i])
        edges = roberts(ct_scan_thresh_mask[i])
        ct_scan_thresh_mask[i] = ndi.binary_fill_holes(edges)

    # assign masked regions to original ct
    ct_scan[ct_scan_thresh_mask == False] = -3024
    ct_scan[ct_scan < -400] = -3024

    # set to between 0 and 1 for model
    for i in range(ct_scan.shape[0]):
        # Need to normal 0-1 and zero_center
        ct_scan[i] = minmax_scale(ct_scan[i], feature_range=(0, 1))

    # 3d spline interpolation to 0.5 in xyz directions
    ct_scan = resize(ct_scan, (int(ct_scan.shape[0] / 2), 256, 256), mode="reflect")

    # zero mean to make data symmetric
    meanCenter = lambda x: x - x.mean()
    ct_scan = meanCenter(ct_scan)

    return ct_scan, numpyOrigin, numpySpacing


def output_data(ct_scan, numpyOrigin, numpySpacing, file, annotations):
    zeros = np.zeros(ct_scan.shape)
    nodules = annotations[annotations["seriesuid"] == file]
    nodules.reset_index(drop=True, inplace=True)
    for i in range(len(nodules)):
        print("has nodule")
        coordinates = (nodules["coordZ"][i], nodules["coordY"][i], nodules["coordX"][i])
        nodule_loc = worldToVoxelCoord(coordinates, numpyOrigin, numpySpacing)
        nodule_loc = [int(np.floor(j)) for j in nodule_loc]
        px_diam = pixel_diameter(numpySpacing, nodules["diameter_mm"][i])

        # use a box instead of circle as a circle would be slower
        zeros[
            int(int(nodule_loc[0] / 2) - max(round(px_diam[0] / 4, 0), 1)) : int(
                int(nodule_loc[0] / 2) + max(round(px_diam[0] / 4, 0), 1) + 1
            ),
            int(nodule_loc[1] / 2 - max(round(px_diam[1] / 4, 0), 1)) : int(
                nodule_loc[1] / 2 + max(round(px_diam[1] / 4, 0), 1) + 1
            ),
            int(nodule_loc[2] / 2 - max(round(px_diam[2] / 4, 0), 1)) : int(
                nodule_loc[2] / 2 + max(round(px_diam[2] / 4, 0), 1) + 1
            ),
        ] = 1
    return zeros


def get_unet_input():
    folders = [
        "subset0/subset0/",
        "subset1/subset1/",
        "subset2/subset2/",
        "subset3/subset3/",
        "subset4/subset4/",
        "subset5/subset5/",
        "subset6/subset6/",
        "subset7/subset7/",
        "subset8/subset8/",
        "subset9/subset9/",
    ]
    inputs = []
    outputs = []
    files = os.listdir(folder)
    files = [i[:-4] for i in files if i[-4:] == ".mhd"]
    print("files: ", len(files))
    for file in files:
        image = str(folder) + str(file) + ".mhd"
        ct_scan, numpyOrigin, numpySpacing = preprocess(image)
        zeros = output_data(ct_scan, numpyOrigin, numpySpacing, file, annotations)
        inputs.extend(ct_scan)
        outputs.extend(zeros)
    return inputs, outputs


def train_unet(X_train, y_train):
    from keras.utils import normalize
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        Input,
        UpSampling2D,
        Dropout,
        concatenate,
    )
    from tensorflow.keras.models import Model

    # convolutions
    inputs = Input((256, 256, 1))
    s1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    s1 = Dropout(0.2)(s1)
    s1 = Conv2D(64, 3, activation="relu", padding="same")(s1)
    p1 = MaxPooling2D(pool_size=(2, 2))(s1)

    s2 = Conv2D(128, 3, activation="relu", padding="same")(p1)
    s2 = Dropout(0.2)(s2)
    s2 = Conv2D(128, 3, activation="relu", padding="same")(s2)
    p2 = MaxPooling2D(pool_size=(2, 2))(s2)

    s3 = Conv2D(256, 3, activation="relu", padding="same")(p2)
    s3 = Dropout(0.2)(s3)
    s3 = Conv2D(256, 3, activation="relu", padding="same")(s3)
    p3 = MaxPooling2D(pool_size=(2, 2))(s3)

    s4 = Conv2D(512, 3, activation="relu", padding="same")(p3)
    s4 = Dropout(0.2)(s4)
    s4 = Conv2D(512, 3, activation="relu", padding="same")(s4)
    p4 = MaxPooling2D(pool_size=(2, 2))(s4)

    # bridge across U
    b1 = Conv2D(1024, 3, activation="relu", padding="same")(p4)
    b1 = Dropout(0.2)(b1)
    b1 = Conv2D(1024, 3, activation="relu", padding="same")(b1)

    d1 = Conv2D(512, 2, activation="relu", padding="same")(
        UpSampling2D(size=(2, 2), interpolation="bilinear")(b1)
    )
    d1 = concatenate([s4, d1], axis=3)
    d1 = Conv2D(512, 3, activation="relu", padding="same")(d1)
    d1 = Dropout(0.2)(d1)
    d1 = Conv2D(512, 3, activation="relu", padding="same")(d1)

    d2 = Conv2D(256, 2, activation="relu", padding="same")(
        UpSampling2D(size=(2, 2), interpolation="bilinear")(d1)
    )
    d2 = concatenate([s3, d2], axis=3)
    d2 = Conv2D(256, 3, activation="relu", padding="same")(d2)
    d2 = Dropout(0.2)(d2)
    d2 = Conv2D(256, 3, activation="relu", padding="same")(d2)

    d3 = Conv2D(128, 2, activation="relu", padding="same")(
        UpSampling2D(size=(2, 2), interpolation="bilinear")(d2)
    )
    d3 = concatenate([s2, d3], axis=3)
    d3 = Conv2D(128, 3, activation="relu", padding="same")(d3)
    d3 = Dropout(0.2)(d3)
    d3 = Conv2D(128, 3, activation="relu", padding="same")(d3)

    d4 = Conv2D(64, 2, activation="relu", padding="same")(
        UpSampling2D(size=(2, 2), interpolation="bilinear")(d3)
    )
    d4 = concatenate([s1, d4], axis=3)
    d4 = Conv2D(64, 3, activation="relu", padding="same")(d4)
    d4 = Dropout(0.2)(d4)
    d4 = Conv2D(64, 3, activation="relu", padding="same")(d4)
    d4 = Conv2D(2, 3, activation="relu", padding="same")(d4)

    outputs = Conv2D(1, 1, activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="unet")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # get top 8 suspicious non-overlapping regions for cancer/non-cancer

    from keras.callbacks import ModelCheckpoint
    from sklearn.metrics import accuracy_score, recall_score, precision_score

    # fit model and show results of each epoch
    checkpoint_name = "Weights-{epoch:03d}--{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(
        checkpoint_name, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"
    )
    callbacks_list = [checkpoint]
    #train model
    model.fit(X_train, y_train, epochs=10, batch_size=10, callbacks=callbacks_list)
    return model

#get training data
inputs, outputs = get_unet_input()
np_input = np.array(inputs)
np_input = np_input.reshape(len(np_input), 256, 256, 1)
np_output = np.array(outputs)
np_output = np_output.reshape(len(np_output), 256, 256, 1)

#save training data
import pickle
filename = "inputs_0502".sav"
pickle.dump(np_input, open(filename, 'wb'))
filename = "outputs_0502".sav"
pickle.dump(np_output, open(filename, 'wb'))

#train unet
train_unet(np_input, np_output)
filename = "model_0502TF_2".sav"
pickle.dump(model, open(filename, 'wb'))

