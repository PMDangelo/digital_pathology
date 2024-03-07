#set of basic function, import and other things
#v0.5
#last update 16/02/24
#Pietro Marco D'Angelo


from __future__ import print_function, unicode_literals, absolute_import, division

import sys
import subprocess

subprocess.call([sys.executable, '-m','pip','install','stardist','imagecodecs','csbdeep','openslide'])

import numpy as np
import pandas as pd
import os
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
#import openslide
import imagecodecs
import tensorflow as tf
import torch

from skimage import exposure, io
from scipy import ndimage
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from collections import Counter
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist.utils import mask_to_categorical
from stardist.plot import render_label
from PIL import Image
#from openslide import open_slide
#from openslide.deepzoom import DeepZoomGenerator
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter


np.random.seed(42)
lbl_cmap = random_label_cmap()
lbl_cmap_classes = matplotlib.cm.tab20


#last update 28/08/23
def load_svs_to_np(path,level=0):
    '''
    Loads a WSI.svs from pat, return a numpy.array, uint8, value betwen 0 to 255.
    Requires path of the .wsi file and the level (so you can load different resolutions)


    created by Pietro Marco D'Angelo
    '''

    if path[-1]!='/':
        path=path+'/'
        img = open_slide(path)
        dims = img.level_dimensions
        img = img.get_thumbnail(size=dims[level])
        img = img.convert('RGB')
        img = np.array(img)

        return(img)

#last update 28/08/23
#Pietro Marco D'Angelo
def load_tif_from_path(path):
    '''
    loads all tif files of a folder into a numpy array
    '''
    if path[-1]!='/':
        path=path+'/'

    l=[filename for filename in os.listdir(path)]
    l.sort()
    list=[io.imread(path+x) for x in l]

    return(np.array(list))

#last update 28/08/23
#Pietro Marco D'Angelo
def load_tif_and_augmentation(path):
    '''
    loads all tif files of a folder into a numpy array, for all file generate 3 rotation and 4 flip
    '''
    if path[-1]!='/':
        path=path+'/'

    l=[filename for filename in os.listdir(path)]
    l.sort()
    list=[]
    for x in l:
        t=io.imread(path+x)
        f=cv2.flip(t, 0)
        list.append(t)
        list.append(f)
        list.append(cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE))
        list.append(cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE))
        list.append(cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE))
        list.append(cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE))
        list.append(cv2.rotate(t, cv2.ROTATE_180))
        list.append(cv2.rotate(f, cv2.ROTATE_180))

    return(np.array(list))

#normalization functions

#last update 16/02/2024
#Pietro Marco D'Angelo
def standar_stardist_norm(images, percentile_min=1,percentile_max=99):
    norm_images = [normalize(x, percentile_min, percentile_max , axis=(0,1) ) for x in images]
    norm_images=np.array(norm_images)
  
    return(norm_images)

#last update 28/08/23
#Pietro Marco D'Angelo
def min_max_scaling(image):
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = (image - min_val) / (max_val - min_val)

    return scaled_image

#last update 28/08/23
#Pietro Marco D'Angelo
def z_score_normalization(image, zero_one=True):
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    if zero_one:
        normalized_image=min_max_scaling(normalized_image)

    return normalized_image

#last update 28/08/23
#Pietro Marco D'Angelo
def pca_whitening(image):
    _, s, Vt = np.linalg.svd(image, full_matrices=False)
    epsilon = 1e-10  # Small constant to avoid division by zero
    whitened_image = np.dot(np.diag(1.0 / np.sqrt(s + epsilon)), Vt)

    return whitened_image

#last update 28/08/23
#Pietro Marco D'Angelo
def contrast_stretching(image, zero_one=True):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))
    if zero_one:
        stretched_image=min_max_scaling(stretched_image)

    return stretched_image

#last update 28/08/23
#Pietro Marco D'Angelo
def histogram_equalization(image, zero_one=True):
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    equalized_image = equalized_image.reshape(image.shape)
    if zero_one:
        equalized_image=min_max_scaling(equalized_image)

    return equalized_image

#last update 28/08/23
#Pietro Marco D'Angelo
def clahe(image, clip_limit=0.03, grid_size=(8, 8), zero_one=True):
    equalized_image = exposure.equalize_adapthist(image, clip_limit=clip_limit, kernel_size=grid_size)
    if zero_one:
        equalized_image=min_max_scaling(equalized_image)

    return equalized_image

#last update 28/08/23
#Pietro Marco D'Angelo
def power_law_transform(image, gamma=1.0, zero_one=True):
    transformed_image = np.power(image / 255.0, gamma)
    if zero_one:
        transformed_image=min_max_scaling(transformed_image)

    return transformed_image

#last update 28/08/23
#Pietro Marco D'Angelo
def normalize_to_minus_one_one(image, zero_one=True):
    normalized_image = 2 * ((image - np.min(image)) / (np.max(image) - np.min(image))) - 1
    if zero_one:
        normalized_image=min_max_scaling(normalized_image)

    return normalized_image

#last update 28/08/23
#Pietro Marco D'Angelo
def pre_elaborazione(img, level=0):
    dims = img.level_dimensions
    img = img.get_thumbnail(size=dims[level])
    img = img.convert('RGB')
    img = np.array(img)

    return(img)

#last update 28/08/23
#Pietro Marco D'Angelo
def my_normalize_and_HE(img, Io = 240, alpha = 1, beta = 0.15, zero_one=True):
    '''
    custom macenko method for normalization, given a svs or a np.array
    return 3 matrix: img normalized, H an E (...)
    if zero_one=True returm matrxi whit value betwen 0 and 1
    else betwen 0 and 255
    '''

    #da sistemare
    #if type(img) == openslide.OpenSlide:
    #    img = pre_elaborazione(img)

    HERef = np.array([[0.5626,0.2159],
                 [0.7201,0.8012],
                 [0.4062,0.5581]])
    maxCRef = np.array([1.9705,1.0308])
    img = np.array(img)
    h, w, c = img.shape
    img = img.reshape((-1,3))
    
    OD = -np.log10((img.astype(np.float64)+1)/Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:,1:3])
    phi = np.arctan2(That[:,1],That[:,0])
    phi = np.arctan2(That[:,1],That[:,0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T

    Y = np.reshape(OD, (-1, 3)).T

    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if zero_one:
        Inorm=Inorm/254
        H=H/254
        E=E/254

    return(Inorm,H,E)

#other functions

#last update 28/08/23
#Pietro Marco D'Angelo
def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)

    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)

    return img, mask

#last update 28/08/23
#Pietro Marco D'Angelo
def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)

    return img

#last update 28/08/23
#Pietro Marco D'Angelo
def augmenter(x, y):
    '''
    Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    '''
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)

    return x, y

#last update 28/08/23
#Pietro Marco D'Angelo
def class_from_res(res):
    cls_dict = dict((i+1,c) for i,c in enumerate(res['class_id']))

    return cls_dict

#last update 28/08/23
#Pietro Marco D'Angelo
def cls_dict_from_mask_MC_old(mask):
    '''
    create a dictionary for the multiclass model from the mask
    It is very important to know the structure of the masks! This can create problems for the model!!!
    for the singleclass model it is not necessary
    '''
    cls_dict={}
    unique = np.unique(mask)
    for x in unique:
        if x:
            cls_dict[x]=x

    return cls_dict

#last update 28/08/23
#Pietro Marco D'Angelo
def plot_img_label_MC(img, lbl, cls_dict, n_classes=2, img_title="image", lbl_title="label", cls_title="classes", **kwargs):
    c = mask_to_categorical(lbl, n_classes=n_classes, classes=cls_dict)
    res = np.zeros(lbl.shape, np.uint16)
    for i in range(1,c.shape[-1]):
        m = c[...,i]>0
        res[m] = i
    class_img = lbl_cmap_classes(res)
    class_img[...,:3][res==0] = 0
    class_img[...,-1][res==0] = 1

    fig, (ai,al,ac) = plt.subplots(1,3, figsize=(17,7), gridspec_kw=dict(width_ratios=(1.,1,1)))
    im = ai.imshow(img, cmap='gray')
    #fig.colorbar(im, ax = ai)
    ai.set_title(img_title)
    al.imshow(render_label(lbl, .8*normalize(img, clip=True), normalize_img=False, alpha_boundary=.8,cmap=lbl_cmap))
    al.set_title(lbl_title)
    ac.imshow(class_img)
    ac.imshow(render_label(res, .8*normalize(img, clip=True), normalize_img=False, alpha_boundary=.8, cmap=lbl_cmap_classes))
    ac.set_title(cls_title)
    plt.tight_layout()
    for a in ai,al,ac:
        a.axis("off")

    return ai,al,ac

#last update 28/08/23
#Pietro Marco D'Angelo
def plot_img_label_SC(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray')
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


#last update 28/08/23
#Pietro Marco D'Angelo
def split_train_val_MC(tiles,masks,Percentage_validation=0.1,cls_dict=None):
    '''
    split masks nad tiles in train and validation set whit the clasess, defolt percentage is 0.1.
    return X_val, Y_val, C_val, X_trn, Y_trn, C_trn
    '''
    rng = np.random.RandomState(42)

    try:
        cls_dict
    except NameError:
        print("cls_dict not defined, I got this!")
        cls_dict=cls_dict_from_mask_mask_MC(masks)

    C=[cls_dict for x in range(len(tiles))]
    ind = rng.permutation(len(tiles))
    n_val = max(1, int(round(Percentage_validation * len(ind))))

    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val, C_val = [tiles[i] for i in ind_val]  , [masks[i] for i in ind_val] , [C[i] for i in ind_val]
    X_trn, Y_trn, C_trn = [tiles[i] for i in ind_train], [masks[i] for i in ind_train],  [C[i] for i in ind_train]

    return(X_val, Y_val, C_val, X_trn, Y_trn, C_trn)

#last update 04/09/23
#Pietro Marco D'Angelo
def create_conf(tiles, n_classes=2, grid = (2,2), n_rays = 32, print_summary=False):
    n_rays = n_rays
    n_channel_in = 1 if tiles.ndim == 2 else tiles[0].shape[-1]
    use_gpu = True and gputools_available()
    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel_in,
        n_classes    = n_classes
        )

    if print_summary:
        print(conf)

    return conf

#last update 28/08/23
#Pietro Marco D'Angelo
def autoname():
    return str(datetime.datetime.now().strftime("%Y_%m_%d_%H:%M"))

#last update 28/08/23
#Pietro Marco D'Angelo
def create_model(conf, name = autoname(), basedir = 'Models'):
    model = StarDist2D(conf, name=name, basedir=basedir)

#last update 04/09/23
#Pietro Marco D'Angelo
def create_model_auto_grid(tiles, masks, name = autoname(), basedir = 'Models', n_classes=2, n_rays = 32, print_summary=False):
    n_rays = n_rays
    n_channel_in = 1 if tiles.ndim == 2 else tiles[0].shape[-1]
    use_gpu = True and gputools_available()
    n_g=2
    conf = Config2D (
        n_rays       = n_rays,
        grid         = (n_g,n_g),
        use_gpu      = use_gpu,
        n_channel_in = n_channel_in,
        n_classes    = n_classes
        )

    model = StarDist2D(conf, name=name, basedir=basedir)
    median_size = calculate_extents(list(masks), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))

    while any(median_size > fov):
        n_g*=2
        conf = create_conf(tiles, grid = (n_g,n_g))
        model = StarDist2D(conf, name=name, basedir=basedir)
        median_size = calculate_extents(list(masks), np.median)
        fov = np.array(model._axes_tile_overlap('YX'))

    print(f"grid: ({n_g},{n_g})")
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")

    return(model)

#last update 28/08/23
#Pietro Marco D'Angelo
def train_model(model, X_trn, Y_trn,C_trn,X_val,Y_val,C_val,epochs=50,steps_per_epoch=100):
    model.train(X_trn, Y_trn, classes=C_trn,
                validation_data=(X_val,Y_val,C_val),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs)

    model.optimize_thresholds(X_val, Y_val)

    return(model)

#last update 28/08/23
#Pietro Marco D'Angelo
def plot_metrics(Y_val, Y_val_pred):
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()

#last update 12/01/24
#Pietro Marco D'Angelo
def separa_2_classi(masks):
    masks_classe_1 = (masks == 1)
    masks_classe_2 = (masks == 2)
    masks1 = masks * masks_classe_1
    masks2 = masks * masks_classe_2

    return(masks1, masks2)

#last update 06/02/24
#Pietro Marco D'Angelo
def separa_n_classi(masks):
    arr_masks=[]

    for x in range (masks.max()):
        temp=[]
        masks_classe_temp = (masks == x+1)
        masks_tamp = masks * masks_classe_temp
        arr_masks.append(masks_tamp)

    return(arr_masks)

#last update 14/02/24
#Pietro Marco D'Angelo
def assegna_numeri_univoci(masks):
    oggetti_etichettati = np.zeros_like(masks, dtype=np.int32)

    for i in range(masks.shape[0]):
        # Etichettatura degli oggetti connessi all'interno di ciascuna immagine
        #la funzione label etichetta gli oggetti connessi all'interno di ciascuna immagine
        #assegnando loro un numero univoco
        oggetti, numero_oggetti = ndimage.label(masks[i])

        # Assegnamento di un numero univoco a ciascun oggetto
        oggetti_etichettati[i] = oggetti

    return oggetti_etichettati


#last update 15/01/24
#Pietro Marco D'Angelo
def assegna_numeri_univoci_n_classi(masks_separate):
    new_arr=[]
    for x in range(len(masks_separate)):
        new_arr.append(assegna_numeri_univoci(masks_separate[x]))

    return(new_arr)


#last update 12/02/24
#Pietro Marco D'Angelo
def merge_masks_separate_items(masks_1,masks_2):
    #unique1  = np.unique(masks_1, return_counts=False)
    #unique2  = np.unique(masks_2, return_counts=False)
    #masks_2[masks_2>0]+=unique1.max()+1
    
    masks_2[masks_2>0]+=masks_1.max()+1
    masksM=masks_1+masks_2

    return(masksM)

#last update 14/02/24
#Pietro Marco D'Angelo
def merge_masks_to_n_separate_items(arr_masks):
    arr_max=[x.max() for x in arr_masks]
    to_add=arr_max[0]
    new_arr=[]
    new_arr.append(arr_masks[0])

    for x in range(len(arr_masks)-1):
        temp=arr_masks[x+1].copy()
        to_add+=1
        temp[temp>0]+=to_add
        to_add+=arr_max[x+1]
        new_arr.append(temp)

    masksM=new_arr[0]+new_arr[1]+new_arr[2]
    return(masksM)

#last update 10/02/24
#Pietro Marco D'Angelo
def split_enumerate_and_marge_n(masks):
  temp=separa_n_classi(masks)
  temp=assegna_numeri_univoci_n_classi(temp)
  masksM=merge_masks_to_n_separate_items(temp)
  return(masksM)

#last update 04/12/23
#Pietro Marco D'Angelo
def crea_dic_from_merged_masks(merged_masks):
    uniqueM, counts = np.unique(merged_masks, return_counts=True)
    indice_divisore = np.where(np.diff(uniqueM) > 1)[0]
    arr1 = uniqueM[: indice_divisore[0] + 1]
    arr2 = uniqueM[indice_divisore[0] + 1 :]

    dic={}
    for x in arr1:
        dic[x]=1

    for x in arr2:
        dic[x]=2

    del(dic[0])

    return(dic)

#last update 10/02/24
#Pietro Marco D'Angelo
def crea_dic_from_merged_masks_n(masksM):
    uniqueM = np.unique(masksM)
    indici_salto = np.where(np.diff(uniqueM) != 1)[0] + 1

    # Aggiungi gli indici di inizio e fine dell'array
    indici_inizio = np.concatenate(([0], indici_salto))
    indici_fine = np.concatenate((indici_salto, [len(uniqueM)]))

    # Crea gli array consecutivi
    array_consecutivi = [uniqueM[inizio:fine] for inizio, fine in zip(indici_inizio, indici_fine)]

    dic={}
    for x in range(len(array_consecutivi)):
      for y in array_consecutivi[x]:
        dic[y]=x+1
    
    del(dic[0])
    return(dic)

#last update 12/01/24
#Pietro Marco D'Angelo
def split_and_enumerate(masks):
    m1,m2=separa_2_classi(masks)

    m1=assegna_numeri_univoci(m1)
    m2=assegna_numeri_univoci(m2)

    return(m1,m2)

#last update 17/01/24
#Pietro Marco D'Angelo
def np_augmentation(array):
    list=[]
    for x in array:
        f=np.flip(x, 0)
        list.append(x)
        list.append(f)
        list.append(np.rot90(x, 1))
        list.append(np.rot90(f, 1))
        list.append(np.rot90(x, 2))
        list.append(np.rot90(f, 2))
        list.append(np.rot90(x, 3))
        list.append(np.rot90(f, 3))

    return(np.array(list))

#last update 12/01/24
#Pietro Marco D'Angelo
def plot_mask_with_pixel_value(masks,figsize=(8,8)):
    plt.figure(figsize=figsize)
    values = np.unique(masks.ravel())
    im = plt.imshow(masks)
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label="Pixel value = {l}".format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.show()




print('Load compleate!')