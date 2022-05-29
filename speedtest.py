# -*- coding: utf-8 -*-
"""
Created on Sun May 29 01:07:06 2022

@author: MPardo
"""
# %%defs
from os import chdir, listdir, path, getcwd
import os
from time import sleep
from gc import collect
from multiprocessing import Pool, freeze_support
from configparser import ConfigParser
import sys
import signal
import numpy as np
import rawpy as rp
import cv2
import tqdm
import funcs as f
import exiftool
from copy import deepcopy as copy # for debugging
from funcs import show # for debugging
from numba import njit
import warnings


def gamma(src, gammaA, gammaB=None, gammaG=None, gammaR=None):
    if gammaB and gammaG and gammaR:
        vec_gamma = np.array([gammaB, gammaG, gammaR]) * gammaA
        if src.ndim == 4:
            vec_gamma = vec_gamma[None, ...]
        # with np.errstate(invalid='ignore'):
            # src = np.power(src, 1 / vec_gamma)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            src = np.exp(np.log(src) / vec_gamma) #equivalent but faster

        return src

    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            src = np.exp(np.log(src) / gammaA)
        return src

def CCM(src_arr, ccm):
    return np.matmul(src_arr,np.flip(ccm.T))


def compress_shadows(src, fixed, fac):
    if fac < 0:
        raise ValueError("fac must be greater than 0")
    if fixed <= 0 or fixed >= 1:
        raise ValueError("Fixed point must be between 0 and 1")
    gamma_fac = (np.log(fixed + fac) - np.log(1 + fac)) / np.log(fixed)

    src_out = (src + fac) / (1 + fac)
    src_out = gamma(src_out, gamma_fac)

    return src_out


args_full = dict(
    demosaic_algorithm=rp.DemosaicAlgorithm.DHT, #now set by setup.ini
    # half_size=True,
    fbdd_noise_reduction=rp.FBDDNoiseReductionMode.Off,
    use_camera_wb=False,
    use_auto_wb=False,
    user_wb=[1, 1, 1, 1],
    output_color=rp.ColorSpace.raw,
    output_bps=16,
    user_sat=65535,
    user_black=0,
    no_auto_bright=True,
    no_auto_scale=True,
    gamma=(1, 1),
    # user_flip=0
    # bright=4
    )

chdir(r"E:\Fotos\1 RAW\2021-02-11 Rollos Embarazo, Coq, Sur, Caro\Rollo Caro shibari")
def unpack_params(orig = False):
    unpack_path = "params.txt"
    if not orig:
        unpack_path = "original/" + unpack_path
    with open(unpack_path, "r") as file:
        params = file.readlines()
    need_vig = False
    if params[1] == "True\n":
        need_vig = True
    if "None" not in (params[4], params[7]):
        perc_min_img = np.fromstring(params[4], sep=",")
        perc_max_img = np.fromstring(params[7], sep=",")
    else:
        perc_min_img = None
        perc_max_img = None
    gamma_all = float(params[10])
    gamma_r = float(params[13])
    gamma_g = float(params[16])
    gamma_b = float(params[19])
    ccm = np.fromstring(params[22], sep=",").reshape((3, 3))
    crop = params[25].split(",")
    crop = [int(dim) for dim in crop]
    black = np.fromstring(params[28], sep=",")
    white = np.fromstring(params[31], sep=",")
    comp_lo = float(params[34])

    return (need_vig, perc_min_img, perc_max_img, black, white,
            gamma_all, gamma_b, gamma_g, gamma_r, ccm, crop, comp_lo)

process_params = unpack_params()

(need_vig, perc_min_img, perc_max_img, black, white,
gamma_all, gamma_b, gamma_g, gamma_r, ccm, crop, comp_lo) = process_params

with rp.imread(r"E:\Fotos\1 RAW\2021-02-11 Rollos Embarazo, Coq, Sur, Caro\Rollo Caro shibari\original\IMG_2285.CR2") as raw:
    imgp = raw.postprocess(**args_full)
    black_level = raw.black_level_per_channel[0]

imgp = f.r2b(imgp) / 65535
imgp = imgp - black_level / 65535
imgp = imgp[slice(*crop[:2]),slice(*crop[2:])]

process_vig = np.ones_like(imgp)*0.5

if process_vig is not None:
    imgp = np.divide(imgp, process_vig)

imgp = 1 - imgp
imgp = (imgp - perc_min_img) / (perc_max_img - perc_min_img)

imgp = (imgp + black[0]) / (1 + black[0])
imgp = (imgp + black[1:]) / (1 + black[1:])

_white = (1 + white[0]) * (1 + white[1:])
imgp = imgp * _white

imgp = gamma(imgp, gamma_all, gamma_b, gamma_g, gamma_r)
imgp = CCM(imgp, ccm)
imgp = compress_shadows(imgp, 0.55, comp_lo)

imgp = (np.clip(imgp, 0, 1) * 65535).astype(np.uint16)
