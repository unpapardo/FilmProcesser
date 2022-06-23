# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:30:13 2021

@author: mpardo
"""

# from time import sleep
from subprocess import run as sub_run
import cv2
import numpy as np
import rawpy as rp
import warnings
import psutil
import functools


def kill_cv2():
    cv2.destroyAllWindows()


def kill_cv2_on_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            wrap_out = func(*args, **kwargs)
        except:
            kill_cv2()
            raise
        return wrap_out
    return wrapper


def kill_process(name):
    for proc in psutil.process_iter():
        if proc.name() == name:
            proc.kill()


def cmd(comando):
    out = sub_run(comando, shell=True, text=True, capture_output=True)
    if out.stdout != "":
        print(out.stdout)
    elif out.stderr != "":
        print(out.stderr)


@kill_cv2_on_error
def show(src, fac=100, dim1: int = None, dim2: int = None, verbose=False):
    """
    Shows a scaled image or sequence of images. Also prints the current index.

    Parameters
    ----------
    src : np.ndarray
        Single image or array of images.
    fac : float, optional
        Scaling factor for the selection in percentage. By default no scaling
        is performed.
    """
    if src.ndim == 2:
        src = src[None, ...]
    elif src.ndim == 3:
        if src.shape[-1] == 3:
            src = src[None, ...]

    if src.dtype in (float, np.float16, np.float32):
        if src.max() > 50 and src.max() < 300:
            src /= 255
        elif src.max() >= 300:
            src /= 65535
        src = src.astype(float)

    if src.dtype == bool:
        src = src.astype(float)

    ilen = len(src)
    i_actual = 0

    change = True
    while 1:
        if change:
            if verbose:
                print(i_actual)
            cv2.imshow("asdf", resize(src[i_actual], fac, dim1, dim2))
        change = False

        key = cv2.waitKeyEx(1) & 0xFF
        if key == 27:
            break
        if key == ord("z"):
            change = True
            i_actual -= 1
            i_actual %= ilen
        elif key == ord("x"):
            change = True
            i_actual += 1
            i_actual %= ilen

    cv2.destroyAllWindows()


def show2(src):
    show(src, fac=None, dim1=None, dim2=720)


@kill_cv2_on_error
def show_crop(src, fac=100, dim1: int = None, dim2: int = None,
              verbose=False, coords=None) -> tuple:
    # crop is (y1,y2,x1,x2)
    # cv2 rectangle is (x1,y1), (x2,y2)
    _pressed = mode = y_check = x_check = x0 = y0 = None

    def mouse_click(event, x, y, flags, param):
        nonlocal coord_change
        nonlocal _pressed
        nonlocal mode
        nonlocal y_check, x_check
        nonlocal x0, y0

        if event == cv2.EVENT_LBUTTONDOWN:
            _pressed = True
            x_check = 0
            y_check = 0

            if np.abs(y - coords[0]) < 10:
                y_check = 1
            elif np.abs(y - coords[1]) < 10:
                y_check = 2
            if np.abs(x - coords[2]) < 10:
                x_check = 1
            elif np.abs(x - coords[3]) < 10:
                x_check = 2

            if x_check and y_check:
                mode = "vertex"
            elif x_check:
                mode = "col"
            elif y_check:
                mode = "row"
            else:
                mode = "drag"
                x0 = x
                y0 = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if _pressed:
                if mode == "vertex":
                    coords[-1+y_check] = y
                    coords[1+x_check] = x

                elif mode == "col":
                    coords[1+x_check] = x

                elif mode == "row":
                    coords[-1+y_check] = y

                elif mode == "drag":
                    coords[0:2] += (y - y0)
                    coords[2:4] += (x - x0)
                    x0 = x
                    y0 = y

        elif event == cv2.EVENT_LBUTTONUP:
            _pressed = False
            coords[:2] = np.clip(coords[:2], 2, temp.shape[0] - 2)
            coords[2:] = np.clip(coords[2:], 2, temp.shape[1] - 2)

    # crop is (y1,y2,x1,x2)
    # cv2 rectangle is (x1,y1), (x2,y2)
    if src.ndim == 2:
        src = src[None, ...]
    elif src.ndim == 3:
        if src.shape[-1] == 3:
            src = src[None, ...]

    src = f2u(src.copy(), 8)

    ilen = len(src)
    i_actual = 0

    if coords is None:
        src_check = resize(src[0], fac, dim1, dim2)
        coords = np.array(
            (src_check.shape[0] * 0.02, src_check.shape[0] * 0.98,
             src_check.shape[1] * 0.02, src_check.shape[1] * 0.98)).astype(int)
    else:
        coords = np.array(coords, dtype=int)
        if dim1 is not None:
            fac = dim1 / src.shape[1] * 100
        elif dim2 is not None:
            fac = dim2 / src.shape[2] * 100

        coords = (coords * (fac / 100)).astype(int)

    i_change = True
    cv2.namedWindow("Crop Select")
    cv2.setMouseCallback('Crop Select', mouse_click)
    while 1:
        if i_change:
            if verbose:
                print(i_actual)
            temp = resize(src[i_actual], fac, dim1, dim2)
            i_change = False
            coord_change = True

        vertex1 = (coords[2], coords[0])
        vertex2 = (coords[3], coords[1])
        temp2 = temp.copy()
        cv2.rectangle(temp2, vertex1, vertex2, (20, 20, 255), 2)
        cv2.imshow("Crop Select", temp2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:
            break
        if key == ord("z"):
            i_change = True
            i_actual -= 1
            i_actual %= ilen
        elif key == ord("x"):
            i_change = True
            i_actual += 1
            i_actual %= ilen

    cv2.destroyAllWindows()

    if dim1 is not None:
        fac = dim1 / src.shape[2] * 100
    elif dim2 is not None:
        fac = dim2 / src.shape[1] * 100

    coords = (coords / (fac / 100)).astype(int)
    coords[:2] = np.clip(coords[:2], 0, src.shape[1])
    coords[2:] = np.clip(coords[2:], 0, src.shape[2])

    return np.array((
        min(coords[0], coords[1]), max(coords[0], coords[1]),
        min(coords[2], coords[3]), max(coords[2], coords[3]),
    ))


def resize(src: np.ndarray, fac: float = None,
           dim1: int = None, dim2: int = None,
           nearest: bool = False, no_upsize: bool = False) -> np.ndarray:
    inter = cv2.INTER_NEAREST if nearest else None

    if (isinstance(src, np.ndarray) and (src.ndim == 2
                                         or (src.ndim == 3 and src.shape[-1] == 3))):
        if fac is not None and (dim1, dim2) == (None, None):
            if fac == 100:
                return src
            if fac > 100 and no_upsize:
                return src
            return cv2.resize(src, None, None, fac/100, fac/100, interpolation=inter)

        elif dim1 or dim2:
            if dim1 and dim2:
                warnings.warn(
                    "As dim1 and dim2 are provided, stretching might occur",
                    SyntaxWarning)
                dims = (dim1, dim2)
                if np.sign(src.shape[1] - src.shape[0]) != np.sign(dim1 - dim2):
                    dims = (dim2, dim1)
            elif dim1:
                fac = src.shape[1] / dim1
                dim2 = src.shape[0] / fac
                dims = (dim1, dim2)
            elif dim2:
                fac = src.shape[0] / dim2
                dim1 = src.shape[1] / fac
                dims = (dim1, dim2)

            if fac > 100 and no_upsize:
                return src

            return cv2.resize(src, [int(dim) for dim in dims], interpolation=inter)

        else:
            raise AttributeError("No dimensions specified")

    else:
        ret_type = type(src)
        if ret_type is np.ndarray:
            ret_type = np.array
        return ret_type([resize(_img, fac, dim1, dim2, nearest) for _img in src])


# def cropresize(src, fac, crop, y=0, x=0):
#     if fac == 100:
#         return src.copy()
#     fac /= 100
#     crop /= 100

#     if x > 0.5:
#         x = min(x, (1 - crop / 2))
#     else:
#         x = max(x, (crop / 2))
#     if y > 0.5:
#         y = min(y, (1 - crop / 2))
#     else:
#         y = max(y, (crop / 2))

#     win_h = int(src.shape[0] * fac)
#     win_w = int(src.shape[1] * fac)
#     dim = (win_w, win_h)

#     h = int(src.shape[0] * crop)
#     w = int(src.shape[1] * crop)
#     off_y = int(src.shape[0] * (y - crop / 2))
#     off_x = int(src.shape[1] * (x - crop / 2))

#     src = src[off_y:off_y + h, off_x:off_x + w]
#     src = cv2.resize(src, dim)
#     return src


def CCM(src_arr, ccm):
    src_arr = r2b(src_arr)
    src_arr = np.matmul(src_arr, ccm.T)
    src_arr = r2b(src_arr)
    return src_arr


# def gamma(src, gammaA, gammaB=None, gammaG=None, gammaR=None):
#     if gammaB and gammaG and gammaR:
#         vec_gamma = np.array([gammaB, gammaG, gammaR]) * gammaA
#         if src.ndim == 4:
#             vec_gamma = vec_gamma[None, ...]
#         # with np.errstate(invalid='ignore'):
#             # src = np.power(src, 1 / vec_gamma)
#         with warnings.catch_warnings():
#             warnings.filterwarnings('ignore')
#             src = np.exp(np.log(src) / vec_gamma)  # equivalent but faster

#     else:
#         with warnings.catch_warnings():
#             warnings.filterwarnings('ignore')
#             src = np.exp(np.log(src) / gammaA)

#     src[np.isnan(src)] = 0
#     return src


def gamma(src, gammaA, gammaB=None, gammaG=None, gammaR=None):
    src_out = np.empty_like(src)
    if gammaB and gammaG and gammaR and src.shape[-1] == 3:
        vec_gamma = np.array([gammaB, gammaG, gammaR]) * gammaA
        vec_gamma = 1 / vec_gamma
        for _j in range(3):
            src_out[..., _j] = cv2.pow(src[..., _j], vec_gamma[_j])

    else:
        src_out = cv2.pow(src, 1 / gammaA)

    src_out[np.isnan(src_out)] = 0
    return src_out


@kill_cv2_on_error
def ccmGamma(src_arr: np.ndarray, fac: float = None, dim1=None, dim2=720,
             init_gamma: np.ndarray = None) -> dict:
    def nothing(p): pass

    def gammaChange(p):
        global gamma_update
        gamma_update = True

    if src_arr.ndim == 3:
        src_arr = src_arr[None, ...]

    i = 0

    if 1:  # create trackbars
        cv2.namedWindow('image')
        cv2.namedWindow('histogram')
        cv2.namedWindow('tracks', cv2.WINDOW_NORMAL)

        cv2.createTrackbar('Clipping', 'tracks', 0, 1, nothing)

        cv2.createTrackbar('Black point', 'tracks', 25, 100, gammaChange)
        cv2.createTrackbar('White point', 'tracks', 50, 100, gammaChange)
        # cv2.createTrackbar('Compress lo', 'tracks', 0, 100, gammaChange)
        # cv2.createTrackbar('Compress hi', 'tracks', 0, 100, gammaChange)

        cv2.createTrackbar('Black R', 'tracks', 50, 100, gammaChange)
        cv2.createTrackbar('Black G', 'tracks', 50, 100, gammaChange)
        cv2.createTrackbar('Black B', 'tracks', 50, 100, gammaChange)

        cv2.createTrackbar('White R', 'tracks', 50, 100, gammaChange)
        cv2.createTrackbar('White G', 'tracks', 50, 100, gammaChange)
        cv2.createTrackbar('White B', 'tracks', 50, 100, gammaChange)

        if init_gamma is None:
            cv2.createTrackbar('All-gamma', 'tracks', 80, 100, gammaChange)
            cv2.createTrackbar('Rgamma',    'tracks', 80, 100, gammaChange)
            cv2.createTrackbar('Ggamma',    'tracks', 80, 100, gammaChange)
            cv2.createTrackbar('Bgamma',    'tracks', 80, 100, gammaChange)

        else:
            init_gamma[0] = 100/2 * (np.log(init_gamma[0])/np.log(4) + 1.6)
            init_gamma[1:] = 100/2 * (np.log(init_gamma[1:])/np.log(3) + 1.6)
            init_gamma = init_gamma.astype(int)
            cv2.createTrackbar('All-gamma', 'tracks',
                               init_gamma[0], 100, gammaChange)
            cv2.createTrackbar('Rgamma',    'tracks',
                               init_gamma[3], 100, gammaChange)
            cv2.createTrackbar('Ggamma',    'tracks',
                               init_gamma[2], 100, gammaChange)
            cv2.createTrackbar('Bgamma',    'tracks',
                               init_gamma[1], 100, gammaChange)

        cv2.createTrackbar('Disable CCM', 'tracks', 0, 1, nothing)
        cv2.createTrackbar('Reset',       'tracks', 0, 1, nothing)

        cv2.createTrackbar('R-R', 'tracks', 100, 250, nothing)
        cv2.createTrackbar('R-G/B', 'tracks', 100, 200, nothing)

        cv2.createTrackbar('G-G', 'tracks', 100, 250, nothing)
        cv2.createTrackbar('G-R/B', 'tracks', 100, 200, nothing)

        cv2.createTrackbar('B-B', 'tracks', 100, 250, nothing)
        cv2.createTrackbar('B-R/G', 'tracks', 100, 200, nothing)

    src = resize(src_arr[i], fac, dim1, dim2)

    gamma_update = True
    while 1:
        k = cv2.waitKeyEx(1) & 0xFF
        gamma_update = True
        if k == 27 or k == 13:
            break
        elif k == ord("z"):
            i -= 1
            i %= len(src_arr)
            src = resize(src_arr[i], fac, dim1, dim2)
        elif k == ord("x"):
            i += 1
            i %= len(src_arr)
            src = resize(src_arr[i], fac, dim1, dim2)

        if 1:  # Getting positions
            # comp_lo = cv2.getTrackbarPos('Compress lo', 'tracks') / 100
            # comp_hi = cv2.getTrackbarPos('Compress hi', 'tracks') / 100

            black = np.empty(4, dtype=float)
            black[0] = cv2.getTrackbarPos(
                'Black point', 'tracks') / 1000 * 15 - 0.375
            black[1] = cv2.getTrackbarPos('Black B', 'tracks') / 1000 * 4 - 0.2
            black[2] = cv2.getTrackbarPos('Black G', 'tracks') / 1000 * 4 - 0.2
            black[3] = cv2.getTrackbarPos('Black R', 'tracks') / 1000 * 4 - 0.2

            white = np.empty(4, dtype=float)
            white[0] = cv2.getTrackbarPos(
                'White point', 'tracks') / 1000 * 4 - 0.2
            white[1] = cv2.getTrackbarPos('White B', 'tracks') / 1000 * 2 - 0.1
            white[2] = cv2.getTrackbarPos('White G', 'tracks') / 1000 * 2 - 0.1
            white[3] = cv2.getTrackbarPos('White R', 'tracks') / 1000 * 2 - 0.1

            gammaa = cv2.getTrackbarPos('All-gamma', 'tracks')
            gammar = cv2.getTrackbarPos('Rgamma', 'tracks')
            gammag = cv2.getTrackbarPos('Ggamma', 'tracks')
            gammab = cv2.getTrackbarPos('Bgamma', 'tracks')

            gammaa = 4**((gammaa / 100) * 2 - 1.6)
            gammar = 3**((gammar / 100) * 2 - 1.6)
            gammag = 3**((gammag / 100) * 2 - 1.6)
            gammab = 3**((gammab / 100) * 2 - 1.6)

            clipping = cv2.getTrackbarPos('Clipping', 'tracks')
            disable = cv2.getTrackbarPos('Disable CCM', 'tracks')
            reset = cv2.getTrackbarPos('Reset', 'tracks')

            r1 = cv2.getTrackbarPos('R-R', 'tracks') / 100
            r2 = cv2.getTrackbarPos('R-G/B', 'tracks') / 100 - 1

            g1 = cv2.getTrackbarPos('G-G', 'tracks') / 100
            g2 = cv2.getTrackbarPos('G-R/B', 'tracks') / 100 - 1

            b1 = cv2.getTrackbarPos('B-B', 'tracks') / 100
            b2 = cv2.getTrackbarPos('B-R/G', 'tracks') / 100 - 1

        if reset:
            reset = 0

            cv2.setTrackbarPos('R-R', 'tracks', 100)
            cv2.setTrackbarPos('R-G/B', 'tracks', 100)

            cv2.setTrackbarPos('G-G', 'tracks', 100)
            cv2.setTrackbarPos('G-R/B', 'tracks', 100)

            cv2.setTrackbarPos('B-B', 'tracks', 100)
            cv2.setTrackbarPos('B-R/G', 'tracks', 100)

            cv2.setTrackbarPos('Reset', 'tracks', 0)

        if disable:
            ccm = np.eye(3)
        else:
            ccm = np.array([
                [r1, 0.5 + 0.5 * r2 - 0.5 * r1, 0.5 - 0.5 * r2 - 0.5 * r1],
                [0.5 + 0.5 * g2 - 0.5 * g1, g1, 0.5 - 0.5 * g2 - 0.5 * g1],
                [0.5 + 0.5 * b2 - 0.5 * b1, 0.5 - 0.5 * b2 - 0.5 * b1, b1]])

        # if gamma_update:
        if 1:
            temp = (src.copy() + black[0]) / (1 + black[0])
            temp = (temp + black[1:]) / (1 + black[1:])

            _white = (1 + white[0]) * (1 + white[1:])
            temp = temp * _white

            temp = gamma(temp, gammaa, gammab, gammag, gammar)
            temp = CCM(temp, ccm)
            gamma_update = False

            # temp = compress_shadows(temp, 0.55, comp_lo)
            # temp = compress_highlights(temp, 0.5, comp_hi)

            hist = histogram(temp)

        if clipping:
            temp[temp >= 1] = 0.001
            temp[temp <= 0] = 1

        cv2.imshow("image", temp)
        cv2.imshow("histogram", hist)

    cv2.destroyAllWindows()

    out = {"ccm": ccm,
           "white": white,
           "black": black,
           "gamma": (gammaa, gammab, gammag, gammar),
           # "shadow": comp_lo
           }

    return out


def ccmGamma_apply(src: np.ndarray, params_in: dict) -> np.ndarray:
    black = params_in["black"]
    white = params_in["white"]
    gammas = params_in["gamma"]
    ccm = params_in["ccm"]

    src = (src + black[0]) / (1 + black[0])
    src = (src + black[1:]) / (1 + black[1:])

    _white = (1 + white[0]) * (1 + white[1:])
    src = src * _white

    src = gamma(src, *gammas)
    src = CCM(src, ccm)

    # src = compress_shadows(src, 0.55, params_in["shadow"])

    return src

# def ccmGammaIR(src, fac=20, crop=100, y=0, x=0, apply=True):
#     b, g, r = cv2.split(src)
#     src = cv2.merge((g, r, b))

#     def gammaChange(p):
#         gamma_update = True

#     def nothing(p):
#         pass

#     cv2.namedWindow('image')
#     cv2.namedWindow('tracks', cv2.WINDOW_NORMAL)

#     cv2.createTrackbar('Gamma', 'tracks', 75, 200, gammaChange)

#     cv2.createTrackbar('R-R', 'tracks', 225 + 500, 1000, nothing)
#     cv2.createTrackbar('R-G', 'tracks', 100 + 500, 1000, nothing)
#     cv2.createTrackbar('R-B', 'tracks', -225 + 500, 1000, nothing)

#     cv2.createTrackbar('G-R', 'tracks', -125 + 500, 1000, nothing)
#     cv2.createTrackbar('G-G', 'tracks', 125 + 500, 1000, nothing)
#     cv2.createTrackbar('G-B', 'tracks', 100 + 500, 1000, nothing)

#     cv2.createTrackbar('B-R', 'tracks', -100 + 500, 1000, nothing)
#     cv2.createTrackbar('B-G', 'tracks', 0 + 500, 1000, nothing)
#     cv2.createTrackbar('B-B', 'tracks', 200 + 500, 1000, nothing)

#     cv2.createTrackbar('Autoset', 'tracks', 1, 1, nothing)
#     cv2.createTrackbar('Disable CCM', 'tracks', 0, 1, nothing)
#     cv2.createTrackbar('Reset', 'tracks', 0, 1, nothing)

#     gamma_update = True
#     src2 = cropresize(src, fac, crop, y, x)
#     while 1:
#         sleep(0.1)

#         k = cv2.waitKeyEx(1) & 0xFF
#         if k != 27:
#             gamma_update = True
#         if k == 27:
#             break
#         elif k == ord("r"):
#             fac = np.clip(max(fac * 1.1, fac + 1), 1, 100)
#             src2 = cropresize(src, fac, crop, y, x)
#         elif k == ord("f"):
#             fac = np.clip(min(fac * 0.9, fac - 1), 1, 100)
#             src2 = cropresize(src, fac, crop, y, x)

#         gammaa = cv2.getTrackbarPos('Gamma', 'tracks')

#         gammaa = 3**((gammaa / 100) * 2 - 1)

#         if gamma_update:
#             gammaTemp = gamma(src2.copy(), gammaa)
#             gamma_update = False

#         rr = cv2.getTrackbarPos('R-R', 'tracks') / 100 - 5
#         rg = cv2.getTrackbarPos('R-G', 'tracks') / 100 - 5
#         rb = cv2.getTrackbarPos('R-B', 'tracks') / 100 - 5

#         gr = cv2.getTrackbarPos('G-R', 'tracks') / 100 - 5
#         gg = cv2.getTrackbarPos('G-G', 'tracks') / 100 - 5
#         gb = cv2.getTrackbarPos('G-B', 'tracks') / 100 - 5

#         br = cv2.getTrackbarPos('B-R', 'tracks') / 100 - 5
#         bg = cv2.getTrackbarPos('B-G', 'tracks') / 100 - 5
#         bb = cv2.getTrackbarPos('B-B', 'tracks') / 100 - 5

#         disable = cv2.getTrackbarPos('Disable CCM', 'tracks')
#         reset = cv2.getTrackbarPos('Reset', 'tracks')
#         autoset = cv2.getTrackbarPos('Autoset', 'tracks')

#         if reset:
#             reset = 0

#             cv2.setTrackbarPos('R-R', 'tracks', 225 + 500)
#             cv2.setTrackbarPos('R-G', 'tracks', 100 + 500)
#             cv2.setTrackbarPos('R-B', 'tracks', -225 + 500)

#             cv2.setTrackbarPos('G-R', 'tracks', -125 + 500)
#             cv2.setTrackbarPos('G-G', 'tracks', 125 + 500)
#             cv2.setTrackbarPos('G-B', 'tracks', 100 + 500)

#             cv2.setTrackbarPos('B-R', 'tracks', -100 + 500)
#             cv2.setTrackbarPos('B-G', 'tracks', 0 + 500)
#             cv2.setTrackbarPos('B-B', 'tracks', 200 + 500)

#             cv2.setTrackbarPos('Reset', 'tracks', 0)

#         if disable:
#             ccm = np.array(
#                 [[2.25, 1., -2.25],
#                  [-1.25, 1.25, 1.],
#                  [-1., 0., 2.]])
#         else:
#             ccm2 = np.array([
#                 [rr, rg, rb],
#                 [gr, gg, gb],
#                 [br, bg, bb]])
#             if autoset:
#                 ccm = ccm2 / (np.sum(ccm2, axis=1)
#                               [:, None] * 0.75 + np.ones((3, 1)) * 0.25)
#             else:
#                 ccm = ccm2

#         temp = CCM(gammaTemp, ccm)
#         cv2.imshow("image", temp)

#     cv2.destroyAllWindows()

#     if apply:
#         src = gamma(src, gammaa)
#         src2 = CCM(src, ccm)
#         return src2
#     return ccm, gammaa


def diffshow(src, fac):
    src = np.interp(src, (src.min(), src.max()), (0, 1))
    show(src, fac)


def bgr2gray(src):
    return src[..., 2] * 0.299 + src[..., 1] * 0.587 + src[..., 0] * 0.114


def mask_edge(src):
    src = u2f(src)
    short = src.shape[0]

    a = 0.5
    src_norm = src / cv2.blur(src, k2(short*a))
    src_norm /= np.percentile(src_norm, 99.5)
    gray = src_norm[..., 0] ** 0.5 * \
        src_norm[..., 1] ** 2 * src_norm[..., 2] ** 0.5
    gray = np.clip(gray, 0, 5)

    b = 0.075
    blur = cv2.blur(gray, k2(short*b/3))
    for __ in range(3):
        blur = cv2.blur(blur, k2(short*b/3))
    diff = gray - blur

    c = 0.05
    diffblur = cv2.blur(np.abs(diff), k2(short*c/3))
    for __ in range(3):
        diffblur = cv2.blur(diffblur, k2(short*c/3))
    diffblur /= np.percentile(diffblur, 99.9)

    d = 2
    diff3 = diff - diffblur * d

    e = 0.15
    diff4 = np.where(diff3 > e, 255, 0)

    return diff4.astype(np.uint8)


def mask_thres(src, thres):
    c = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    __, c = cv2.threshold(c, thres, 255, cv2.THRESH_BINARY)
    return c.astype(np.uint8)


def k(ksize):
    ksize = max(ksize, 1)
    ker = np.ones((ksize, ksize), np.uint8)
    return ker


def k2(ksize):
    ksize = max(ksize, 1)
    return (int(ksize), int(ksize))


def sponerMsk(img1, img2, fac):
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2[:, :, 0] = np.multiply(img2[:, :, 0], (fac[0] / 255))
    img2[:, :, 1] = np.multiply(img2[:, :, 1], (fac[1] / 255))
    img2[:, :, 2] = np.multiply(img2[:, :, 2], (fac[2] / 255))

    out = (img1).astype(np.uint16) | img2.astype(np.uint16)

    return out.astype(np.uint8)


def inv(arr):
    return 255 - arr


def add(arr1, arr2):
    arr1 = arr1 // 255
    arr2 = arr2 // 255
    arr1 = 1 - arr1
    arr2 = 1 - arr2
    arrout = arr1 * arr2
    arrout = 1 - arrout

    return 255 * arrout


def norm(src: np.ndarray, th_lo: float = 0.1, th_hi: float = 99.9,
         skip: int = 3, dtype=float,
         rgb: bool = True, clip: bool = True) -> np.ndarray:
    src_out = u2f(src)
    if ((src_out.shape[-1] == 3 and src_out.ndim == 3) or src_out.ndim == 2):
        src_out = src_out[None, ...]

    src_perc = src_out[:, ::skip, ::skip, ...]
    if src_out.shape[-1] == 3 and rgb:
        perc_lo = np.percentile(src_perc, th_lo, axis=(0, 1, 2))
        perc_hi = np.percentile(src_perc, th_hi, axis=(0, 1, 2))
        src_out = (src_out - perc_lo) / (perc_hi - perc_lo)
    else:
        perc_lo = np.percentile(src_perc, th_lo)
        perc_hi = np.percentile(src_perc, th_hi)
        src_out = (src_out - perc_lo) / (perc_hi - perc_lo)

    if clip:
        src_out = np.clip(src_out, 0, 1)

    if dtype == np.uint8:
        src_out *= 255
    elif dtype == np.uint16:
        src_out *= 65535

    return src_out.astype(dtype)


def float2uint(src, out=16):
    assert out in (0, 8, 16), "'out' must be either 8 or 16"

    if src.dtype in (float, np.float64, np.float32, np.float16):
        src = np.clip(src, 0, 1)
        if out == 16:
            src *= 65535
            return src.astype(np.uint16)

        src *= 255
        return src.astype(np.uint8)

    if src.dtype == np.uint16:
        if out == 16:
            return src
        if out == 8:
            return (src//257).astype(np.uint8)
        if out == 0:
            return src.astype(int)

    if src.dtype == np.uint8:
        if out == 16:
            src = src.astype(np.uint16)
            return src * 257
        if out == 8:
            return src
        if out == 0:
            return src.astype(int)

    if src.dtype == int:
        if out == 16:
            return np.clip(src, 0, 65535).astype(np.uint16)
        if out == 8:
            return np.clip(src, 0, 255).astype(np.uint8)
        if out == 0:
            return src


def f2u(src, out):
    return (float2uint(src, out))


def uint2float(src):
    if src.dtype == int:
        return src.astype(float)
    if src.dtype == np.uint16:
        return src / 65535
    if src.dtype == np.uint8:
        return src / 255

    return src.copy()


def u2f(src):
    return (uint2float(src))


def r2b(src):
    return np.flip(src, axis=2)


def compress_shadows(src, fixed, fac):
    if fac < 0:
        raise ValueError("fac must be greater than 0")
    if fixed <= 0 or fixed >= 1:
        raise ValueError("Fixed point must be between 0 and 1")
    gamma_fac = (np.log(fixed + fac) - np.log(1 + fac)) / np.log(fixed)

    src_out = (src + fac) / (1 + fac)
    src_out = gamma(src_out, gamma_fac)

    return src_out


def compress_highlights(src, fixed, fac):
    if fac < 0:
        raise ValueError("fac must be greater than 0")
    if fixed <= 0 or fixed >= 1:
        raise ValueError("Fixed point must be between 0 and 1")
    fixed = 1 - fixed
    gamma_fac = (np.log(fixed + fac) - np.log(1 + fac)) / np.log(fixed)

    src_out = (1 - src + fac) / (1 + fac)
    src_out = 1 - gamma(src_out, gamma_fac)

    return src_out


def inpaint(src, msk, radius=6):
    msk = cv2.dilate(msk, k(radius))

    if src.dtype in (float, np.float16):
        src = src.astype(np.float32)
    src_out = np.empty_like(src)
    for _j in range(3):
        src_out[..., _j] = cv2.inpaint(src[..., _j], msk, 1, cv2.INPAINT_NS)

    return src_out


def histogram(src: np.ndarray, h_res: int = 256, v_res: int = 100,
              col_range: tuple = (0.2, 0.9), rgb: bool = True,
              in_fac: float = 100, out_fac: float = 100) -> np.ndarray:

    src = resize(src, in_fac)
    if not rgb and src.ndim == 3:
        src = bgr2gray(src)
    if src.ndim == 2:
        src = src[..., None]

    src = u2f(src)
    src = np.clip(src, 0, 1)
    src *= (h_res - 1)
    src = src.astype(int)

    out = []
    bins = []
    for channel in range(src.shape[-1]):
        bins.append(np.bincount(src[..., channel].flatten(), minlength=h_res))

    bins = np.array(bins)
    bins = bins/np.percentile(bins, 99) * v_res
    bins = bins.astype(int)

    for channel in range(src.shape[-1]):
        draw = np.zeros((v_res, h_res))
        for _col, _height in enumerate(bins[channel]):
            draw[:_height, _col] = 1
        draw = np.flipud(draw)
        draw *= (col_range[1] - col_range[0])
        draw += col_range[0]

        out.append(draw)

    out = np.squeeze(np.array(out))
    if out.ndim == 3:
        out = np.rollaxis(out, 0, 3)
    out = resize(out, out_fac)

    return out

# def XYZ_to_Oklab(src: np.ndarray)->np.ndarray:
#     # https://bottosson.github.io/posts/oklab/#converting-from-xyz-to-oklab
#     M1 = np.array([
#         [+0.8189330101, +0.0329845436, +0.0482003018],
#         [+0.3618667424, +0.9293118715, +0.2643662691],
#         [-0.1288597137, +0.0361456387, +0.6338517070],
#         ])

#     M2 = np.array([
#         [+0.2104542553, +1.9779984951, +0.0259040371],
#         [+0.7936177850, -2.4285922050, +0.7827717662],
#         [-0.0040720468, +0.4505937099, -0.8086757660],
#         ])

#     src = u2f(src)
#     # xyz to lms
#     src = r2b(src)
#     src = np.matmul(src,M1)
#     # lms to l'm's'
#     src = gamma(src, 1/3)
#     # l'm's' to Lab
#     src = np.matmul(src, M2)

#     return src

# def XYZ_OK(src):
#     return XYZ_to_Oklab(src)


# def Oklab_to_XYZ(src: np.ndarray)->np.ndarray:
#     # https://bottosson.github.io/posts/oklab/#converting-from-xyz-to-oklab
#     M1_inv = np.array([
#         [+1.22701385, -0.04058018, -0.07638128],
#         [-0.55779998,  1.11225687, -0.42148198],
#         [ 0.28125615, -0.07167668,  1.58616322],
#         ])

#     M2_inv = np.array([
#         [ 1.        ,  1.00000001,  1.00000005],
#         [ 0.39633779, -0.10556134, -0.08948418],
#         [ 0.21580376, -0.06385417, -1.29148554],
#         ])

#     src = u2f(src)
#     # xyz to lms
#     src = np.matmul(src, M2_inv)
#     # lms to l'm's'
#     src = gamma(src, 3)
#     # l'm's' to Lab
#     src = np.matmul(src, M1_inv)
#     src = r2b(src)

#     return src

# def OK_XYZ(src):
#     return XYZ_to_Oklab(src)


def calc_gamma(src, target_gamma: float = 0.5):
    if src.shape[-1] != 3:
        raise ValueError("Input image was not in color")

    if src.ndim == 2:
        src = src[None, ...]

    src = u2f(src)
    gammas = np.log(src.mean((0, 1, 2))) / np.log(target_gamma)
    gamma_max = gammas.max()
    gammas /= gamma_max

    return np.array((gamma_max, *gammas))


def thumb(filename, fac=100, dim1=None, dim2=None):
    with rp.imread(filename) as raw:
        thumb = raw.extract_thumb().data
    src = cv2.imdecode(np.frombuffer(thumb, dtype=np.uint8), -1)
    src = resize(src, fac, dim1, dim2)

    return src
