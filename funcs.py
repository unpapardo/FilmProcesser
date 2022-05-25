# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:30:13 2021

@author: mpardo
"""

from time import sleep
from subprocess import run as sub_run
import cv2
import numpy as np


def cmd(comando):
    out = sub_run(comando, shell=True, text=True, capture_output=True)
    if out.stdout != "":
        print(out.stdout)
    elif out.stderr != "":
        print(out.stderr)


def show(src, fac=100, full=True):
    src = np.squeeze(src)
    im_y = int(src.shape[0] * fac / 100)
    im_x = int(src.shape[1] * fac / 100)
    dim = (im_x, im_y)
    src2 = cv2.resize(src, dim)
    cv2.imshow("asdf", src2)
    if full:
        cv2.waitKey()
        cv2.destroyAllWindows()


def resize(src, fac=30):
    if fac == 100:
        return src

    im_y = int(src.shape[0] * fac / 100)
    im_x = int(src.shape[1] * fac / 100)
    dim = (im_x, im_y)
    src = cv2.resize(src, dim)
    return src


def cropresize2(src, fac, crop, y, x):
    if fac == 100:
        return src.copy()
    fac /= 100
    crop /= 100

    if x > 0.5:
        x = min(x, (1 - crop / 2))
    else:
        x = max(x, (crop / 2))
    if y > 0.5:
        y = min(y, (1 - crop / 2))
    else:
        y = max(y, (crop / 2))

    win_h = int(src.shape[0] * fac)
    win_w = int(src.shape[1] * fac)
    dim = (win_w, win_h)

    h = int(src.shape[0] * crop)
    w = int(src.shape[1] * crop)
    off_y = int(src.shape[0] * (y - crop / 2))
    off_x = int(src.shape[1] * (x - crop / 2))

    src = src[off_y:off_y + h, off_x:off_x + w]
    src = cv2.resize(src, dim)
    return src


def CCM(src_arr, ccm):
    ccm = np.flip(ccm)
    out = np.empty_like(src_arr)
    for i in range(3):
        out[..., i] = (src_arr[..., 0] * ccm[i, 0] +
                       src_arr[..., 1] * ccm[i, 1] +
                       src_arr[..., 2] * ccm[i, 2])
    return out


def gamma(src, gamma):
    if gamma != 1:
        with np.errstate(invalid='ignore'):
            src = src**(1 / gamma)
    return src


def gammaBGR(src, gammaA, gammaB, gammaG, gammaR):
    src[..., 0] = gamma(src[..., 0], gammaB * gammaA)
    src[..., 1] = gamma(src[..., 1], gammaG * gammaA)
    src[..., 2] = gamma(src[..., 2], gammaR * gammaA)

    return src


def nothing(p):
    pass


def ccmGamma(src_arr, fac=100, crop=100, y=0, x=0):
    def gammaChange(p):
        gammaUpdate = True

    if src_arr.ndim == 3:
        src_arr = src_arr[None, ...]

    i = 0

    cv2.namedWindow('image')
    cv2.namedWindow('tracks', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Clipping', 'tracks', 0, 1, nothing)

    cv2.createTrackbar('Compress lo', 'tracks', 0, 100, gammaChange)
    # cv2.createTrackbar('Compress hi', 'tracks', 0, 100, gammaChange)

    cv2.createTrackbar('Black point', 'tracks', 25, 100, gammaChange)
    # cv2.createTrackbar('Black R', 'tracks', 50, 100, gammaChange)
    # cv2.createTrackbar('Black G', 'tracks', 50, 100, gammaChange)
    # cv2.createTrackbar('Black B', 'tracks', 50, 100, gammaChange)

    cv2.createTrackbar('White point', 'tracks', 50, 100, gammaChange)
    # cv2.createTrackbar('White R', 'tracks', 50, 100, gammaChange)
    # cv2.createTrackbar('White G', 'tracks', 50, 100, gammaChange)
    # cv2.createTrackbar('White B', 'tracks', 50, 100, gammaChange)

    cv2.createTrackbar('All-gamma', 'tracks', 80, 100, gammaChange)
    cv2.createTrackbar('Rgamma',    'tracks', 80, 100, gammaChange)
    cv2.createTrackbar('Ggamma',    'tracks', 80, 100, gammaChange)
    cv2.createTrackbar('Bgamma',    'tracks', 80, 100, gammaChange)

    cv2.createTrackbar('Autoset',     'tracks', 1, 1, nothing)
    cv2.createTrackbar('Normalize',   'tracks', 0, 1, nothing)
    cv2.createTrackbar('Disable CCM', 'tracks', 0, 1, nothing)
    cv2.createTrackbar('Reset',       'tracks', 0, 1, nothing)

    cv2.createTrackbar('R-R', 'tracks',  175 + 250, 500, nothing)
    cv2.createTrackbar('R-G', 'tracks', -100 + 250, 500, nothing)
    cv2.createTrackbar('R-B', 'tracks',   25 + 250, 500, nothing)

    cv2.createTrackbar('G-R', 'tracks', - 15 + 250, 500, nothing)
    cv2.createTrackbar('G-G', 'tracks',  130 + 250, 500, nothing)
    cv2.createTrackbar('G-B', 'tracks', - 15 + 250, 500, nothing)

    cv2.createTrackbar('B-R', 'tracks',   20 + 250, 500, nothing)
    cv2.createTrackbar('B-G', 'tracks', - 95 + 250, 500, nothing)
    cv2.createTrackbar('B-B', 'tracks',  175 + 250, 500, nothing)

    src = cropresize2(src_arr[i], fac, crop, y, x)

    gammaUpdate = True
    while 1:
        if 1:
            sleep(0.1)
            k = cv2.waitKeyEx(1) & 0xFF
            if k != 27:
                gammaUpdate = True
            if k == 27:
                break
            elif k == ord("z"):
                if i == 0:
                    i = len(src_arr) - 1
                else:
                    i -= 1
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("x"):
                if i == len(src_arr) - 1:
                    i = 0
                else:
                    i += 1
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("r"):
                fac = np.clip(max(fac * 1.1, fac + 1), 1, 100)
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("f"):
                fac = np.clip(min(fac * 0.9, fac - 1), 1, 100)
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("q"):
                crop = np.clip(crop + 5, 1, 100)
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("e"):
                crop = np.clip(crop - 5, 1, 100)
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("w"):
                y = np.clip(y - 0.05, 0, 1)
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("s"):
                y = np.clip(y + 0.05, 0, 1)
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("a"):
                x = np.clip(x - 0.05, 0, 1)
                src = cropresize2(src_arr[i], fac, crop, y, x)
            elif k == ord("d"):
                x = np.clip(x + 0.05, 0, 1)
                src = cropresize2(src_arr[i], fac, crop, y, x)

        comp_lo = cv2.getTrackbarPos('Compress lo', 'tracks') / 100
        # comp_hi = cv2.getTrackbarPos('Compress hi', 'tracks') / 100

        black = [0] * 4
        black[0] = cv2.getTrackbarPos('Black point', 'tracks') / 1000 * 15 - 0.375
        # black[1] = cv2.getTrackbarPos('Black B', 'tracks') / 1000 * 4 - 0.2
        # black[2] = cv2.getTrackbarPos('Black G', 'tracks') / 1000 * 4 - 0.2
        # black[3] = cv2.getTrackbarPos('Black R', 'tracks') / 1000 * 4 - 0.2

        white = [0] * 4
        white[0] = cv2.getTrackbarPos('White point', 'tracks') / 1000 * 4 - 0.2
        # white[1] = cv2.getTrackbarPos('White B', 'tracks') / 1000 * 2 - 0.1
        # white[2] = cv2.getTrackbarPos('White G', 'tracks') / 1000 * 2 - 0.1
        # white[3] = cv2.getTrackbarPos('White R', 'tracks') / 1000 * 2 - 0.1

        gammaa = cv2.getTrackbarPos('All-gamma', 'tracks')
        gammar = cv2.getTrackbarPos('Rgamma', 'tracks')
        gammag = cv2.getTrackbarPos('Ggamma', 'tracks')
        gammab = cv2.getTrackbarPos('Bgamma', 'tracks')

        gammaa = 4**((gammaa / 100) * 2 - 1.6)
        gammar = 3**((gammar / 100) * 2 - 1.6)
        gammag = 3**((gammag / 100) * 2 - 1.6)
        gammab = 3**((gammab / 100) * 2 - 1.6)

        if gammaUpdate:
            gammaTemp = (src.copy() + black[0]) / (1 + black[0])
            # for _c in range(1,4):
            #     gammaTemp[...,_c-1] = (gammaTemp[...,_c-1] + black[_c]) / (1 + black[_c])

            gammaTemp = gammaTemp * (1 + white[0])
            # for _c in range(1,4):
            #     gammaTemp[...,_c-1] = gammaTemp[...,_c-1] * (1 + white[_c])

            gammaTemp = gammaBGR(gammaTemp, gammaa, gammab, gammag, gammar)
            gammaUpdate = False

        clipping = cv2.getTrackbarPos('Clipping', 'tracks')
        disable = cv2.getTrackbarPos('Disable CCM', 'tracks')
        reset = cv2.getTrackbarPos('Reset', 'tracks')
        normalize = cv2.getTrackbarPos('Normalize', 'tracks')
        autoset = cv2.getTrackbarPos('Autoset', 'tracks')

        rr = cv2.getTrackbarPos('R-R', 'tracks') / 100 - 2.5
        rg = cv2.getTrackbarPos('R-G', 'tracks') / 100 - 2.5
        rb = cv2.getTrackbarPos('R-B', 'tracks') / 100 - 2.5

        gr = cv2.getTrackbarPos('G-R', 'tracks') / 100 - 2.5
        gg = cv2.getTrackbarPos('G-G', 'tracks') / 100 - 2.5
        gb = cv2.getTrackbarPos('G-B', 'tracks') / 100 - 2.5

        br = cv2.getTrackbarPos('B-R', 'tracks') / 100 - 2.5
        bg = cv2.getTrackbarPos('B-G', 'tracks') / 100 - 2.5
        bb = cv2.getTrackbarPos('B-B', 'tracks') / 100 - 2.5

        if reset:
            reset = 0

            cv2.setTrackbarPos('R-R', 'tracks', 175 + 250)
            cv2.setTrackbarPos('R-G', 'tracks', -100 + 250)
            cv2.setTrackbarPos('R-B', 'tracks', 25 + 250)

            cv2.setTrackbarPos('G-R', 'tracks', -15 + 250)
            cv2.setTrackbarPos('G-G', 'tracks', 130 + 250)
            cv2.setTrackbarPos('G-B', 'tracks', -15 + 250)

            cv2.setTrackbarPos('B-R', 'tracks', 20 + 250)
            cv2.setTrackbarPos('B-G', 'tracks', -95 + 250)
            cv2.setTrackbarPos('B-B', 'tracks', 175 + 250)

            cv2.setTrackbarPos('Reset', 'tracks', 0)

        if normalize:
            normalize = 0

            norm_fac_r = rr+rg+rb
            rr /= norm_fac_r
            rg /= norm_fac_r
            rb /= norm_fac_r

            norm_fac_g = gr+gg+gb
            gr /= norm_fac_g
            gg /= norm_fac_g
            gb /= norm_fac_g

            norm_fac_b = br+bg+bb
            br /= norm_fac_b
            bg /= norm_fac_b
            bb /= norm_fac_b

            cv2.setTrackbarPos('R-R', 'tracks', int(rr * 100 + 250))
            cv2.setTrackbarPos('R-G', 'tracks', int(rg * 100 + 250))
            cv2.setTrackbarPos('R-B', 'tracks', int(rb * 100 + 250))

            cv2.setTrackbarPos('G-R', 'tracks', int(gr * 100 + 250))
            cv2.setTrackbarPos('G-G', 'tracks', int(gg * 100 + 250))
            cv2.setTrackbarPos('G-B', 'tracks', int(gb * 100 + 250))

            cv2.setTrackbarPos('B-R', 'tracks', int(br * 100 + 250))
            cv2.setTrackbarPos('B-G', 'tracks', int(bg * 100 + 250))
            cv2.setTrackbarPos('B-B', 'tracks', int(bb * 100 + 250))

            cv2.setTrackbarPos('Normalize', 'tracks', 0)

        if disable:
            ccm = np.array([
                [1.75, -1.00, 0.25],
                [-0.15, 1.30, -0.15],
                [0.20, -0.95, 1.75]])
        else:
            ccm2 = np.array([
                [rr, rg, rb],
                [gr, gg, gb],
                [br, bg, bb]])
            if autoset:
                ccm = ccm2 / (np.sum(ccm2, axis=1)[:, None] * 0.75 +
                              np.ones((3, 1)) * 0.25)
            else:
                ccm = ccm2

        temp = CCM(gammaTemp, ccm)


        temp = compress_shadows(temp, 0.55, comp_lo)
        # temp = compress_highlights(temp, 0.5, comp_hi)

        if clipping:
            temp[temp >= 1] = 0.001
            temp[temp <= 0] = 1
            temp[np.isnan(temp)] = 1

        cv2.imshow("image", temp)

    cv2.destroyAllWindows()

    return ccm, black, white, gammaa, gammab, gammag, gammar, comp_lo


def ccmGammaIR(src, fac=20, crop=100, y=0, x=0, apply=True):
    b, g, r = cv2.split(src)
    src = cv2.merge((g, r, b))

    def gammaChange(p):
        global gammaUpdate
        gammaUpdate = True

    cv2.namedWindow('image')
    cv2.namedWindow('tracks', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Gamma', 'tracks', 75, 200, gammaChange)

    cv2.createTrackbar('R-R', 'tracks', 225 + 500, 1000, nothing)
    cv2.createTrackbar('R-G', 'tracks', 100 + 500, 1000, nothing)
    cv2.createTrackbar('R-B', 'tracks', -225 + 500, 1000, nothing)

    cv2.createTrackbar('G-R', 'tracks', -125 + 500, 1000, nothing)
    cv2.createTrackbar('G-G', 'tracks', 125 + 500, 1000, nothing)
    cv2.createTrackbar('G-B', 'tracks', 100 + 500, 1000, nothing)

    cv2.createTrackbar('B-R', 'tracks', -100 + 500, 1000, nothing)
    cv2.createTrackbar('B-G', 'tracks', 0 + 500, 1000, nothing)
    cv2.createTrackbar('B-B', 'tracks', 200 + 500, 1000, nothing)

    cv2.createTrackbar('Autoset', 'tracks', 1, 1, nothing)
    cv2.createTrackbar('Disable CCM', 'tracks', 0, 1, nothing)
    cv2.createTrackbar('Reset', 'tracks', 0, 1, nothing)

    gammaUpdate = True
    src2 = cropresize2(src, fac, crop, y, x)
    while 1:
        from time import sleep
        sleep(0.1)

        k = cv2.waitKeyEx(1) & 0xFF
        if k != 27:
            gammaUpdate = True
        if k == 27:
            break
        elif k == ord("r"):
            fac = np.clip(max(fac * 1.1, fac + 1), 1, 100)
            src2 = cropresize2(src, fac, crop, y, x)
        elif k == ord("f"):
            fac = np.clip(min(fac * 0.9, fac - 1), 1, 100)
            src2 = cropresize2(src, fac, crop, y, x)

        gammaa = cv2.getTrackbarPos('Gamma', 'tracks')

        gammaa = 3**((gammaa / 100) * 2 - 1)

        if gammaUpdate:
            gammaTemp = gamma(src2.copy(), gammaa)
            gammaUpdate = False

        rr = cv2.getTrackbarPos('R-R', 'tracks') / 100 - 5
        rg = cv2.getTrackbarPos('R-G', 'tracks') / 100 - 5
        rb = cv2.getTrackbarPos('R-B', 'tracks') / 100 - 5

        gr = cv2.getTrackbarPos('G-R', 'tracks') / 100 - 5
        gg = cv2.getTrackbarPos('G-G', 'tracks') / 100 - 5
        gb = cv2.getTrackbarPos('G-B', 'tracks') / 100 - 5

        br = cv2.getTrackbarPos('B-R', 'tracks') / 100 - 5
        bg = cv2.getTrackbarPos('B-G', 'tracks') / 100 - 5
        bb = cv2.getTrackbarPos('B-B', 'tracks') / 100 - 5

        disable = cv2.getTrackbarPos('Disable CCM', 'tracks')
        reset = cv2.getTrackbarPos('Reset', 'tracks')
        autoset = cv2.getTrackbarPos('Autoset', 'tracks')

        if reset:
            reset = 0

            cv2.setTrackbarPos('R-R', 'tracks', 225 + 500)
            cv2.setTrackbarPos('R-G', 'tracks', 100 + 500)
            cv2.setTrackbarPos('R-B', 'tracks', -225 + 500)

            cv2.setTrackbarPos('G-R', 'tracks', -125 + 500)
            cv2.setTrackbarPos('G-G', 'tracks', 125 + 500)
            cv2.setTrackbarPos('G-B', 'tracks', 100 + 500)

            cv2.setTrackbarPos('B-R', 'tracks', -100 + 500)
            cv2.setTrackbarPos('B-G', 'tracks', 0 + 500)
            cv2.setTrackbarPos('B-B', 'tracks', 200 + 500)

            cv2.setTrackbarPos('Reset', 'tracks', 0)

        if disable:
            ccm = np.array(
                [[2.25, 1., -2.25],
                 [-1.25, 1.25, 1.],
                 [-1., 0., 2.]])
        else:
            ccm2 = np.array([
                [rr, rg, rb],
                [gr, gg, gb],
                [br, bg, bb]])
            if autoset:
                ccm = ccm2 / (np.sum(ccm2, axis=1)
                              [:, None] * 0.75 + np.ones((3, 1)) * 0.25)
            else:
                ccm = ccm2

        temp = CCM(gammaTemp, ccm)
        cv2.imshow("image", temp)

    cv2.destroyAllWindows()

    if apply:
        src = gamma(src, gammaa)
        src2 = CCM(src, ccm)
        return src2
    else:
        return ccm, gammaa


def mask_edge(src):
    a = np.interp(src, (10000, 65535), (0, 65535)).astype(np.uint16)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.blur(a, (35, 35))
    diff = np.clip(a.astype(np.int32) - b.astype(np.int32),
                   0, 65535).astype(np.uint16)
    diffblur = cv2.blur(diff, (99, 99))
    diffblur = cv2.blur(diffblur, (99, 99))
    diffblur = cv2.blur(diffblur, (99, 99))
    diffnorm = np.interp(
        diff,
        (np.min(diff), np.max(diff) * 0.9),
        (0, 65535)
    ).astype(np.uint16)
    diffblurnorm = np.interp(
        diffblur,
        (np.min(diffblur), np.max(diffblur)),
        (0, 65535)
    ).astype(np.uint16)
    diff3 = np.clip(
        diffnorm.astype(np.int32) - diffblurnorm.astype(np.int32) * 1.3,
        0, 65535
    ).astype(np.uint16)
    ret, diff4 = cv2.threshold(diff3, 5000, 255, cv2.THRESH_BINARY)

    return (diff4).astype(np.uint8)


def mask_thres(src, thres):
    c = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, c = cv2.threshold(c, thres, 255, cv2.THRESH_BINARY)
    return c.astype(np.uint8)


def k(ksize):
    ker = np.ones((ksize, ksize), np.uint8)
    return ker


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


def norm(src, thLow=0.1, thHigh=99.9, dtype=float):
    for j in range(3):
        src[..., j] = np.interp(src[..., j],
            (np.percentile(src[..., j], thLow),
             np.percentile(src[..., j], thHigh)),
            (0, 1)
        )
    if dtype == np.uint8:
        src *= 255
    elif dtype == np.uint16:
        src *= 65535

    return src.astype(dtype)


def float2uint(src, out=16):
    assert out in (8, 16), "'out' must be either 8 or 16"
    src = np.clip(src, 0, 1)
    if out == 16:
        src *= 65535
        return src.astype(np.uint16)
    else:
        src *= 255
        return src.astype(np.uint8)


def r2b(src):
    return cv2.cvtColor(src, cv2.COLOR_RGB2BGR)

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
