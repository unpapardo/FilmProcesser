# %%defs
from os import chdir, listdir
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
# TODO: import exiftool
from copy import deepcopy as copy # for debugging
from funcs import show # for debugging

formats = [
    "cr2",
    "nef",
    "arw",
    "cr3",
    "crw",
    "rw2",
    "raf",
    "dng",
    "crw",
    ]

args_proxy = dict(
    demosaic_algorithm=rp.DemosaicAlgorithm.LINEAR,
    half_size=True,
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
    user_flip=0,
    # bright=4
    )

args_full = dict(
    # demosaic_algorithm=rp.DemosaicAlgorithm.DHT, #now set by setup.ini
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

# https://stackoverflow.com/questions/68688044/cant-terminate-multiprocessing-program-with-ctrl-c
def init_pool():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# TODO: Update to configparser would be nice
def pack_params(need_vig, perc_min_img, perc_max_img, black, white,
               gamma_all, gamma_b, gamma_g, gamma_r, ccm, crop, comp_lo):
    out = "\n".join([
        "Vig correction:",
        str(need_vig),
        "",
        "Minimum values:",
        np.array2string(perc_min_img, separator=',')[1:-1],
        "",
        "Maximum values:",
        np.array2string(perc_max_img, separator=',')[1:-1],
        "",
        "General Gamma:",
        str(gamma_all),
        "",
        "R Gamma:",
        str(gamma_r),
        "",
        "G Gamma:",
        str(gamma_g),
        "",
        "B Gamma:",
        str(gamma_b),
        "",
        "CCM:",
        np.array2string(ccm.flatten(), separator=',', max_line_width=150)[1:-1],
        "",
        "Crop:",
        ",".join([str(dim) for dim in crop]),
        "",
        "Black correction:",
        ",".join([str(item) for item in black]),
        "",
        "White correction:",
        ",".join([str(item) for item in white]),
        "",
        "Shadow compression",
        str(comp_lo)
    ])
    with open('params.txt', 'w') as file:
        file.write(out)


# TODO: Update to configparser would be nice
def unpack_params(orig=False):
    if orig:
        path = "original/params.txt"
    else:
        path = "params.txt"
    with open(path, "r") as file:
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

# %% Processer

def img_process(name):
    try:
        config = ConfigParser()
        config.read("setup.ini")
        _interp = config["IMAGE PROCESSING"]["Interpolation"]
        if _interp == "LINEAR":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.LINEAR
        elif _interp == "AHD":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.AHD
        elif _interp == "DHT":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.DHT
        bit_depth = config.getint("IMAGE OUTPUT", "Bit depth")
        bit_depth = (2**bit_depth-1)<<(16-bit_depth)

        (need_vig, perc_min_img, perc_max_img, black, white,
        gamma_all, gamma_b, gamma_g, gamma_r, ccm, crop, comp_lo) = unpack_params(orig=True)

        with rp.imread("original/" + name) as raw:
            imgp = raw.postprocess(**args_full)
            black_level = raw.black_level_per_channel[0]
        imgp = f.r2b(imgp) / 65535
        imgp = imgp - black_level / 65535
        imgp = imgp[slice(*crop[:2]),slice(*crop[2:])]
        collect()

        if need_vig:
            vigp = np.load("original/vig.npy")
            imgp = np.divide(imgp, vigp)
            del vigp
        collect()

        imgp = 1 - imgp
        for j in range(3):
            imgp[...,j] = imgp[...,j] - perc_min_img[j]
            imgp[...,j] = imgp[...,j] / (perc_max_img[j] - perc_min_img[j])

        imgp = (imgp + black[0]) / (1 + black[0])
        for _c in range(1,4):
            imgp[...,_c-1] = (imgp[...,_c-1] + black[_c]) / (1 + black[_c])

        imgp = imgp * (1 + white[0])
        for _c in range(1,4):
            imgp[...,_c-1] = imgp[...,_c-1] * (1 + white[_c])

        imgp = f.gammaBGR(imgp, gamma_all, gamma_b, gamma_g, gamma_r)
        imgp = f.CCM(imgp, ccm)
        imgp = f.compress_shadows(imgp, 0.55, comp_lo)

        imgp = np.interp(imgp, (0, 1), (0, 65535)).astype(np.uint16)
        imgp = imgp & bit_depth
        cv2.imwrite(name[:-4] + ".png", imgp)
    except KeyboardInterrupt:
        sys.exit()

# =============================================================================
# %% first run
# =============================================================================
def read_proxy(name):
    try:
        with rp.imread(name) as raw:
            _img = raw.postprocess(**args_proxy)
        _img = f.r2b(_img)
        return _img
    except KeyboardInterrupt:
        sys.exit()

def main():
# if __name__ == "__main__":
    freeze_support()
    print("FilmProcesser v0.02")
    print("-------------------")
    print()

    if "setup.ini" in listdir():
        config = ConfigParser()
        config.read("setup.ini")
        try:
            max_processes = config.getint("SYSTEM", "Process count")
        except ValueError:
            max_processes = None
        need_crop = config.getboolean("IMAGE PROCESSING", "Cropping")
        need_roi = config.getboolean("IMAGE PROCESSING", "Always set manual crop")
        need_vig = config.getboolean("IMAGE PROCESSING", "Luminosity correction")
        _interp = config["IMAGE PROCESSING"]["Interpolation"]
        if _interp == "LINEAR":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.LINEAR
        elif _interp == "AHD":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.AHD
        elif _interp == "DHT":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.DHT
        reduce_factor = config.getint("IMAGE PROCESSING", "Previsualization reduce factor")
    else:
        print("Por favor ejecute setup.exe primero")
        print("Cerrando programa en 5 segundos...")
        sleep(5)
        sys.exit()

    # filename = r"C:/new"
    while 1:
        filename = input("Ruta de carpeta: ")
        print()
        try:
            chdir(filename)
            files_found = False
            if "original" in listdir():
                chdir("original")
            for extension in formats:
                if [file for file in listdir() if extension in file.lower()]:
                    files_found = True
                    break
            if not files_found:
                raise FileNotFoundError
            break
        except FileNotFoundError:
            print("Carpeta no válida")

    chdir(filename)
    flist = listdir()
    if "original" not in flist:
        f.cmd("mkdir original")
    f.cmd(f"for %a in (*.{extension}) do move %a original/%a")
    chdir(filename + "/original")
    flist = [file.lower() for file in listdir()]

    if "params.txt" in flist:
        (need_vig, perc_min_img, perc_max_img, black, white,
         gamma_all, gamma_b, gamma_g, gamma_r, ccm, crop, comp_lo) = unpack_params()
        print("Se ha encontrado archivo de configuración")
        key = input("Recolorizar? [y]/n: ").lower()
        # key = "n"
        if key in ("", "y"):
            need_proxy = 2
        elif key == "n":
            need_proxy = 0
        else:
            while key.lower() not in ("y", "n"):
                key = input("y/n: ").lower()
                if key == "y":
                    need_proxy = 2
                    break
                if key == "n":
                    need_proxy = 0
    else:
        need_proxy = 1

    imglist = []
    if need_vig:
        if f"vig.{extension}" in flist:
            imglist.append(f"vig.{extension}")
        else:
            need_vig = False
            print(f"ATENCION: No se encontró archivo vig.{extension}")
            print("Desea continuar?")
            while 1:
                key = input("[y]/n: ").lower()
                if key in ("", "y"):
                    break
                if key == "n":
                    sys.exit()

    for file in flist:
        if file[-3:] == extension and "vig" not in file.lower():
            imglist.append(file)
    if need_proxy == 2:
        if "proxy.npy" not in listdir():
            need_proxy = 1
    chdir(filename + "/original")

    if need_proxy == 1:
        print("Leyendo RAWs...")
        # maybe specify max amount of processes
        with Pool(processes=None, initializer=init_pool) as exe:
            img = []
            for _res in exe.imap(read_proxy, imglist):
                img.append(_res)

        with rp.imread(imglist[0]) as raw:
            black_level = raw.black_level_per_channel[0]

        # TODO: In the future for metadata related stuff
        # with exiftool.ExifTool(r"D:\MPardo HDD\Downloads\exiftool-12.41\exiftool.exe") as et:
        #     meta = et.execute_json(*imglist)

        # img0=copy(img) #for debugging purposes

        margin=5
        # crop is (y1,y2,x1,x2)
        crop = []
        if need_crop and not need_roi:
            for i in range(2):
                crop_bool = np.sum(np.sum(img[0].astype(int)-black_level, axis=2), axis=(1-i))
                crop_bool = np.divide(crop_bool, np.percentile(crop_bool, 75))
                crop_bool = (crop_bool > 0.6).astype(int)
                crop_bool = np.diff(crop_bool)
                crop_bool2 = crop_bool.nonzero()[0]
                if len(crop_bool2) == 0:
                    crop.append([margin, -margin])
                elif len(crop_bool2) == 1:
                    # -1 is true to false (start crop region)
                    if crop_bool[crop_bool2[0]] == -1:
                        # add cropping region warnings if needed
                        crop.append([margin, crop_bool2[0]-margin])
                    # 1 is false to true (end crop region)
                    else:
                        crop.append([crop_bool2[0]+margin, -margin])
                else:
                    if crop_bool[crop_bool2[0]] == 1 and crop_bool[crop_bool2[-1]] == -1:
                        crop.append([crop_bool2[0]+margin, crop_bool2[-1]-margin])
                    else:
                        # TODO: do something if it doesnt match in the future, meanwhile do nothing
                        crop.append([margin, -margin])
            crop = list(np.array(crop).flatten())

            crop_img_prev = img[0].copy()[slice(*crop[:2]),slice(*crop[2:])]
            crop_img_prev = 1-f.norm(crop_img_prev)
            crop_img_prev = f.resize(crop_img_prev, reduce_factor)

            while 1:
                print("----------")
                print("Previsualización de recorte...")
                print("Apretar Esc para cerrar")
                print("NO CERRAR CON EL ÍCONO DE CERRAR")
                f.show(crop_img_prev)
                key = input("El area del auto recorte es correcto? [Y]/n: ").lower()
                if key in ("", "y"):
                    break
                if key == "n":
                    need_roi = True
                    break

        if need_crop and need_roi:
            crop_img_prev = img[0].copy()
            crop_img_prev = 1-f.norm(crop_img_prev)
            crop_img_prev = f.resize(crop_img_prev, reduce_factor)
            print("Ingrese área de recorte manualmente...")
            print("Para finalizar aprete Enter")
            roi = cv2.selectROI(crop_img_prev)
            cv2.destroyAllWindows()
            roi = [int(dim/(reduce_factor/100)) for dim in roi]
            crop = (roi[1], roi[1]+roi[3], roi[0], roi[0]+roi[2])

        if not need_crop:
            crop = [None] * 4

        img = np.array(img)
        img = img[:,slice(*crop[:2]),slice(*crop[2:])]
        crop = [dim*2 if dim is not None else None for dim in crop] #bc it was half-sized
        img = np.array([f.resize(image, reduce_factor) for image in img])
        img = img / 65535
        img = img - black_level / 65535

        if need_vig:
            if "vig.npy" not in listdir():
                print("Creando archivo de corrección de luminosidad...")
                with rp.imread(imglist[0]) as raw:
                    vig = raw.postprocess(**args_full)
                vig = vig[slice(*crop[:2]),slice(*crop[2:])]
                vig = vig - black_level
                vig = f.r2b(vig)
                vig = cv2.blur(vig, (150,150))
                vig = np.divide(vig, np.max(vig))
                np.save("vig.npy", vig.astype(np.float16))

            vig = cv2.blur(img[0], (30,30))
            vig = np.divide(vig, np.max(vig))
            img = np.delete(img, 0, 0)
            del imglist[0]

            print("Aplicando corrección de luminosidad...")
            img = np.divide(img, vig)

        print("Invirtiendo...")
        img = 1 - img

        print("Obteniendo percentiles extremos...")
        perc_max_img = np.empty(3)
        perc_min_img = np.empty(3)
        for j in range(3):
            perc_min_img[j] = np.percentile(img[..., j], 0.075)
            perc_max_img[j] = np.percentile(img[..., j], 99.8)*1.0025

        print("Normalizando...")
        for j in range(3):
            img[...,j] = img[...,j] - perc_min_img[j]
            img[...,j] = img[...,j] / (perc_max_img[j] - perc_min_img[j])

        print("Guardando proxies...")
        np.save("proxy.npy",img.astype(np.float32))

    if need_proxy == 2:
        print("Leyendo proxies...")
        img = np.load("proxy.npy")

    if need_proxy:
        print("Obteniendo valores de colorización...")
        ccm, black, white, gamma_all, gamma_b, gamma_g, gamma_r, comp_lo = f.ccmGamma(img)

        pack_params(need_vig, perc_min_img, perc_max_img, black, white,
                   gamma_all, gamma_b, gamma_g, gamma_r, ccm, crop, comp_lo)
    # =============================================================================
    # %% final pass
    # =============================================================================
    print("-----------")
    print("Procesando RAWs")
    print("Esto puede tomar unos minutos...")
    print("CTRL+C para cancelar (demora unos segundos)")
    print("-----------")
    chdir(filename)

    imglist2 = [name for name in imglist if "vig" not in name.lower()]

    with Pool(processes=max_processes, initializer=init_pool) as pool:
        with tqdm.trange(len(imglist2), unit="photo") as progress_bar:
            res = pool.imap_unordered(img_process, imglist2, chunksize=1)
            for __ in res:
                progress_bar.update(1)

    # for _img in imglist2:
    #     img_process(_img)

    del_files = input("Desea eliminar los archivo proxy? y/[n]: ")
    while del_files not in ("y", "n", ""):
        del_files = input("y/n: ").lower()
    if del_files == "y":
        f.cmd("del original\\*.npy")

if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("--------------")
        print("Interrupción detectada")
        print("Cerrando programa...")
        sleep(2)
        sys.exit()
