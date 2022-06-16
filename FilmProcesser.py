# %%defs
from os import chdir, listdir, path, getcwd
import os
from time import sleep
from multiprocessing import Pool, freeze_support
from configparser import ConfigParser
import pickle
import sys
import signal
import numpy as np
import rawpy as rp
import cv2
import tqdm
import funcs as f
from exiftool import ExifTool
from copy import deepcopy as copy # for debugging
from funcs import show, kill_cv2, show2 # for debugging

__version__ = 0.04

formats = [
    "crw",
    "cr2",
    "cr3",
    "nef",
    "arw",
    "rw2",
    "raf",
    "dng",
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


def unpack_params():
    return_folder = False
    if "original" not in getcwd():
        return_folder = True
        chdir("original")

    out = ConfigParser()
    out.read("params.ini")

    need_vig = out.getboolean("Process", "luminosity correction")
    need_exp = out.getboolean("Process", "exposure correction")
    crop = np.fromstring(out["Process"]["crop"], sep=",", dtype=int)

    img_mins = np.fromstring(out["Exposure"]["minimum values"], sep=",")
    img_maxs = np.fromstring(out["Exposure"]["maximum values"], sep=",")
    white = np.fromstring(out["Exposure"]["white correction"], sep=",")
    black = np.fromstring(out["Exposure"]["black correction"], sep=",")
    # comp_lo = np.fromstring(out["Exposure"]["shadow compression"], sep=",")
    exp_ev = None
    if need_exp:
        with open("params_exp.pkl", "rb") as file:
            exp_ev = pickle.load(file)

    gamma = np.fromstring(out["Color"]["gamma"], sep=",")
    ccm = np.fromstring(out["Color"]["ccm"], sep=",").reshape((3, 3))


    out = {
        "vig": need_vig,
        "exp": need_exp,
        "crop": crop,
        "exp ev": exp_ev,
        "mins": img_mins,
        "maxs": img_maxs,
        "white": white,
        "black": black,
        "gamma": gamma,
        # "shadow": comp_lo,
        "ccm": ccm,
        }

    if return_folder:
        chdir("..")

    return out


# https://stackoverflow.com/questions/68688044/cant-terminate-multiprocessing-program-with-ctrl-c
def init_pool_read():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# %% Processer defs

write_bit_depth = 65535
params = None
process_vig = None
et = None
def init_pool_process(og_path, workpath):
    global write_bit_depth
    global params
    global process_vig
    global et

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    chdir(og_path)
    config = ConfigParser()
    config.read("setup.ini")
    _interp = config["IMAGE PROCESSING"]["Interpolation method"]
    if _interp == "LINEAR":
        args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.LINEAR
    elif _interp == "AHD":
        args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.AHD
    elif _interp == "DHT":
        args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.DHT
    write_bit_depth = config.getint("IMAGE OUTPUT", "Bit depth")
    write_bit_depth = (2**write_bit_depth-1)<<(16-write_bit_depth)
    chdir(workpath)

    params = unpack_params()
    if params["vig"]:
        process_vig = np.load("original/vig.npy")

    et = ExifTool(path.join(og_path, "exiftool.exe"))
    et.run()


def img_process(name):
    try:
        with rp.imread("original/" + name) as raw:
            imgp = raw.postprocess(**args_full)
            black_level = raw.black_level_per_channel[0]

        crop = params["crop"]
        imgp = f.r2b(imgp) / 65535
        imgp -= (black_level / 65535)
        imgp = imgp[slice(*crop[:2]),slice(*crop[2:])]

        if params["exp"]:
            imgp *= 2**params["exp ev"][name]

        if process_vig is not None:
            imgp = np.divide(imgp, process_vig)

        imgp = 1 - imgp
        imgp = (imgp - params["mins"]) / (params["maxs"] - params["mins"])

        imgp = f.ccmGamma_apply(imgp, params)

        # TODO: dust mask integration
        # maskp = np.empty_like(imgp[...,0]).astype(np.uint8)
        # maskp = np.where(imgp>1, 255, 0)

        imgp = (np.clip(imgp, 0, 1) * 65535).astype(np.uint16)
        imgp = imgp & write_bit_depth

        cv2.imwrite(name[:-3].upper() + "tiff", imgp,
                    (cv2.IMWRITE_TIFF_COMPRESSION, 32946)) #to use deflate compression

        et.execute(f'-tagsfromfile=original/{name}',
                    "-overwrite_original_in_place", name[:-3]+"tiff")
    except KeyboardInterrupt:
        sys.exit()


def read_proxy(name):
    try:
        with rp.imread(name) as raw:
            _img = raw.postprocess(**args_proxy)
        _img = f.r2b(_img)
        return _img
    except KeyboardInterrupt:
        sys.exit()

# %% main defs
def main():
    def pack_params():
        # list 2 string
        def l2s(src:list):
            return ", ".join([str(item) for item in src])

        out = ConfigParser()
        out["Process"] = {
            "Luminosity correction": str(need_vig),
            "Exposure correction": str(need_exp),
            "Crop": l2s(crop),
            }
        out["Exposure"] = {
            "Minimum values": l2s(img_mins),
            "Maximum values": l2s(img_maxs),
            "White correction": l2s(color["white"]),
            "Black correction": l2s(color["black"]),
            # "Shadow compression": str(color["shadow"]),
            }
        out["Color"] = {
            "Gamma": l2s(color["gamma"]),
            "CCM": np.array2string(color["ccm"].flatten(), separator=',')[1:-1],
            }

        with open('params.ini', 'w') as file:
            out.write(file)

        if need_exp:
            exp_ev_out = dict(zip(imglist, exp_ev))
            with open("params_exp.pkl", "wb") as file:
                pickle.dump(exp_ev_out, file)

    # %% main init
    original_path = getcwd()
    freeze_support()
    print(f"FilmProcesser v{__version__}")
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
        need_vig = config.getboolean("IMAGE PROCESSING", "Luminosity correction")
        _interp = config["IMAGE PROCESSING"]["Interpolation method"]
        if _interp == "LINEAR":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.LINEAR
        elif _interp == "AHD":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.AHD
        elif _interp == "DHT":
            args_full["demosaic_algorithm"] = rp.DemosaicAlgorithm.DHT
        # reduce_factor = config.getint("IMAGE PROCESSING", "Previsualization reduce factor")
        reduce_height = config.getint("IMAGE PROCESSING", "Previsualization reduce height")
    else:
        print("Por favor ejecute setup.exe primero")
        print("Cerrando programa en 5 segundos...")
        sleep(5)
        sys.exit()

    while 1:
        filename = input("Ruta de carpeta: ")
        print()
        try:
            if "original" in filename:
                filename = path.dirname(filename)
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
    if "original" not in listdir():
        os.mkdir("original")
        flist = [file for file in listdir() if extension in file.lower()]
        for _file in flist:
            os.rename(_file, f"original/{_file}")

    chdir("original")
    flist = [file.lower() for file in listdir()]

    if "params.ini" in flist and "proxy.npy" in flist:
        print("Se ha encontrado archivo de configuración")
        key = input("Recolorizar? [Y]/n: ").lower()
        while key.lower() not in ("y", "n", ""):
            key = input("[Y]/n: ").lower()
        if key in ("y", ""):
            need_proxy = 2
        if key == "n":
            need_proxy = 0
    else:
        need_proxy = 1

    imglist = []
    if need_vig and need_proxy:
        if f"vig.{extension}" in flist:
            imglist.append(f"vig.{extension}")
        else:
            need_vig = False
            print(f"ATENCION: No se encontró archivo vig.{extension}")
            print("Desea continuar?")
            key = input("[Y]/n: ").lower()
            while key.lower() not in ("y", "n", ""):
                key = input("[Y]/n: ").lower()
            if key == "n":
                sys.exit()

    for file in flist:
        if file[-3:] == extension and "vig" not in file.lower():
            imglist.append(file)

    # %% proxy loading
    if need_proxy == 1:
        print("Leyendo RAWs...")
        # maybe specify max amount of processes
        with Pool(processes=None, initializer=init_pool_read) as exe:
            img = []
            for _res in exe.imap(read_proxy, imglist):
                img.append(_res)
            img = np.array(img)
        if img.ndim == 3:
            img = img[None,...]

        with rp.imread(imglist[0]) as raw:
            black_level = raw.black_level_per_channel[0]
        # black_level = 0

        with ExifTool(path.join(original_path, "exiftool.exe")) as et:
            meta = []
            for _img in imglist:
                meta.append(et.execute_json(_img)[0])
        exp_shutter = np.array([float(item["EXIF:ShutterSpeedValue"]) for item in meta])
        exp_aperture = np.array([float(item["EXIF:ApertureValue"]) for item in meta])
        exp_iso = np.array([float(item["EXIF:ISO"]) for item in meta])
        exp_ev = np.log2(exp_aperture**2 / (exp_shutter * exp_iso/100))
        exp_ev -= exp_ev.max()

        need_exp = False
        if len(set(exp_ev)) > 1:
            print("Advertencia: se han encontrado diferentes valores de exposición")
            key = input("Desea normalizar exposiciones? [Y]/n: ").lower()
            while 1:
                if key in ("y", ""):
                    need_exp = True
                    break
                if key == "n":
                    break

        # img0=copy(img) #for debugging purposes
        # %% proxy cropping
        # crop is (y1,y2,x1,x2)
        crop = []
        if need_crop:
            for i in range(2):
                crop_bool = np.sum(np.sum(img[0].astype(int)-black_level, axis=2), axis=(1-i))
                crop_bool = np.divide(crop_bool, np.percentile(crop_bool, 75))
                crop_bool = (crop_bool > 0.6).astype(int)
                crop_bool = np.diff(crop_bool)
                crop_bool2 = crop_bool.nonzero()[0]
                if len(crop_bool2) == 0:
                    crop.append([0, img.shape[1+i]])
                elif len(crop_bool2) == 1:
                    # -1 is true to false (start crop region)
                    if crop_bool[crop_bool2[0]] == -1:
                        # add cropping region warnings if needed
                        crop.append([0, crop_bool2[0]])
                    # 1 is false to true (end crop region)
                    else:
                        crop.append([crop_bool2[0], img.shape[1+i]])
                else:
                    if crop_bool[crop_bool2[0]] == 1 and crop_bool[crop_bool2[-1]] == -1:
                        crop.append([crop_bool2[0], crop_bool2[-1]])
                    else:
                        crop.append([0, img.shape[1+i]])
            crop = np.array(crop).flatten()

            print("----------")
            print("Previsualización de recorte...")
            print("Apretar Esc o Enter para cerrar")
            temp_img = f.resize(img[need_vig:], dim2=reduce_height)
            temp_img = f.norm(65535-temp_img,
                              th_lo=10, th_hi=90, skip=4).astype(np.float16)
            crop_temp = np.array([dim * reduce_height / img.shape[1] for dim in crop])
            crop = f.show_crop(temp_img, coords=crop_temp)
            del temp_img

        else:
            crop = [None] * 4

        crop = (crop * img.shape[1] / reduce_height).astype(int)
        img = img[:,slice(*crop[:2]),slice(*crop[2:])]
        crop = [dim*2 if dim is not None else None for dim in crop] #bc it was half-sized
        img = f.resize(img, dim2=reduce_height)

        # %% proxy process
        img = img / 65535
        img -= (black_level / 65535)
        if need_exp:
            img *= 2**exp_ev[:,None,None,None]


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

        print("Obteniendo percentiles extremos...")
        img = 1-img
        img_mins = np.percentile(img, 0.055, axis=(0,1,2))
        img_maxs = np.percentile(img, 99.75, axis=(0,1,2))

        print("Normalizando...")
        img = (img - img_mins) / (img_maxs - img_mins)

        print("Calculando gammas automáticamente...")
        gamma_pre = f.calc_gamma(img)

        print("Guardando proxies...")
        np.save("proxy.npy", img.astype(np.float32))

    if need_proxy == 2:
        print("Leyendo proxies...")
        img = np.load("proxy.npy")

        params = unpack_params()
        need_vig = params["vig"]
        need_exp = params["exp"]
        crop = params["crop"]
        img_mins = params["mins"]
        img_maxs = params["maxs"]

        print("Calculando gammas automáticamente...")
        gamma_pre = f.calc_gamma(img)

    if need_proxy:
        print("Obteniendo valores de colorización...")
        print("Apretar Esc o Enter para cerrar")
        color = f.ccmGamma(img, dim2=reduce_height, init_gamma=gamma_pre)

        pack_params()

    # =============================================================================
    # %% final pass
    # =============================================================================
    print("-----------")
    print("Iniciando proceso en 3 segundos...")
    print("Ctrl+C para cancelar")
    for __ in range(3*60):
        sleep(1/60)
    print("-----------")
    print("Procesando RAWs")
    print("Esto puede tomar unos minutos...")
    print("CTRL+C para cancelar (demora unos segundos)")
    print("-----------")

    imglist2 = [name for name in imglist if "vig" not in name.lower()]

    chdir(filename)
    with Pool(processes=max_processes, initializer=init_pool_process,
              initargs=[original_path, filename]) as pool:
        with tqdm.trange(len(imglist2), unit="photo") as progress_bar:
            res = pool.imap_unordered(img_process, imglist2, chunksize=1)
            for __ in res:
                progress_bar.update(1)
    f.kill_process("exiftool.exe")

    # for _img in imglist2:
    #     img_process(_img)

    chdir(filename)
    if "img" in dir():
        del_files = input(f"Desea eliminar los archivo proxy? ({round(img.nbytes/10**6)}Mb)\n"
                          "y/[N]: ")
    else:
        del_files = input("Desea eliminar los archivo proxy?\n"
                          "y/[N]: ")
    while del_files not in ("y", "n", ""):
        del_files = input("y/n: ").lower()
    if del_files == "y":
        f.cmd("del original\\*.npy")

if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        f.kill_process("exiftool.exe")
        print()
        print("--------------")
        print("Interrupción detectada")
        print("Cerrando programa...")
        sleep(2)
        sys.exit()
