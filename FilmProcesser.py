# %%defs
from os import chdir, listdir
from time import time, sleep
from gc import collect
import rawpy as rp
# from copy import deepcopy as copy
from multiprocessing import Pool, freeze_support
import cv2
import numpy as np
import funcs as f
import signal
import tqdm
# import exiftool
# from funcs import show

maxWorkers = None
roi_needed = False

proxy_args = dict(
    demosaic_algorithm=rp.DemosaicAlgorithm.LINEAR,
    half_size=True,
    fbdd_noise_reduction=rp.FBDDNoiseReductionMode.Off,
    use_camera_wb=False,
    use_auto_wb=False,
    user_wb=[1, 1, 1, 1],
    output_color=rp.ColorSpace.raw,
    output_bps=16,
    user_sat=16383,
    user_black=0,
    no_auto_bright=True,
    no_auto_scale=True,
    gamma=(1, 1),
    user_flip=0,
    # bright=4
    )

full_args = dict(
    demosaic_algorithm=rp.DemosaicAlgorithm.DHT,
    # half_size=True,
    fbdd_noise_reduction=rp.FBDDNoiseReductionMode.Off,
    use_camera_wb=False,
    use_auto_wb=False,
    user_wb=[1, 1, 1, 1],
    output_color=rp.ColorSpace.raw,
    output_bps=16,
    user_sat=16383,
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

def PackParams(vigBool, percMinImg, percMaxImg, black, white,
               ga, gb, gg, gr, ccm, crop, comp_lo):
    out = "\n".join([
        "Vig correction:",
        str(vigBool),
        "",
        "Minimum values:",
        np.array2string(percMinImg, separator=',')[1:-1],
        "",
        "Maximum values:",
        np.array2string(percMaxImg, separator=',')[1:-1],
        "",
        "General Gamma:",
        str(ga),
        "",
        "R Gamma:",
        str(gr),
        "",
        "G Gamma:",
        str(gg),
        "",
        "B Gamma:",
        str(gb),
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


def UnpackParams(orig=False):
    if orig:
        path = "original/params.txt"
    else:
        path = "params.txt"
    with open(path, "r") as file:
        params = file.readlines()
    vigBool = False
    if params[1] == "True\n":
        vigBool = True
    if "None" not in (params[4], params[7]):
        percMinImg = np.fromstring(params[4], sep=",")
        percMaxImg = np.fromstring(params[7], sep=",")
    else:
        percMinImg = None
        percMaxImg = None
    ga = float(params[10])
    gr = float(params[13])
    gg = float(params[16])
    gb = float(params[19])
    ccm = np.fromstring(params[22], sep=",").reshape((3, 3))
    crop = params[25].split(",")
    crop = [int(dim) for dim in crop]
    black = np.fromstring(params[28], sep=",")
    white = np.fromstring(params[31], sep=",")
    comp_lo = float(params[34])

    return (vigBool, percMinImg, percMaxImg, black, white,
            ga, gb, gg, gr, ccm, crop, comp_lo)

# %% Processer

def img_process(name):
    try:
        (vigBool, percMinImg, percMaxImg, black, white,
        ga, gb, gg, gr, ccm, crop, comp_lo) = UnpackParams(orig=True)

        with rp.imread("original/" + name) as raw:
            imgp = raw.postprocess(**full_args)
            black_level = raw.black_level_per_channel[0]
        imgp = f.r2b(imgp) / 65535
        imgp = imgp - black_level / 65535
        imgp = imgp[slice(*crop[:2]),slice(*crop[2:])]
        collect()

        if vigBool:
            vigp = np.load("original/vig.npy")
            imgp = np.divide(imgp, vigp)
            del vigp
        collect()

        imgp = 1 - imgp
        for j in range(3):
            imgp[...,j] = imgp[...,j] - percMinImg[j]
            imgp[...,j] = imgp[...,j] / (percMaxImg[j] - percMinImg[j])

        imgp = (imgp + black[0]) / (1 + black[0])
        # for _c in range(1,4):
        #     imgp[...,_c-1] = (imgp[...,_c-1] + black[_c]) / (1 + black[_c])

        imgp = imgp * (1 + white[0])
        # for _c in range(1,4):
        #     imgp[...,_c-1] = imgp[...,_c-1] * (1 + white[_c])

        imgp = f.gammaBGR(imgp, ga, gb, gg, gr)
        imgp = f.CCM(imgp, ccm)
        imgp = f.compress_shadows(imgp, 0.55, comp_lo)

        imgp = np.interp(imgp, (0, 1), (0, 65535)).astype(np.uint16)
        cv2.imwrite(name[:-4] + ".png", imgp)
    except KeyboardInterrupt:
        import sys
        sys.exit()

# =============================================================================
# %% first run
# =============================================================================
def read_proxy(name):
    try:
        with rp.imread(name) as raw:
            _img = raw.postprocess(**proxy_args)
        _img = f.r2b(_img)
        return _img
    except KeyboardInterrupt:
        import sys
        sys.exit()

def main():
# if __name__ == "__main__":
    freeze_support()
    # filename = r"C:/new"
    while 1:
        filename = input("Ruta de carpeta: ")
        try:
            chdir(filename)
            flist = listdir()
            if not ("original"  in flist or
                    [True for file in flist if "cr2" in file.lower()]):
                raise FileNotFoundError
            break
        except FileNotFoundError:
            print("Carpeta no válida")

    if "original" not in flist:
        f.cmd("mkdir original")
    f.cmd("for %a in (*.CR2) do move %a original/%a")
    chdir(filename + "/original")
    flist = listdir()

    if "params.txt" in flist:
        (vigBool, percMinImg, percMaxImg, black, white,
         ga, gb, gg, gr, ccm, crop, comp_lo) = UnpackParams()
        print("Se ha encontrado archivo de configuración")
        key = input("Recolorizar? [y]/n: ").lower()
        # key = "n"
        if key == "" or key == "y":
            proxyNeeded = 2
        elif key == "n":
            proxyNeeded = 0
        else:
            while key.lower() not in ("y", "n"):
                key = input("y/n: ").lower()
                if key == "y":
                    proxyNeeded = 2
                    break
                elif key == "n":
                    proxyNeeded = 0
    else:
        proxyNeeded = 1

    imglist = []
    if "vig.CR2" in flist:
        imglist.append("vig.CR2")
        vigBool = True
    else:
        vigBool = False
        print("ATENCION: No se encontró archivo vig.CR2")
        print("Desea continuar?")
        while 1:
            key = input("[y]/n: ").lower()
            if key == "y" or key == "":
                break
            if key == "n":
                import sys
                sys.exit()

    for file in flist:
        if file[-3:] == "CR2" and file != "vig.CR2":
            imglist.append(file)
    if proxyNeeded == 2:
        if "proxy.npy" not in listdir():
            proxyNeeded = 1
    chdir(filename + "/original")

    if proxyNeeded == 1:
        print("Leyendo RAWs...")
        # try:
        with Pool(maxWorkers, initializer=init_pool) as exe:
            img = []
            for _res in exe.imap(read_proxy, imglist):
                img.append(_res)

        with rp.imread(imglist[0]) as raw:
            black_level = raw.black_level_per_channel[0]

        # with exiftool.ExifTool(r"D:\MPardo HDD\Downloads\exiftool-12.41\exiftool.exe") as et:
        #     meta = et.execute_json(*imglist)
        # img0=copy(img)

        # crop is (y1,y2,x1,x2)
        crop_needed = True

        margin=5
        crop = []
        if crop_needed:
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
                        # do something if it doesnt match, meanwhile do nothing
                        crop.append([margin, -margin])
            crop = list(np.array(crop).flatten())

        reduce_factor = 40
        img = np.array(img)
        img = img[:,slice(*crop[:2]),slice(*crop[2:])]
        crop = [dim*2 for dim in crop] #bc it was half-sized
        img = np.array([f.resize(image, reduce_factor) for image in img])
        img = img / 65535
        img = img - black_level / 65535

        if vigBool:
            if "vig.npy" not in listdir():
                print("Creando archivo de corrección de luminosidad...")
                with rp.imread(imglist[0]) as raw:
                    vig = raw.postprocess(**full_args)
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
        percMaxImg = np.empty(3)
        percMinImg = np.empty(3)
        for j in range(3):
            percMinImg[j] = np.percentile(img[..., j], 0.075)
            percMaxImg[j] = np.percentile(img[..., j], 99.8)*1.0025

        print("Normalizando...")
        for j in range(3):
            img[...,j] = img[...,j] - percMinImg[j]
            img[...,j] = img[...,j] / (percMaxImg[j] - percMinImg[j])

        print("Guardando proxies...")
        np.save("proxy.npy",img.astype(np.float32))

    if proxyNeeded == 2:
        print("Leyendo proxies...")
        img = np.load("proxy.npy")

    if proxyNeeded:
        print("Obteniendo valores de colorización...")
        ccm, black, white, ga, gb, gg, gr, comp_lo = f.ccmGamma(img)

        PackParams(vigBool, percMinImg, percMaxImg, black, white,
                   ga, gb, gg, gr, ccm, crop, comp_lo)
        # input("Apriete cualquier tecla para continuar con el proceso...")
    # =============================================================================
    # %% final pass
    # =============================================================================
    # start = time()
    print("-----------")
    print("Procesando RAWs")
    print("Esto puede tomar unos minutos...")
    print("CTRL+C para cancelar (demora unos segundos)")
    print("-----------")
    chdir(filename)

    imglist2 = [name for name in imglist if "vig" not in name.lower()]

    with Pool(maxWorkers, initializer=init_pool) as pool:
        with tqdm.trange(len(imglist2), unit="photo") as progress_bar:
            res = pool.imap_unordered(img_process, imglist2, chunksize=1)
            for __ in res:
                progress_bar.update(1)

    # for _img in imglist2:
    #     img_process(_img)

    # end = time()
    # tiempo = round(end - start)
    # print(f"Proceso terminado en {tiempo // 60} minutos {tiempo % 60} segundos")
    # input("Aprete enter para terminar")
    # collect()

    delFiles = input("Desea eliminar los archivo proxy? y/[n]: ")
    while delFiles not in ("y", "n", ""):
        delFiles = input("y/n: ").lower()
    if delFiles == "y":
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
        import sys
        sys.exit()
