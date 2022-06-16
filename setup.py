# -*- coding: utf-8 -*-
"""
Created on Thu May 26 23:16:46 2022

@author: MPardo
"""

import os
import sys
from configparser import ConfigParser
from time import sleep


def close():
    print()
    print("--------")
    print("Cerrando programa en 2 segundos...")
    sleep(2)
    sys.exit()

def main():
    print("FilmProcesser Setup")
    print("Presione Ctrl+C para cancelar")
    print("-------------------")

    if "setup.ini" in os.listdir():
        print()
        print("Ya existe un archivo setup.ini")
        while 1:
            key = input("Desea reconfigurar? [Y]/n: ").lower()
            if key == "n":
                close()
            elif key in ("", "y"):
                break

    config = ConfigParser()
    print()
    print("Especifique cantidad de procesos simultáneos:")
    print("Dejar en blanco para configuración automática (recomendado)")
    print("Se puede usar una cantidad menor a la cantidad de procesadores disponible\nsi  el consumo de RAM es muy alto.")
    print("No tiene mucho sentido especificar más procesos que procesadores.")
    while 1:
        max_processes = input("Ingrese cantidad: ")
        if max_processes.isdigit():
            if int(max_processes) > 0:
                break
        elif max_processes == "":
            max_processes = "None"
            break

    print()
    print("Especifique reducción de tamaño para previsualizaciones:")
    print("Dejar en blanco para configuración automática (720p)")
    print("720px de alto funciona bien para archivos en una pantalla Full HD.")
    print("Mayor tamaño incrementa la demora del anális previo")
    print("Tamaño mínimo es 400px")
    while 1:
        reduce_height = input("Ingrese cantidad: ")
        if reduce_height.isdigit():
            if int(reduce_height) >= 400:
                break
        elif reduce_height == "":
            reduce_height = "720"
            break

    print()
    print("Especifique método de interpolación RAW")
    print("1: DHT (Recomendado, máxima calidad, demora más)")
    print("2: AHD (Velocidad media, calidad media")
    print("3: Linear (Más rápido, baja calidad)")
    while 1:
        interp = input("Ingrese opción: ")
        if interp == "1":
            interp = "DHT"
            break
        if interp == "2":
            interp = "AHD"
            break
        if interp == "3":
            interp = "LINEAR"
            break

    print()
    print("Desea recortar las imágenes al procesarlas?")
    print("1: Siempre intentar recorte automático")
    print("2: No recortar")
    while 1:
        crop = input("Ingrese opción: ")
        if crop == "1":
            crop = "True"
            break
        if crop == "2":
            crop = "False"
            break

    print()
    print("Desea usar corrección de luminosidad?")
    print("1: Siempre si está disponible")
    print("2: No usar ni pedir")
    while 1:
        vig = input("Ingrese opción: ")
        if vig == "1":
            vig = "True"
            break
        if vig == "2":
            vig = "False"
            break

    print()
    print("Especifique cantidad de bits en la imagen final")
    print("1: 16 bits (máximo tamaño, máxima calidad)")
    print("2: 14 bits (12,5% de ahorro de espacio)")
    print("3: 12 bits (25,0% de ahorro de espacio)")
    print("4: 10 bits (37,5% de ahorro de espacio)")
    print("5: 8 bits (50% de ahorro de espacio, mínima calidad)")
    while 1:
        bit_depth = input("Ingrese opción: ")
        if bit_depth == "1":
            bit_depth = "16"
            break
        if bit_depth == "2":
            bit_depth = "14"
            break
        if bit_depth == "3":
            bit_depth = "12"
            break
        if bit_depth == "4":
            bit_depth = "10"
            break
        if bit_depth == "5":
            bit_depth = "8"
            break

    config["SYSTEM"] = {
        # None = Auto
        "Process count" : max_processes,
        # In the future, internal float32/16 can be used instead of float64 to further decrease memory usage
        }

    config["IMAGE PROCESSING"] = {
        "Cropping" : crop,
        "Luminosity correction" : vig,
        "Interpolation method": interp,
        # "Previsualization reduce factor": reduce_factor,
        "Previsualization reduce height": reduce_height,
        # Possibly add more rawpy settings like half-size interpolation
        }

    config["IMAGE OUTPUT"] = {
        "Bit depth" : bit_depth,
        # "Compression" : "None",
        }

    with open('setup.ini', 'w') as configfile:
        config.write(configfile)

    print("Finalizado correctamente")
    close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        close()
