'''
Manuel Framil de Amorin

Entradas: Imaxe e labels -> Deben ter o mesmo nome (solo cambia a extension)
							e estar na mesma carpeta
		  Tamen se pode pasar directamente unha carpeta onde esten as
		  imaxes e as suas labels
Salidas:  Na carpeta especificada das imaxes crea unha subcarpeta
		  chamada 'smoke_labels' e garda as coordenadas e tamaño dos fumes (Sen normalizar).
		  Formato: X Y Size


O fume colocase nunha posicion aleatoria do lado pequeno
do rectangulo que encuadra o coche.
O tamaño do fume se calcula como un numero aleatorio entre
X/4 e X/2, sendo X a lonxitude do lado pequeno do rectangulo
'''
import sys
import numpy as np
import cv2
import csv
from random import randint
import os


def process_file(path, img_name, out_subdir='smoke_labels', label_format='.txt'):
    filepath = os.path.join(path, img_name)
    labels_name = filepath.split('.')[0] + label_format

    out_path = os.path.join(path, out_subdir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    output = os.path.join(out_path, "s" + img_name.split('.')[0] + label_format)

    img = cv2.imread(filepath)
    altura, anchura, canales = img.shape

    fout = open(output, 'w+')

    with open(labels_name) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            # Coordenadas punto top-left
            pos_x, pos_y, size = generate_smoke_coords(altura, anchura, row[1:])

            # fout.write(f'{pos_x/anchura} {pos_y/altura} {size/anchura} {size/altura}\n')
            fout.write(f'{pos_x} {pos_y} {size} \n')

    fout.close()


def generate_smoke_coords(altura, anchura, label):
    x_tl = int(float(label[0]) * anchura)
    y_tl = int(float(label[1]) * altura)

    # Coordenadas punto bot-right
    x_br = x_tl + int(float(label[2]) * anchura)
    y_br = y_tl + int(float(label[3]) * altura)

    # Coordenadas punto bot-left
    x_bl = x_tl
    y_bl = y_br

    # Coordenadas punto top-right
    x_tr = x_br
    y_tr = y_tl

    rect_anch = x_br - x_bl
    rect_alt = y_bl - y_tl

    pos_x = 0
    pos_y = 0
    size = 0

    if (rect_anch < rect_alt):
        pos_x = randint(x_bl, int(x_br - rect_anch / 2))
        pos_y = y_bl - int(rect_anch / 2)
        size = randint(int(rect_anch / 4), int(rect_anch / 2))
    else:
        pos_x = x_br - int(rect_alt / 2)
        pos_y = randint(y_tl, int(y_bl - rect_alt / 2))
        size = randint(int(rect_alt / 4), int(rect_alt / 2))

    return (pos_x, pos_y, size)


def main(argv):
    if os.path.isfile(argv[0]):

        path_list = argv[0].split(os.sep)
        path = '.'
        for p in path_list[:-1]:
            path = os.path.join(path, p)

        process_file(path, path_list[-1])

    if os.path.isdir(argv[0]):
        for subdir, dirs, files in os.walk(argv[0]):
            for filename in files:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    process_file(subdir, filename)


if __name__ == "__main__":
    main(sys.argv[1:])