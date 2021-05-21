'''
Manuel Framil de Amorín

Script to crop the smoky-regions from the original image
'''
import sys
import os
import cv2
import numpy as np

resizeDim = (32, 32)
trimShape = (16, 16)


def get_rect_from_roi(roi, frame_w, frame_h):
    x_r, y_r, w_r, h_r = roi
    width = int(w_r * frame_w)
    height = int(h_r * frame_h)
    x1 = int(x_r * frame_w - width / 2)
    y1 = int(y_r * frame_h - height / 2)

    x2 = x1 + width
    y2 = y1 + height

    return x1, y1, x2, y2


def split_roi(roi_line):
    parts = roi_line.split(' ')
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def create_dataset(files, target, output_dir):
    # Read all labels from file

    target_file = open(target, 'w+')

    count = 0

    x_list = []
    y_list = []

    removed = 0

    for f in files:

        roi_file = f.split('.')[0] + '.txt'

        img = cv2.imread(f)
        frame_h, frame_w, _ = img.shape
        rois = []

        with open(roi_file, 'r') as roi:
            rois = roi.readlines()

        for roi in rois:
            roi_splitted = split_roi(roi)

            rect = get_rect_from_roi(roi_splitted[1:], frame_w, frame_h)
            xmin, ymin, xmax, ymax = rect

            w = xmax - xmin
            h = ymax - ymin

            newXmin = int(xmin - w / 4)
            newYmin = int(ymax - h / 4)
            newXmax = int(xmax + w / 4)
            newYmax = int(ymax + h)

            crop = img[newYmin:newYmax, newXmin:newXmax]

            x_list.append(crop.shape[0])
            y_list.append(crop.shape[1])

            if not all(crop.shape):
                continue

            if crop.shape[0] < trimShape[0] or crop.shape[1] < trimShape[1]:
                removed += 1
                continue

            resized = cv2.resize(crop, resizeDim, interpolation=cv2.INTER_LINEAR)

            outfile = os.path.join(output_dir, str(count) + '.png')

            cv2.imwrite(outfile, resized)
            target_file.write(f'{roi_splitted[0]}\n')

            count += 1

    print('--------------------- Metrics -------------------------')
    print(f' Mean shape {np.mean(x_list):.2f} x {np.mean(y_list):.2f}')
    print(f' Max height: {np.max(x_list)}, Min height: {np.min(x_list)}')
    print(f' Max width: {np.max(y_list)}, Min width: {np.min(y_list)}')
    print(f' Sdt shape: {np.std(x_list):.2f} x {np.std(y_list):.2f}')
    print()
    print(f' Trim shape: {trimShape}')
    print(f' Total removed rois: {removed}')
    print(f' % of removed rois: {(removed / len(x_list)) * 100:.2f}%')
    print()
    print(f' Dataset remaining')
    print(f' Output shape {resizeDim}')
    print(f' Number of images: {count + 1}')
    print('--------------------------------------------------------')


def main(argv):
    print()
    print('--------------------- Usage --------------------------')
    print()
    print('# python crop_smoke.py IMG_FOLDER LABELS_OUTPUT NEW_IMG_OUTPUT')
    print()

    files = []
    path = argv[0]
    target = argv[1]
    output_dir = argv[2]

    if path.endswith(".jpg") or path.endswith(".png"):
        files.append(path)
    else:
        for filename in sorted(os.listdir(path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                files.append(os.path.join(path, filename))

    create_dataset(files, target, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
