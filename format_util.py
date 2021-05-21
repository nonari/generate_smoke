import os
import cv2
import ntpath
from typing import List, Tuple

yroi_type = Tuple[float, float, float, float]
rect_type = Tuple[int, int, int, int]


def check_file(path: str) -> None:
    """Check if file exists and return an exception if not

    Args:
        path (str): Path of file
    Raises:
        FileNotFoundError

    """
    if not os.path.isfile(path):
        raise FileNotFoundError


def check_file_bool(path: str) -> bool:
    """Check if file exists and return a boolean

    Args:
        path (str): Path of file
    Return:
        Bool

    """
    if not os.path.isfile(path):
        return False
    else:
        return True


def format_dir(path: str, create=False) -> str:
    """Checks if path is a directory and returns it with a separator at the end

    Args:
        path (str): Path of file
        create (bool): Create directory if it does not exist
    Returns:
        path (str): Normalized path
    Raises:
        NotADirectoryError

    """

    normalized = os.path.normpath(path) + os.path.sep

    if os.path.exists(path):
        if not os.path.isdir(path):
            raise NotADirectoryError
        return normalized
    else:
        if create:
            os.makedirs(path)
            return normalized
        else:
            raise NotADirectoryError(path)


def create_file(path, override=False):
    if override or not os.path.exists(path):
        file = open(path, "w")
        file.close()


def change_extension(path: str, ext: str) -> str:
    """Changes the extension of the file

    Args:
        path (str): Path of file
        ext (str): New extension

    """

    return path.split('.')[0] + '.' + ext


def has_img_format(path: str) -> bool:
    """Check if file has an image extension

    Args:
        path (str): Path of file
    Returns:
        bool: true if file is an image

    """
    return path.endswith('jpg') or path.endswith('jpeg') or path.endswith('svg') or path.endswith('png')


def override_file(txt_file: str, rois: List[yroi_type]) -> None:
    """Overrides file content

    Args:
        txt_file (str): Path of .txt file
        rois (list): List of rois to write

    """
    txt = open(txt_file, 'w')
    for roi in rois:
        x1_r, y1_r, w_r, h_r = roi
        txt.write('0 {:06.4f} {:06.4f} {:06.4f} {:06.4f}\n'.format(x1_r, y1_r, w_r, h_r))

    txt.close()


def get_rois(txt_file: str) -> List[yroi_type]:
    """Get list of ROIs from YOLO .txt file

    Args:
        txt_file (str): Path to .txt file
    Returns:
        list[tuple]: list of YOLO ROIs

    """
    txt = open(txt_file, 'r')
    rois = []
    for line in txt:
        rois.append(split_yolo_roi(line))
    txt.close()

    return rois


def draw_rectangles(img, rois: List[yroi_type], thickness=1):
    """Draw ROIs over the image

    Args:
        img: cv2 Mat
        rois (list[tuple]): List of YOLO ROIs
        thickness (int): Thickness of the region delimiter
    Returns:
        img: cv2 Mat with ROIs drawn

    """
    dh, dw, _ = img.shape

    for roi in rois:
        x1, y1, x2, y2 = yolo_roi_to_corners(roi, dw, dh)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=thickness)

    return img


def yolo_roi_to_corners(roi: yroi_type, frame_w, frame_h) -> rect_type:
    x_r, y_r, w_r, h_r = roi

    x1 = int((x_r - w_r / 2) * frame_w)
    x2 = int((x_r + w_r / 2) * frame_w)
    y1 = int((y_r - h_r / 2) * frame_h)
    y2 = int((y_r + h_r / 2) * frame_h)

    if x1 < 0:
        x1 = 0
    if x2 > frame_w - 1:
        x2 = frame_w - 1
    if y1 < 0:
        y1 = 0
    if y2 > frame_h - 1:
        y2 = frame_h - 1

    return x1, y1, x2, y2


def split_yolo_roi(roi: str) -> yroi_type:
    """
    Parses roi from YOLO .txt file

    Args:
        roi (str): Space separated roi from .txt roi file
    Returns:
        (tuple): tuple containing
            (float): center x ratio
            (float): center y ratio
            (float): width x ratio
            (float): height x ratio

    """
    roi_parts = roi.split(' ')
    return float(roi_parts[1]), float(roi_parts[2]), float(roi_parts[3]), float(roi_parts[4])


def get_rect_from_yroi(roi: yroi_type, frame_w, frame_h) -> rect_type:
    """Convert YOLO roi to rect

    Args:
        roi (str): Space separated line from .txt YOLO roi file
        frame_h (int): target image height
        frame_w (int): target image width
    Returns:
        (tuple): tuple containing:
            x (int): Left top corner x coord of rect
            y (int): Left top corner y coord of rect
            width (int): Width of rect
            height (int): Height of rect

    """
    x_r, y_r, w_r, h_r = roi
    width = int(w_r * frame_w)
    height = int(h_r * frame_h)
    x = int(x_r * frame_w - width / 2)
    y = int(y_r * frame_h - height / 2)

    return x, y, width, height


def remove_unlabeled(img_dir: str):
    """Removes images without labels

        Will remove every image whose rois .txt is empty,
        images without .txt will remain in dir.

    Args:
        img_dir (str): Path of directory to delete unlabeled images

    """
    img_files = get_images(img_dir)
    for img_name in img_files:
        img_path = img_dir + img_name
        txt_path = change_extension(img_path, "txt")
        if os.path.exists(txt_path):
            if os.path.getsize(txt_path) < 10:
                os.remove(img_path)
                os.remove(txt_path)


def count_number_of_lines(path) -> int:
    """Counts number of lines in file

    Args:
        path (str):  Path of file
    Return:
        int: Number of lines

    """
    file = open(path, "r")

    i = 0
    for _ in file:
        i += 1
    file.close()

    return i


def corner_points_to_yroi(corner_points: yroi_type) -> yroi_type:
    """Convert roi of two corner points to rect

    Args:
        corner_points (tuple[int, int, int, int]): Roi with corner points format
    Returns:
        (tuple): tuple containing:
            center_x (int): Center x coord of rect
            center_y (int): Center y coord of rect
            width (int): Width of r
            height (int): Height of roi

    """
    x_top, y_top, x_bot, y_bot = corner_points
    w_r = x_bot - x_top
    h_r = y_bot - y_top
    center_x_r = x_top + w_r / 2
    center_y_r = y_top + h_r / 2

    return center_x_r, center_y_r, w_r, h_r


def corners_abs_to_yroi(rect: rect_type, frame_w: int, frame_h: int) -> yroi_type:
    """Convert roi of two corner points to YOLO format

        Args:
            rect (tuple[int, int, int, int]): rect
            frame_w (int): Image width
            frame_h (int): Image height
        Returns:
            (tuple): tuple containing:
                x_r (int): Center x coord ratio of roi
                y_r (int): Center y coord ratio of roi
                w_r (int): Width ratio of roi
                h_r (int): Height ratio of roi
    """
    x1, y1, x2, y2 = rect

    w = x2 - x1
    h = y2 - y1
    center_x = x1 + w / 2
    center_y = y1 + h / 2
    x_r = center_x / frame_w
    y_r = center_y / frame_h
    w_r = w / frame_w
    h_r = h / frame_h

    return x_r, y_r, w_r, h_r


def irois_to_corners(yrois: List[yroi_type], frame_w: int, frame_h: int) -> List[rect_type]:
    corners = []
    for roi in yrois:
        corners.append(yolo_roi_to_corners(roi, frame_w, frame_h))

    return corners


def write_corners(rect: rect_type, frame_w: int, frame_h: int, txt_file: str) -> None:
    """Appends corners roi in yolo format to .txt file

    Args:
        rect (tuple): ROI
        frame_w (int): Image width
        frame_h (int): Image height
        txt_file (str): File to write

    """
    x1_r, y1_r, w_r, h_r = corners_abs_to_yroi(rect, frame_w, frame_h)
    with open(txt_file, 'a') as log:
        log.write('0 {:06.4f} {:06.4f} {:06.4f} {:06.4f}\n'.format(x1_r, y1_r, w_r, h_r))
        log.close()


def write_roi(roi: yroi_type, txt_file: str) -> None:
    """Appends rect in yolo format to .txt file

    Args:
        roi (tuple): yolo ROI
        txt_file (str): File to write

    """
    x1_r, y1_r, w_r, h_r = roi
    with open(txt_file, 'a') as log:
        log.write('0 {:06.4f} {:06.4f} {:06.4f} {:06.4f}\n'.format(x1_r, y1_r, w_r, h_r))
        log.close()


def get_images(img_dir: str) -> List[str]:
    """Gets an image filename list from directory

    Args:
        img_dir (str): Images directory path
    Returns:
        (list[str]): List of image filenames

    """
    images = []
    if os.path.isfile(img_dir):
        images.append(img_dir)
    elif os.path.isdir(img_dir):
        images = os.listdir(img_dir)
        iterator = filter(lambda file: has_img_format(file), images)
        images = list(iterator)
    else:
        raise NotADirectoryError

    return images


def calc_iou(roi1, roi2):
    """
    Function to calculate the percentage of agreement between detections using the IoU metric.

    Args:
        roi1: Region Of Interest with corners coordinates.
        roi2: Region Of Interest with corners coordinates.
    Returns:
        Percentage of agreement between detections

    """
    x_left = max(roi1[0], roi2[0])
    y_top = max(roi1[1], roi2[1])
    x_right = min(roi1[2], roi2[2])
    y_bottom = min(roi1[3], roi2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    roi1_area = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
    roi2_area = (roi2[2] - roi2[0]) * (roi2[3] - roi2[1])

    iou = intersection_area / float(roi1_area + roi2_area - intersection_area)
    return iou


def calc_giou(roi1, roi2):
    """
    Function to calculate the percentage of agreement between detections using the GIoU metric.

    Args:
        roi1: Region Of Interest with corners coordinates.
        roi2: Region Of Interest with corners coordinates.
    Returns:
        Percentage of agreement between detections

    """
    x_left = max(roi1[0], roi2[0])
    y_top = max(roi1[1], roi2[1])
    x_right = min(roi1[2], roi2[2])
    y_bottom = min(roi1[3], roi2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    roi1_area = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
    roi2_area = (roi2[2] - roi2[0]) * (roi2[3] - roi2[1])

    union = float(roi1_area + roi2_area - intersection_area)

    iou = intersection_area / union

    xP1 = min(roi1[0], roi1[2])
    xP2 = max(roi1[0], roi1[2])
    yP1 = min(roi1[1], roi1[3])
    yP2 = max(roi1[1], roi1[3])

    xC1 = min(xP1, roi2[0])
    xC2 = max(xP2, roi2[2])
    yC1 = min(yP1, roi2[1])
    yC2 = max(yP2, roi2[3])

    area_C = (xC2 - xC1) * (yC2 - yC1)

    giou = iou - ((area_C - union) / area_C)

    return giou


def get_extension(filepath: str) -> str:
    """
    Get file extension from a filepath

    Args:
        filepath: Path of file
    Returns:
        Extension of the file

    """
    filename = get_filename(filepath)
    last_idx = filename.rfind('.')
    if last_idx < 0:
        return ""

    return filename[last_idx:]


def get_filename(filepath: str) -> str:
    """
    Get file name from a filepath

    Args:
        filepath: Path of file
    Returns:
        Name of the file
    Raises:
        IsADirectoryError

    """
    head, tail = ntpath.split(filepath)
    if tail == '':
        raise IsADirectoryError
    return tail


def get_name(filepath: str) -> str:
    """
    Get name of file without extension from a file path

    Args:
        filepath: Path of file
    Returns:
        Name of the file
    Raises:
        IsADirectoryError
    """
    head, tail = ntpath.split(filepath)
    if tail == '':
        raise IsADirectoryError
    point = tail.rfind('.')
    end = 0 if point == -1 else point

    return filepath[:end]


def matches(orig_rois: List[rect_type], found_rois: List[rect_type]):
    orig_rois_rem = orig_rois.copy()
    found_rois_rem = found_rois.copy()
    for orig_roi in orig_rois:
        for found_roi in found_rois:
            if calc_iou(orig_roi, found_roi) > 0.4:
                orig_rois_rem.remove(orig_roi)
                found_rois_rem.remove(found_roi)
                break

    return orig_rois_rem, found_rois_rem


def txt_of_file_dir(path: str, format: str) -> str:
    """Changes the extension of the file by any format

    Args:
        path (str): Path of file
        format (str): Format required for the exit

    """
    texts = path.split('/')
    file = texts[len(texts) - 1]
    return file.split('.')[0] + format


def split_folder(path: str) -> str:
    """Get only the file from the path

        Args:
            path (str): Path of file
        Return:
            file (str): File

        """
    texts = path.split('/')
    file = texts[len(texts) - 1]
    return file


def get_classes(classes_path: str):
    """ Function for get the classes from a txt file and insert them into an array

    Args:
        classes_path (str): Path of the file
    Return:
        classes: Array with the classes
    """

    classes = []
    classes_file = open(classes_path, "r")
    for line in classes_file:
        classid = line.split("\n")
        classes.append(classid[0])
    return classes


def get_value(value_file: str):
    """ Function for get the values from a txt file and insert them into an array

        Args:
            value_file (str): Path of the file
        Return:
            value: Array with the values
        """
    value = []
    lines = open(value_file)
    for line in lines:
        cell = eval(line)
        value.append(cell)
    return value


def has_class(name: str, classes):
    """
    Indicate if a name exists in the array of classes

    Args:
        name: The name of the class to compare.
        classes: THe array with the classes.
    Return:
        bool: True if contains, False if not.
    """
    for class_name in classes:
        if class_name.lower() == name or class_name == name:
            return True
    return False


def id_class(name: str, classes):
    """
        Indicate if a name exists in the array of classes and gives the id of the class.

        Args:
            name: The name of the class to compare.
            classes: THe array with the classes.
        Return:
            int: -1 if not contains, else the id of the class.
        """
    cont = 0
    for class_name in classes:
        if class_name.lower() == name or class_name == name:
            return cont
        cont = cont + 1
    return -1
