from put_smoke_on_cars import generate_smoke_coords
import format_util
import cv2

dir_oid = "/home/alex/smoke_gen/OIDv4/"
image_filenames = format_util.get_images(dir_oid)

for filename in image_filenames:
    img = cv2.imread(dir_oid + filename)
    rois = format_util.get_rois(format_util.change_extension(dir_oid + filename, "txt"))
    h, w = img.shape

    for roi in rois:
        smoke_coords = generate_smoke_coords(h, w, roi)
