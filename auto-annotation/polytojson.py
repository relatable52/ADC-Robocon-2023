import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2 as cv

def genAnno(path, id, filename):
    image = cv.imread(path)
    f = lambda x: 0 if x[0] == 0 else 1
    lst = [list(f(j) for j in i) for i in image]
    mymask = np.array(lst, dtype=np.uint8)

    # ground_truth_binary_mask = np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                      [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                      [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                      [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                      [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                      [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
    #                                      [  1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                      [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #                                      [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=np.uint8)

    ground_truth_binary_mask = mymask

    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": 1
        }
    
    image = {
        "id": id,
        "width": 1138,
        "height": 640,
        "file_name": filename
    }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)

    return annotation, image

annotations = list()
images = list()
for i in range(1, 301):
    order = "{:04d}".format(i)
    path = r"label\robocon_"+str(order)+".jpg"
    id = i
    filename = "robocon_"+str(order)+".jpg"
    anno, image = genAnno(path, id, filename)
    annotations.append(anno)
    images.append(image)
    print(order)

output = {
    "images": images,
    "annotations": annotations
}

with open("annotations\sample.json", "w") as outfile:
    json.dump(output, outfile)
    