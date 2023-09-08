import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2 as cv

def genAnno(path, id, filename):
    image = cv.imread(path)
    image = cv.imread(path)
    image = cv.Canny(image, 100, 200)
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    group = list()
    image = {
            "id": id,
            "width": 1138,
            "height": 640,
            "file_name": filename
        }
    cnt = 1
    for i in contours:
        poly = list()
        for j in i:
            poly.extend([float(j[0][0]), float(j[0][1])])
        area = cv.contourArea(i)
        bbox = list(cv.boundingRect(i))
        annotation = {
            "segmentation": [poly],
            "area": area,
            "iscrowd": 0,
            "image_id": id,
            "bbox": bbox,
            "category_id": 1,
            "id": cnt
        }
        if(area > 100):
            cnt += 1
            group.append(annotation)

    return group, image

#     annotation = {
#             "segmentation": [],
#             "area": ground_truth_area.tolist(),
#             "iscrowd": 0,
#             "image_id": id,
#             "bbox": ground_truth_bounding_box.tolist(),
#             "category_id": 1,
#             "id": 1
#         }
    
#     image = {
#         "id": id,
#         "width": 1138,
#         "height": 640,
#         "file_name": filename
#     }

#     for contour in contours:
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         annotation["segmentation"].append(segmentation)

#     return annotation, image

annotations = list()
images = list()
for i in range(1, 301):
    order = "{:04d}".format(i)
    path = r"label\robocon_"+str(order)+".jpg"
    id = i
    filename = "robocon_"+str(order)+".jpg"
    anno, image = genAnno(path, id, filename)
    annotations.extend(anno)
    images.append(image)
    print(order)

output = {
    "images": images,
    "annotations": annotations
}

with open("annotations\sample.json", "w") as outfile:
    json.dump(output, outfile)
    