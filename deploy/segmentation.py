import numpy as np
from ultralytics import YOLO

class YOLOsegmentation:
    def __init__(self, path):
        self.model = YOLO(path)

    def detect(self, img):
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        try:
            for seg in result.masks.segments:
                seg[:,0] *= width
                seg[:,1] *= height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)
        except Exception:
            pass
        
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        return bboxes, class_ids, segmentation_contours_idx, scores