import json




def evaluate_roi(detection, roi):
    percentage_overlap = 0
    for *bbox, conf, cls in reversed(detection.tolist()):
        det_coord = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        objects_of_interest = list(dict(json.load(open("object_of_interest.json", "r"))).values())
        if cls in objects_of_interest:
            area = intersection_area(det_coord, roi)
            box_area = (det_coord[2] - det_coord[0]) * (det_coord[3] - det_coord[1])
            if(box_area > 0):
                percentage_overlap = area / box_area

    return percentage_overlap


def intersection_area(box, roi):
    area = 0
    dx = min(box[2], roi[2]) - max(box[0], roi[0])
    dy = min(box[3], roi[3]) - max(box[1], roi[1])
    if (dx>=0) and (dy>=0):
        area = dx * dy
    return area

