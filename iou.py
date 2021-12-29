import numpy as np

from utils import get_data, check_results


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    x1,x2,x3,x4 = gt_bbox[0] , gt_bbox[2] , pred_bbox[0] , pred_bbox[2]
    y1,y2,y3,y4 = gt_bbox[1] , gt_bbox[3] , pred_bbox[1] , pred_bbox[3]
    ## IMPLEMENT THIS FUNCTION
    if (x3 >= x2 or x4 <= x1) or (y3 >= y2 or y4 <= y1):
        return 0

    x = [x1,x2,x3,x4]
    y = [y1,y2,y3,y4]
    x.sort()
    y.sort()
    l_intsec , b_intsec = abs(x[1]-x[2]) , abs(y[1]-y[2])
    
    intersection = abs(l_intsec * b_intsec)

    union = abs((x1 - x2) * (y1 - y2)) + abs((x3 - x4) * (y3 - y4)) - intersection
    iou = intersection / union

    return iou


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    check_results(ious)
