import numpy as np

from multiprocessing import Pool

from lyft_dataset_sdk.eval.detection.mAP_evaluation import group_by_key, recall_precision


def get_average_precisions(gt: list,
                           predictions: list,
                           class_names: list,
                           iou_threshold: float) -> np.array:
    assert 0 <= iou_threshold <= 1

    gt_by_class_name = group_by_key(gt, "name")
    pred_by_class_name = group_by_key(predictions, "name")

    pool = Pool(8)
    pool_inputs = []
    for name in class_names:
        if name in gt_by_class_name and name in pred_by_class_name:
            pool_inputs.append((gt_by_class_name[name],
                                pred_by_class_name[name],
                                iou_threshold,
                                name))
    pool_results = pool.starmap(_recall_precision, pool_inputs)

    average_precisions = np.zeros(len(class_names))
    for class_name, average_precision in pool_results:
        average_precisions[class_names.index(class_name)] = average_precision

    pool.close()

    return average_precisions


def _recall_precision(gt, pred, th, class_name):
    recalls, precisions, average_precision = recall_precision(gt, pred, th)
    return class_name, average_precision
