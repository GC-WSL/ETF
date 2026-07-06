import cv2
import joblib
import os
from matplotlib import cm
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
from prettytable import PrettyTable


def postprocessor(img, probs, t=12, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=1 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83 / scale_factor, srgb=1, rgbim=np.copy(img_c), compat=4)

    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def apply_heatmap(img, cam, alpha=0.5, normal=False):
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) if normal else cam
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

def colorful(img,num_cls):
    color_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    if num_cls == 5:
        palette = np.array([[255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])
    elif num_cls == 6:
        palette = np.array([[0,0,0],[255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]])
    else :
        palette = np.array([[0, 0, 0], 
                 [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
                 [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
                 [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
                 [0, 127, 191], [0, 127, 255], [0, 100, 155]])
    keys = np.unique(img)
    for c in keys:
        color_img[img == c] = palette[c]
    return Image.fromarray(color_img)



def metricIoU(label, pred_label, num_classes: int, ignore_index: int = 255):
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true != ignore_index)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=n_class**2
        ).reshape(n_class, n_class)
        return hist

    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label, pred_label):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    
    acc = np.diag(hist).sum() / hist.sum()
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    miou = np.nanmean(iou)
    
    class_iou = {i: iou[i] for i in range(num_classes)}
    return {
        "aAcc": acc,
        "Mean IoU": miou,
        "Class IoU": class_iou
    }

def class2str(metric: dict,num_cls:int):
    table = PrettyTable()
    table.field_names = ["Class", "IoU"]

    if num_cls == 5:
        categories = [
            'impervious surfaces', 'building', 'low vegetation', 'tree', 'car'
        ]
    elif num_cls == 6:
        categories = [
            'BG','impervious surfaces', 'building', 'low vegetation', 'tree', 'car'
        ]
    else :
        categories = ('background', 'ship', 'store tank', 'baseball diamond',
                 'tennis court', 'basketball court', 'Ground Track Field',
                 'Bridge', 'Large Vehicle', 'Small Vehicle', 'Helicopter',
                 'Swimming pool', 'Roundabout', 'Soccer ball field', 'plane',
                 'Harbor')
        
    for cls, iou in metric["Class IoU"].items():
        table.add_row([categories[cls], f"{iou*100:.2f}%"])
    table.add_row(["mIoU", f"{metric['Mean IoU']*100:.2f}%"])
    return str(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--predict-dir", required=True, help="Prediction files directory")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--num-cls", type=int, default=5)
    parser.add_argument("--type", choices=["npy", "png"], default="npy")
    parser.add_argument("--reduce-zero", action="store_true")
    parser.add_argument("--save-heatmap", action="store_true")
    parser.add_argument("--heatmap-dir")
    parser.add_argument("--save-color", action="store_true")
    parser.add_argument("--color-dir")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--mask-dir")
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--split",default='train',choices=["val","test", "train"],type=str)
    parser.add_argument("--sub",default=-1,type=int)
    parser.add_argument("--save-gt", action="store_true")
    args = parser.parse_args()


    split = args.split
    args.img_path = Path(args.root)/f"img_dir/{split}"
    args.gt_folder = Path(args.root)/f"ann_dir/{split}"
    args.list = Path(args.root)/f"{split}.txt"
    

    if args.save_heatmap and not args.heatmap_dir:
        raise ValueError("The --heatmap-dir parameter must be specified.")
    if args.save_color and not args.color_dir:
        raise ValueError("The --color-dir parameter must be specified")
    if args.save_mask and not args.mask_dir:
        raise ValueError("The --mask-dir parameter needs to be specified.")
    

    for d in [args.heatmap_dir, args.color_dir, args.mask_dir]:
        if d: Path(d).mkdir(parents=True, exist_ok=True)


    with open(args.list) as f:
        name_list = [line.strip().split()[0] for line in f]
        if args.sub!=-1:
            name_list=name_list[:args.sub]

    def process_item(i):
        if args.type == "npy":
            data = np.load(Path(args.predict_dir)/f"{name_list[i]}.npy", allow_pickle=True).item()
            keys = data["keys"]
            prob = data["prob"]
            
            img = np.array(Image.open(args.img_path/f"{name_list[i]}.png"))
            if args.crf:
                prob = postprocessor(img, prob, labels=prob.shape[0])
            pred = keys[np.argmax(prob, 0)].astype(np.uint8)
        else:
            pred = np.array(Image.open(Path(args.predict_dir)/f"{name_list[i]}.png"))

        if args.save_color:
            colorful(pred,args.num_cls).save(Path(args.color_dir)/f"{name_list[i]}_pred.png")
        if args.save_mask:
            Image.fromarray(pred).save(Path(args.mask_dir)/f"{name_list[i]}.png")
        if args.save_heatmap and args.type == "npy":
            img = cv2.cvtColor(cv2.imread(str(args.img_path/f"{name_list[i]}.png")), cv2.COLOR_BGR2RGB)
            for idx,cls in enumerate(keys):
                cv2.imwrite(str(Path(args.heatmap_dir)/f"{name_list[i]}_c{cls}.png"),
                          cv2.cvtColor(apply_heatmap(img, prob[idx]), cv2.COLOR_RGB2BGR))

        gt = np.array(Image.open(args.gt_folder/f"{name_list[i]}.png"))
        if args.save_gt:
            colorful(gt,args.num_cls+1).save(Path(args.color_dir)/f"{name_list[i]}_gt.png")
        if args.reduce_zero:
            gt[gt == 0] = 255
            gt -= 1
            gt[gt == 254] = 255
        return pred, gt

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10,pre_dispatch="all")(
        joblib.delayed(process_item)(i) for i in range(len(name_list))
    )

    preds, gts = zip(*results)
    metrics = metricIoU(gts, preds, args.num_cls)
    print(metrics)
    print(class2str(metrics,args.num_cls))