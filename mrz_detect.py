"""
python detect.py --weights ai_models/best.pt --conf 0.25 --img-size 640 --source data/training


"""
import os
import time
import glob
import csv
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import  letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box

DEBUG = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def load_model(opt_weights, opt_img_size=640):
    model = attempt_load(opt_weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt_img_size, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if device!= 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return model




def inference(model, image, output_dir="logs", img_size=640):
    augment, conf_thres, iou_thres, line_thickness = False, 0.25, 0.1, 2
    stride = int(model.stride.max())
    org_img = image.copy()
    im0 = org_img
    img = letterbox(image, img_size, stride=stride, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Apply NMS
    pred = non_max_suppression(
        pred, 
        conf_thres, 
        iou_thres, 
        classes=None, 
        agnostic=False
        )
    # Process detections
    predicted_box = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                box = [int(i) for i in xyxy]
                # print(box)
                predicted_box.append(box)
                # print([xyxy[0], xyxy[1],  xyxy[2],  xyxy[3]])
                if DEBUG:
                    plot_one_box(xyxy, im0, label=label, \
                                 color=(0, 255, 0), line_thickness=1)
    if DEBUG:
        cv2.imwrite(output_dir, im0)
    return predicted_box
    # print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':


    # opt_weights = 'ai_models/epoch_298.pt'
    opt_weights = './ai_models/mrz_detection_best.pt'
    opt_img_dir = './data'
    output_dir = 'logs'

    os.makedirs(output_dir, exist_ok=True)
    print("model loading ..........")
    model = load_model(opt_weights)
    print("Done")

    files = glob.glob(opt_img_dir+"/*.png")

    report_data = []
    for _file in files:
        file_name = os.path.basename(_file)
        print(f" File {file_name} : ", end="", flush=True)
        image = cv2.imread(_file)

        boxes = inference(
            model, 
            image, 
            output_dir= os.path.join(output_dir, file_name)
        )
        print(boxes)
        print("Done")

        report_data.append([file_name, 0, 0, len(boxes), 0, 0, 0])


    # open the file in the write mode
    Header = ["Image_Name", "Have MRZ", "Actual Predict", "MRZ_Detected", "MRZ_Partially_Detected", "MRZ_Missing", "MRZ_Extra_Detected"]
    with open('report.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(Header)
        writer.writerows(report_data)

    
