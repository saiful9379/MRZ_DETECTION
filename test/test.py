"""
python detect.py --weights ai_models/best.pt --conf 0.25 --img-size 640 --source data/training


"""
import os
import time
import glob
# import csv
import cv2

import sys
# sys.path.insert(0, './mrz_detection_api')
from mrz_detect import load_model, inference

def start_processing(model, image, output_dir="logs"):

    box = inference(
        model,
        image, 
        output_dir=output_dir
        )



if __name__ == '__main__':


    # opt_weights = 'ai_models/epoch_298.pt'
    opt_weights = './ai_models/mrz_detection_best.pt'
    
    opt_img_dir = './data'
    output_dir = './logs'

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

        # report_data.append([file_name, 0, 0, len(boxes), 0, 0, 0])

    
