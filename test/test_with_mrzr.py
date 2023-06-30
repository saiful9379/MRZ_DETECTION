"""
python detect.py --weights ai_models/best.pt --conf 0.25 --img-size 640 --source data/training


"""
import os
import time
import glob


import sys
import cv2



sys.path.insert(0, './mrz_detection_api')
from passporteye import read_mrz
# mrz detection
from mrz_detection_api.mrz_detect import load_model, inference
# mrz recognition
from mrz_recognition.mrz_recognizer import read_mrz_data_processing


def image_processing(roi_image, image_path):
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray, h = 15)
    clahe = cv2.createCLAHE(clipLimit = 1.0,tileGridSize = (2,2))
    denoised_image = clahe.apply(denoised_image)
    # name = uuid.uuid1()
    # image_path = path_to_save+"/"+str(name)+".png"
    # os.makedirs(path_to_save, exist_ok=True)
    cv2.imwrite(image_path, denoised_image)


def start_processing(model, image, file_name="unknown.jpg", output_dir="logs"):

    # mrz detection
    bboxs = inference(
        model, image, output_dir=output_dir
        )
    mrz_images = []
    for bbox in bboxs:
        print(bbox)
        x1, y1, x2, y2 = bbox
        roi_image = image[y1:y2, x1:x2]
        # mrz_images.append(roi)
        # image_processing(roi_image, file_name)
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # print(bw)

        cv2.imwrite(file_name, bw)
        language="mrz"
        # # results = read_mrz(file_name)
        results = read_mrz(file_name, extra_cmdline_params='-l {}'.format(language)).to_dict()
        # # mrz_data = read_mrz_data_processing(
        # #     file_name, 
        # # )
        print(results)
    
    # image
    
    # mrz_data = read_mrz_data_processing(
    #     file_name, 
    #     language="mrz"
    #     )



if __name__ == '__main__':

    # opt_weights = 'ai_models/epoch_298.pt'
    opt_weights = './mrz_detection_api/ai_models/mrz_detection_best.pt'
    output_dir = 'mrz_detection_api/logs'

    os.makedirs(output_dir, exist_ok=True)
    print("model loading ..........")
    model = load_model(opt_weights)
    print("Done")

    files = "/home/sayan/Desktop/mrz_module/mrz_detection_api/data/B2 - VALID.png"

    report_data = []
    # for _file in files:
    file_name = os.path.basename(files)
    print(f" File {file_name} : ", end="", flush=True)
    image = cv2.imread(files)

    boxes = start_processing(
        model,
        image, 
        output_dir= os.path.join(output_dir, file_name)
    )
    # print(boxes)
    print("Done")
