
# API Integration

# Environemnt
```
conda create -n mrz_detection python=3.8

conda activate mrz_detection
```

# Requirment,

```
pip install -r requirements.txt
```

# Inference

1. Put the model ```mrz_detection_best.pt``` into ```mrz_detection_api/ai_models``` directory. download model please [click here](https://drive.google.com/file/d/1Yj7bCesqiXVO-ny8C1_suCyuw37qV0_G/view?usp=sharing)

2. Active virtual environment and open terminal into ```mrz_detection_api``` folder.

Run,

```
python3 -m test.test
```

input: ```./data```


output: ```./logs```

check the logs folder for detected mrz

# Integration
```py

import sys
sys.path.insert(0, './mrz_detection_api')

from mrz_detection_api.mrz_detect import load_model, inference

# def start_processing(model, image, output_dir="logs"):
img_file = './mrz_detection_api/data/1.jpg'
model_path_dir = './mrz_detection_api/ai_models/mrz_detection_best.pt'
image = cv2.imread(img_file)
model = load_model(model_path_dir)
box = inference(model, image)

```  
