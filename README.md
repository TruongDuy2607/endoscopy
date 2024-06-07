# Upper Gastrointestinal tract classification using Hierarchical Neural Network
In this task, we propose a method to classify upper gastrointestinal endoscopic landmarks based on the Hierarchical Neural Network architecture.

## 1. Format dataset
Each input data sample has a size of 224x224 and is assigned 2 labels corresponding to 2 branches: coarse branch and fine branch.

    
    ├── Non-Informative frames/            
    │   ├── Unkown        
    │   ├── Blur     
    │   ├── Foam/Fluid 
    │   └── Dark
    └── Informative frames/
        ├── Pharynx        
        ├── Oesophagus     
        ├── Squamocolumnar junction
        ├── Middle upper body
        ├── Fundus
        ├── Antrum
        ├── Angulus
        ├── Greater curvature
        ├── Lesser curvature
        └── Dark
## 2. Model architecture
<p align="center"> <img src="image/model_art/model_architecture.png" alt="landing graphic" width="75%" height="300px"/></p>

## 3. Results
## I. Deployment with video input
- To run and show the result of the model with video input, run the following command:
```bash
python check.py
```
- To save the result of the model as each imge foldes, simulator video with video input, run the following command:
```bash
python check.py --save_image --save_vid
python check.py --video_path "path_to_video" --frame-rate 30 # default frame rate 20
```
## III. Requirements
- Python 3.10
- torch==1.9.0, torchvision==0.18.0, torchsummary==1.5.1
- cuda 12.3


