# -*- coding: utf-8 -*-
"""
@author: SRIVATSAV
"""

from ultralytics import YOLO
    # from ultralytics import YOLOv10 

import torch


if __name__ == '__main__':

    # torch.cuda.set_device(0)
    
    # device = torch.device("cuda:0")


    # Load a model
    model = YOLO("yolov10n.yaml")  # build a new model from scratch
    
    # Load a pretrained YOLO model (recommended for training)
    #model = YOLO('yolov10n.pt')
    model = YOLO('C:/Users/sriva/Downloads/weapons_detection/weights_yolov10n/best.pt')

    #results = model.predict(source="0", show=True, stream=True, classes=0, device='0')
    
    # Use the model
    results = model.train(data="C:/Users/sriva/Downloads/weapons_detection/config.yaml",device='0', epochs=650,imgsz=320,
                          optimizer="AdamW",name="yolov10n_new_god2",augment=True,lr0=0.0001,patience=None,cos_lr=False,workers=8,batch=32)  # train the model
    
    #results = model.tune(data="/home/ubuntu/sabari/Yolo/config.yaml", epochs=100,imgsz=300,name="yolov8n_face_person_white_fine_tune",use_ray=True)  # train the model
    
    
    # Export the model to ONNX format
    model.export(format="onnx",imgsz=320)# dynamic=True, half=True)
    
    #model.export(format="openvino",imgsz=320)# dynamic=True, half=True)
