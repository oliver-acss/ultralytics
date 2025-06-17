from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 使用预定义的YOLOv8n模型
    model = YOLO('yolov8n.yaml')
    model.train(data="E:/pythonproject/datasets_fruit/data.yaml", imgsz=640, epochs=20, batch=16)