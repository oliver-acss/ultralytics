from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 使用自定义的yolov8-package模型配置
    model = YOLO('ultralytics/cfg/models/v8/yolov8-package2.yaml')
    
    # 训练参数设置
    results = model.train(
        data="package-seg/package-seg.yaml",  # 数据集配置文件路径
        epochs=50,                            # 训练轮数
        imgsz=640,                           # 输入图像大小
        batch=8                              # 批次大小
    )
    
    # 验证模型
    model.val()