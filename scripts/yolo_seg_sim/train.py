'''
data format

data/
├── train/
│   ├── images/      # 训练图片
│   └── labels/      # YOLO格式标注文件（.txt）
├── val/
│   ├── images/
│   └── labels/
└── data.yaml        # 数据集配置文件

'''

from ultralytics import YOLO
# 加载预训练模型
model = YOLO('yolo11s-seg.pt')  # 从yolov8s.pt初始化
# 训练配置（关键参数）
results = model.train(
    data='data.yaml',  # 数据集配置路径
    cfg='hyp.yaml',
    epochs=100,                              # 训练轮次
    imgsz=640,                               # 输入图片尺寸
    batch=16,                                # 批次大小（依GPU调整）
    single_cls=True,         # 强制视为单类别（即使nc>1）
    dropout=0.2,
    lr0=0.01,
    close_mosaic=10,
    name='yolov11s_single_obj_tomato_real_and_sim'# 训练任务名称
    )

'''
last.pt
best.pt
'''
