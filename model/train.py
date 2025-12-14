from ultralytics import YOLO

import albumentations as A

arctic_augmentations = A.Compose([
    A.RandomSnow(snow_point_range=(0.1, 0.3), brightness_coeff=1.2, p=0.5),
    A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.ToGray(p=0.1),
])

model = YOLO('yolov8n.pt')

# Very limited training configuration for rapid prototyping
model.train(
    data='model/arctic_military.yaml',
    epochs=10,
    imgsz=224,
    batch=8,
    fraction=0.5,
    amp=True,
    freeze=10,
    project='arctic_military_yolo_prototype',
    name='exp2',
    augmentations=arctic_augmentations
)