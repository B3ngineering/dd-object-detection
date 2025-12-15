from ultralytics import YOLO
import albumentations as A

arctic_augmentations = A.Compose([
    A.RandomSnow(snow_point_range=(0.1, 0.25), brightness_coeff=1.2, p=0.3),
    A.RandomFog(fog_coef_range=(0.05, 0.2), alpha_coef=0.06, p=0.2),
])

model = YOLO('yolov8s.pt')

# Very limited training configuration for rapid prototyping
model.train(
    data='models/arctic_military.yaml',
    epochs=25,
    imgsz=224,
    batch=32,
    fraction=1.0,
    amp=True,
    freeze=5,
    project='arctic_military_yolo_prototype',
    name='exp7',
    augmentations=arctic_augmentations
)
