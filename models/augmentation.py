import cv2
import numpy as np
import albumentations as A
from pathlib import Path

# Separate augmentation pipelines for deterministic output
snow_augmentation = A.Compose([
    A.RandomSnow(snow_point_range=(0.1, 0.3), brightness_coeff=1.3, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.3),
])

fog_augmentation = A.Compose([
    A.RandomFog(fog_coef_range=(0.05, 0.15), alpha_coef=0.05, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.3),
])

# Load image
image_path = "data/archive/military_object_dataset/train/images"
first_image = next(Path(image_path).glob("*.jpg"))
image = cv2.imread(str(first_image))

# Resize for display
image = cv2.resize(image, (320, 320))

# Generate 3 deterministic versions: original, snow, fog
original = image
snow_img = snow_augmentation(image=image)["image"]
fog_img = fog_augmentation(image=image)["image"]

# Create 1x3 grid with labels
images = [original, snow_img, fog_img]
grid = np.hstack(images)

# Add labels
labeled_grid = grid.copy()
labels = ["Original", "Snow", "Fog"]
for i, label in enumerate(labels):
    x = i * 320 + 10
    cv2.putText(labeled_grid, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Save and display
cv2.imwrite("augmentation_preview.png", labeled_grid)
cv2.imshow("Augmentation Comparison", labeled_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
