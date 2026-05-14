import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class WasilPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(self, image_path, augment=False):
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply optional augmentations using OpenCV
        if augment:
            img = self.apply_augmentations(img)

        # Convert to PIL for torchvision transforms
        pil_img = Image.fromarray(img)

        # Apply standard transforms (resize, normalize)
        tensor_img = self.transform(pil_img)

        return tensor_img

    # def apply_augmentations(self, img):
    #     # Small rotation between ±2 and ±5 degrees
    #     angle = np.random.uniform(2, 5) * np.random.choice([-1, 1])
    #     h, w = img.shape[:2]
    #     M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    #     img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    #     # Brightness/contrast jitter
    #     alpha = np.random.uniform(0.8, 1.2)  # Contrast
    #     beta = np.random.uniform(-20, 20)  # Brightness
    #     img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    #     # Slight blur
    #     if np.random.random() > 0.5:
    #         img = cv2.GaussianBlur(img, (3, 3), 0)

    #     # Light noise
    #     if np.random.random() > 0.5:
    #         noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    #         img = cv2.add(img, noise)

    #     return img

if __name__ == "__main__":
    # Test preprocessor
    preprocessor = WasilPreprocessor()
    print("Preprocessor initialized.")
