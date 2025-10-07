import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from typing import Tuple, Dict
import random 


def apply_noise(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    
    params = {
        "rotation": random.uniform(-1.5, 1.5), 
        "skew": random.uniform(-2, 2), 
        "perspective_scale": random.uniform(0.005, 0.02), 
        "blur_sigma": random.uniform(0.1, 0.4),  
        "jpeg_quality": random.randint(80, 98),  
        "downscale_factor": random.uniform(0.85, 1.0),  
        "salt_pepper_p": random.uniform(0.0001, 0.001),  
        "shadow_alpha": random.uniform(0.1, 0.25),  
        "shadow_sigma": random.uniform(0.3, 0.5),  
        "multiply_factor": random.choice([0.7, 0.85])  
    }

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  

    noise_pipeline = iaa.Sequential([
        sometimes(iaa.PerspectiveTransform(scale=params["perspective_scale"], fit_output=True)),
        sometimes(iaa.Affine(
            rotate=params["rotation"],
            shear=params["skew"],
            cval=255, 
            mode='constant'
        )),

        sometimes(iaa.GaussianBlur(sigma=params["blur_sigma"])),
        sometimes(iaa.JpegCompression(compression=params["jpeg_quality"])),

        sometimes(iaa.SaltAndPepper(p=params["salt_pepper_p"])),
        sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.03*255))),
        sometimes(
            iaa.BlendAlpha(
                factor=params["shadow_alpha"],
                foreground=iaa.Grayscale(1.0),
                background=iaa.SimplexNoiseAlpha(
                    first=iaa.Multiply(params["multiply_factor"], per_channel=False),
                    second=iaa.ContrastNormalization((0.6, 1.2)),
                    upscale_method='linear',
                    size_px_max=320,
                )
            )
        ),

        iaa.Resize(params["downscale_factor"]),
        iaa.Resize({"height": image.shape[0], "width": image.shape[1]})
    ], random_order=False)  

    augmented_image = noise_pipeline(image=image)
    return augmented_image, params