import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from typing import Tuple, Dict
import random 
import cv2


def add_jpeg_compression_artifacts(image: np.ndarray, quality: int = None) -> np.ndarray:
    if quality is None:
        quality = random.randint(75, 95)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)
    
    return decoded_image


def add_color_shift(image: np.ndarray, shift_amount: float = None) -> np.ndarray:
    if shift_amount is None:
        shift_amount = random.uniform(5, 20)
    
    result = image.copy().astype(np.float32)
    
    if image.ndim == 3:
        cast_type = random.choice(['warm', 'cool', 'neutral'])
        
        if cast_type == 'warm':
            result[:, :, 0] = np.clip(result[:, :, 0] - shift_amount * 0.3, 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] + shift_amount * 0.2, 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] + shift_amount * 0.5, 0, 255)
        elif cast_type == 'cool':
            result[:, :, 0] = np.clip(result[:, :, 0] + shift_amount * 0.5, 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] + shift_amount * 0.1, 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] - shift_amount * 0.2, 0, 255)
    
    return np.clip(result, 0, 255).astype(np.uint8)



def add_ink_bleed(image: np.ndarray, intensity: float = None) -> np.ndarray:
    if intensity is None:
        intensity = random.uniform(0.3, 0.8)
    
    mask = image < 200
    
    blurred = cv2.GaussianBlur(image, (3, 3), intensity)
    result = image.copy().astype(np.float32)
    blurred = blurred.astype(np.float32)
    
    result = np.where(mask, 
                      result * 0.7 + blurred * 0.3,
                      result)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_dust_spots(image: np.ndarray, num_spots: int = None) -> np.ndarray:
    if num_spots is None:
        num_spots = random.randint(5, 25)
    
    result = image.copy()
    h, w = result.shape[:2]
    
    for _ in range(num_spots):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        
        spot_size = random.randint(1, 4)
        spot_type = random.choice(['dark', 'light'])
        
        if spot_type == 'dark':
            color = random.randint(0, 100)
        else:
            color = random.randint(200, 255)
        
        cv2.circle(result, (x, y), spot_size, 
                  color if image.ndim == 2 else (color, color, color), -1)
    
    return result


def add_horizontal_bands(image: np.ndarray, num_bands: int = None) -> np.ndarray:
    if num_bands is None:
        num_bands = random.randint(2, 5)
    
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]
    
    for _ in range(num_bands):
        band_y = random.randint(int(h * 0.1), int(h * 0.9))
        band_height = random.randint(2, 8)
        band_intensity = random.uniform(0.02, 0.08)
        
        y_start = max(0, band_y - band_height // 2)
        y_end = min(h, band_y + band_height // 2)
        
        multiplier = 1 - band_intensity if random.random() < 0.5 else 1 + band_intensity
        result[y_start:y_end, :] *= multiplier
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_page_curl(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    edge = random.choice(['top', 'bottom', 'left', 'right'])
    curl_size = random.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.15))
    
    gradient = np.linspace(0, 1, curl_size)
    shadow_intensity = random.uniform(0.1, 0.3)
    
    if edge == 'top':
        for i, alpha in enumerate(gradient):
            result[i, :] *= (1 - alpha * shadow_intensity)
    elif edge == 'bottom':
        for i, alpha in enumerate(gradient[::-1]):
            result[h - curl_size + i, :] *= (1 - alpha * shadow_intensity)
    elif edge == 'left':
        for i, alpha in enumerate(gradient):
            result[:, i] *= (1 - alpha * shadow_intensity)
    else:
        for i, alpha in enumerate(gradient[::-1]):
            result[:, w - curl_size + i] *= (1 - alpha * shadow_intensity)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_paper_texture(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    
    noise = np.random.normal(0, 2, (h, w))
    
    result = image.copy().astype(np.float32)
    if image.ndim == 3:
        for c in range(image.shape[2]):
            result[:, :, c] += noise
    else:
        result += noise
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_vertical_scan_lines(image: np.ndarray, num_lines: int = None, 
                           opacity: float = None) -> np.ndarray:
    if num_lines is None:
        num_lines = random.randint(2, 8)
    if opacity is None:
        opacity = random.uniform(0.05, 0.25)
    
    result = image.copy()
    h, w = image.shape[:2]
    
    line_positions = random.sample(range(int(w * 0.1), int(w * 0.9)), 
                                  min(num_lines, w))
    
    for x in line_positions:
        thickness = random.randint(1, 3)
        darkness = int(opacity * 255)
        
        for y in range(h):
            local_opacity = opacity * random.uniform(0.7, 1.0)
            if image.ndim == 3:
                result[y, max(0, x-thickness//2):min(w, x+thickness//2+1)] = \
                    result[y, max(0, x-thickness//2):min(w, x+thickness//2+1)] * (1 - local_opacity)
            else:
                result[y, max(0, x-thickness//2):min(w, x+thickness//2+1)] *= (1 - local_opacity)
    
    return result


def add_corner_shadows(image: np.ndarray, num_corners: int = None,
                       shadow_intensity: float = None) -> np.ndarray:
    if num_corners is None:
        num_corners = random.randint(1, 3)
    if shadow_intensity is None:
        shadow_intensity = random.uniform(0.15, 0.4)
    
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]
    
    corners = ['tl', 'tr', 'bl', 'br']
    selected_corners = random.sample(corners, num_corners)
    
    for corner in selected_corners:
        shadow_size_h = random.randint(int(h * 0.1), int(h * 0.3))
        shadow_size_w = random.randint(int(w * 0.1), int(w * 0.3))
        
        y_grad = np.linspace(1, 0, shadow_size_h)
        x_grad = np.linspace(1, 0, shadow_size_w)
        
        if corner in ['tl', 'bl']:
            x_mesh = np.tile(x_grad, (shadow_size_h, 1))
        else:
            x_mesh = np.tile(x_grad[::-1], (shadow_size_h, 1))
            
        if corner in ['tl', 'tr']:
            y_mesh = np.tile(y_grad.reshape(-1, 1), (1, shadow_size_w))
        else:
            y_mesh = np.tile(y_grad[::-1].reshape(-1, 1), (1, shadow_size_w))
        
        gradient = np.minimum(x_mesh, y_mesh)
        gradient = np.power(gradient, random.uniform(1.5, 2.5))
        
        shadow_multiplier = 1 - (gradient * shadow_intensity)
        
        if corner == 'tl':
            if result.ndim == 3:
                for c in range(result.shape[2]):
                    result[:shadow_size_h, :shadow_size_w, c] *= shadow_multiplier
            else:
                result[:shadow_size_h, :shadow_size_w] *= shadow_multiplier
        elif corner == 'tr':
            if result.ndim == 3:
                for c in range(result.shape[2]):
                    result[:shadow_size_h, -shadow_size_w:, c] *= shadow_multiplier
            else:
                result[:shadow_size_h, -shadow_size_w:] *= shadow_multiplier
        elif corner == 'bl':
            if result.ndim == 3:
                for c in range(result.shape[2]):
                    result[-shadow_size_h:, :shadow_size_w, c] *= shadow_multiplier
            else:
                result[-shadow_size_h:, :shadow_size_w] *= shadow_multiplier
        else:
            if result.ndim == 3:
                for c in range(result.shape[2]):
                    result[-shadow_size_h:, -shadow_size_w:, c] *= shadow_multiplier
            else:
                result[-shadow_size_h:, -shadow_size_w:] *= shadow_multiplier
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_noise(image: np.ndarray) -> Tuple[np.ndarray, Dict]:

    params = {
        "rotation": random.uniform(-2, 2),
        "skew": random.uniform(-0.7, 0.7), 
        "perspective_scale": random.uniform(0.002, 0.09),  
        "blur_sigma": random.uniform(0.0, 0.08), 
        "downscale_factor": random.uniform(0.98, 1.0),  
        "salt_pepper_p": random.uniform(0.00001, 0.0009), 
        "gaussian_noise_scale": random.uniform(0.0002, 0.003), 
        "shadow_alpha": random.uniform(0.01, 0.04), 
        "brightness_multiply": random.uniform(0.95, 1.05),  
        "add_texture": random.random() < 0.85,  
        "add_fold_line": random.random() < 0.4,  
        "edge_fade": random.random() < 0.4,
        "add_scan_lines": random.random() < 0.6,
        "num_scan_lines": random.randint(2, 8),
        "scan_line_opacity": random.uniform(0.05, 0.25),
        "add_corner_shadows": random.random() < 0.5,
        "num_corner_shadows": random.randint(1, 3),
        "corner_shadow_intensity": random.uniform(0.15, 0.4),
        "add_jpeg_artifacts": random.random() < 0.4,
        "jpeg_quality": random.randint(75, 95),
        "add_color_shift": random.random() < 0.3,
        "color_shift_amount": random.uniform(5, 20),
        "add_ink_bleed": random.random() < 0.25,
        "ink_bleed_intensity": random.uniform(0.3, 0.8),
        "add_dust_spots": random.random() < 0.5,
        "num_dust_spots": random.randint(5, 25),
        "add_horizontal_bands": random.random() < 0.35,
        "num_bands": random.randint(2, 5),
        "add_page_curl": random.random() < 0.2,
        "add_paper_texture": random.random() < 0.7,
    }

    augmenters = []

    if random.random() < 0.5:
        augmenters.append(
            iaa.PerspectiveTransform(
                scale=params["perspective_scale"], 
                fit_output=True,
                keep_size=True
            )
        )
    
    if random.random() < 0.4:
        augmenters.append(
            iaa.Affine(
                rotate=params["rotation"],
                shear=params["skew"],
                cval=255,
                mode='constant'
            )
        )
    
    if params["add_texture"]:
        augmenters.append(
            iaa.SimplexNoiseAlpha(
                first=iaa.Multiply(0.98, per_channel=False),
                per_channel=False,
                size_px_max=4,
                upscale_method='linear',
                aggregation_method='max'
            )
        )
    
    if random.random() < 0.3 and params["blur_sigma"] > 0:
        augmenters.append(iaa.GaussianBlur(sigma=params["blur_sigma"]))
    
    if random.random() < 0.5:
        augmenters.append(
            iaa.Multiply(params["brightness_multiply"], per_channel=False)
        )
    
    if random.random() < 0.4:
        augmenters.append(
            iaa.LinearContrast((0.9, 1.1))
        )
    
    if random.random() < 0.3:
        augmenters.append(
            iaa.BlendAlphaSimplexNoise(
                foreground=iaa.Multiply(0.95, per_channel=False),
                background=iaa.Multiply(1.0, per_channel=False),
                per_channel=False,
                size_px_max=100,
                upscale_method='linear',
                aggregation_method='max',
                sigmoid=True,
                sigmoid_thresh=0.5
            )
        )
    
    if params["edge_fade"]:
        augmenters.append(
            iaa.BlendAlphaVerticalLinearGradient(
                iaa.Multiply(0.9, per_channel=False),
                start_at=(0.0, 0.05),
                end_at=(0.0, 0.0)
            )
        )
    
    if random.random() < 0.6:  
        if random.random() < 0.5:
            augmenters.append(
                iaa.AdditiveGaussianNoise(
                    scale=params["gaussian_noise_scale"] * 255,
                    per_channel=False
                )
            )
        else:
            augmenters.append(
                iaa.SaltAndPepper(p=params["salt_pepper_p"])
            )
    
    if params["downscale_factor"] < 1.0:
        augmenters.append(iaa.Resize(params["downscale_factor"]))
        augmenters.append(iaa.Resize({"height": image.shape[0], "width": image.shape[1]}))
    
    if augmenters:
        noise_pipeline = iaa.Sequential(augmenters, random_order=False)
        augmented_image = noise_pipeline(image=image)
    else:
        augmented_image = image
    
    if params["add_paper_texture"]:
        augmented_image = add_paper_texture(augmented_image)
    
    if params["add_ink_bleed"]:
        augmented_image = add_ink_bleed(
            augmented_image,
            intensity=params["ink_bleed_intensity"]
        )
    
    if params["add_dust_spots"]:
        augmented_image = add_dust_spots(
            augmented_image,
            num_spots=params["num_dust_spots"]
        )
    
    if params["add_scan_lines"]:
        augmented_image = add_vertical_scan_lines(
            augmented_image,
            num_lines=params["num_scan_lines"],
            opacity=params["scan_line_opacity"]
        )
    
    if params["add_horizontal_bands"]:
        augmented_image = add_horizontal_bands(
            augmented_image,
            num_bands=params["num_bands"]
        )
    
    if params["add_corner_shadows"]:
        augmented_image = add_corner_shadows(
            augmented_image,
            num_corners=params["num_corner_shadows"],
            shadow_intensity=params["corner_shadow_intensity"]
        )
    
    if params["add_page_curl"]:
        augmented_image = add_page_curl(augmented_image)
    
    if params["add_color_shift"]:
        augmented_image = add_color_shift(
            augmented_image,
            shift_amount=params["color_shift_amount"]
        )
    
    if params["add_jpeg_artifacts"]:
        augmented_image = add_jpeg_compression_artifacts(
            augmented_image,
            quality=params["jpeg_quality"]
        )
    
    return augmented_image, params