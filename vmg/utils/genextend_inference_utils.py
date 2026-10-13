import numpy as np
from PIL import ImageEnhance

def calculate_brightness(image):
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)
    for index in range(scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)
    return 1 if brightness == 255 else brightness / scale

def calculate_contrast(image):
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = sum(histogram)
    mean = sum(i * w for i, w in enumerate(histogram)) / pixels
    contrast = sum((i - mean) ** 2 * w for i, w in enumerate(histogram)) / pixels
    return contrast ** 0.5

def adjust_brightness_contrast(image, target_brightness, target_contrast):
    current_brightness = calculate_brightness(image)

    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(target_brightness / current_brightness)

    current_contrast = calculate_contrast(image)
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(target_contrast / current_contrast)

    return image

def adjust_statistics_to_match_reference(video_batch,
                                         start_index_prediction=17,
                                         reference_window_size=3):
    assert start_index_prediction > 1, f"Need at least 1 frame before prediction start"
    assert start_index_prediction > reference_window_size, f"Reference window size incorrect: {start_index_prediction} <= {reference_window_size}"
    for video in video_batch:

        window_start = max(start_index_prediction - reference_window_size, 0)

        window_brightness = np.mean(
            [calculate_brightness(video[jj]) for jj in range(window_start, start_index_prediction)])
        window_contrast = np.mean(
            [calculate_contrast(video[jj]) for jj in range(window_start, start_index_prediction)])
        for ii in range(start_index_prediction, len(video)):
            video[ii] = adjust_brightness_contrast(video[ii], window_brightness, window_contrast)

    return video_batch
