import cv2
import numpy as np


# --------------------------------------------------
# Variance
# --------------------------------------------------

def calculate_variance(block):
    return float(np.var(block))


# --------------------------------------------------
# Entropy
# --------------------------------------------------

def calculate_entropy(block):
    histogram = cv2.calcHist(
        [block.astype(np.uint8)],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()

    probabilities = histogram / np.sum(histogram)

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


# --------------------------------------------------
# Edge Density
# --------------------------------------------------

def calculate_edge_density(block):
    block = block.astype(np.uint8)

    edges = cv2.Canny(
        block,
        20,
        60
    )

    edge_pixels = np.count_nonzero(edges)

    total_pixels = block.shape[0] * block.shape[1]

    return float(edge_pixels / total_pixels)


# --------------------------------------------------
# Gradient Magnitude
# --------------------------------------------------

def calculate_gradient_magnitude(block):

    block = block.astype(np.float32)

    gx = cv2.Sobel(
        block,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        block,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    return float(np.mean(magnitude))


# --------------------------------------------------
# Pairwise Pixel Differences
# --------------------------------------------------

def calculate_pairwise_differences(block):

    block = block.astype(np.int32)

    # Horizontal differences
    horizontal = np.abs(
        block[:, 1:] - block[:, :-1]
    )

    # Vertical differences
    vertical = np.abs(
        block[1:, :] - block[:-1, :]
    )

    differences = np.concatenate([
        horizontal.flatten(),
        vertical.flatten()
    ])

    return differences


# --------------------------------------------------
# Mean Pixel Difference
# --------------------------------------------------

def calculate_mean_pixel_difference(block):

    differences = calculate_pairwise_differences(block)

    return float(np.mean(differences))


# --------------------------------------------------
# Maximum Pixel Difference
# --------------------------------------------------

def calculate_max_pixel_difference(block):

    differences = calculate_pairwise_differences(block)

    return float(np.max(differences))


# --------------------------------------------------
# Standard Deviation of Pixel Differences
# --------------------------------------------------

def calculate_std_pixel_difference(block):

    differences = calculate_pairwise_differences(block)

    return float(np.std(differences))


# --------------------------------------------------
# Extract all features
# --------------------------------------------------

def extract_features(block):

    variance = calculate_variance(block)

    entropy = calculate_entropy(block)

    edge_density = calculate_edge_density(block)

    gradient_magnitude = calculate_gradient_magnitude(block)

    mean_difference = calculate_mean_pixel_difference(block)

    max_difference = calculate_max_pixel_difference(block)

    std_difference = calculate_std_pixel_difference(block)

    return [
        variance,
        entropy,
        edge_density,
        gradient_magnitude,
        mean_difference,
        max_difference,
        std_difference
    ]