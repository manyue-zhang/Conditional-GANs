import numpy as np
from scipy.stats import entropy
import cv2
import os


def calculate_JSD(images1, images2):
    # Convert each image to a probability distribution
    p = np.histogram(images1, bins=256, range=(0, 256), density=True)[0]
    q = np.histogram(images2, bins=256, range=(0, 256), density=True)[0]

    # Normalize each histogram
    p /= np.sum(p)
    q /= np.sum(q)

    # Calculate the average histogram
    m = (p + q) / 2

    # Compute KL Divergence.
    kl1 = entropy(p, m)
    kl2 = entropy(q, m)

    # Calculate the JSD
    jsd = np.sqrt((kl1 + kl2) / 2)
    return jsd


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == "__main__":
    # folder1 = "D:\\Desktop\\Ergebniss\\cifar10"
    folder1 = "D:\\Desktop\\Ergebniss\\cifar10"
    folder2 = "D:\\Desktop\\generator_model_75\\generator_model_75.h5"
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)

    # Convert images to numpy arrays
    images1 = np.array(images1)
    images2 = np.array(images2)

    jsd = calculate_JSD(images1, images2)
    print("JSD:", jsd)
