import ssl
from sklearn.datasets import fetch_openml
import jax.numpy as jnp

import utils

ssl._create_default_https_context = ssl._create_unverified_context

def get_training_data():
    images, labels = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    images = images.to_numpy()
    images = images.reshape(images.shape[0], 28, 28, 1) # (n, 28, 28, 1)
    normalized_images = jnp.array(utils.normalize_to_neg_one_to_one(images))
    labels = jnp.uint16(labels.to_numpy())

    return normalized_images
