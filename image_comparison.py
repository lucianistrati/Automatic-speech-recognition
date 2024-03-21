from PIL import Image

import imagehash


def similar_frames(image_a_path: str, image_b_path: str, threshold: float = 4.0):
    image_a_hash = imagehash.average_hash(Image.open(image_a_path))
    image_b_hash = imagehash.average_hash(Image.open(image_b_path))
    return abs(image_a_hash - image_b_hash) <= threshold