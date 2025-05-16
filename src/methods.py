from datetime import datetime

from PIL import Image, ImageOps


def get_timestamp():
    return (datetime.now().__str__()
            .replace(" ", "_")
            .replace(":", "")
            .replace("-", "")
            .replace(".", "-")[:-3])


def load_image(image_path):
    return Image.open(image_path)


def resize_image(image, width, height, padding_rgb=(0, 0, 0)):
    image.thumbnail((width, height))
    delta_w = width - image.width
    delta_h = height - image.height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(image, padding, fill=padding_rgb)


