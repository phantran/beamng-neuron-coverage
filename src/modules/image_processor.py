from random import randrange

import numpy as np
import cv2


class ImageProcessor:
    abbr = {
        "Original": "Original",
        "Translation": "Tr",
        "Shearing": "Sh",
        "Rotation": "Ro",
        "Contrast": "Co",
        "Blurring": "Bl",
        "Brightness": "Br"
    }

    @staticmethod
    def image_translation(img):
        """
        Args:
            img:
        """
        p = randrange(2, 20) / 2
        params = [p * 10, p * 10]
        rows, cols, ch = img.shape
        m = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        dst = cv2.warpAffine(img, m, (cols, rows))
        return dst

    @staticmethod
    def image_shear(img):
        """
        Args:
            img:
        """
        p = randrange(42, 60) / 2 - 20
        params = 0.1 * p
        rows, cols, ch = img.shape
        factor = params * (-1.0)
        m = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(img, m, (cols, rows))
        return dst

    @staticmethod
    def image_rotation(img):
        """
        Args:
            img:
        """
        p = randrange(62, 80) / 2 - 30
        params = p * 3
        rows, cols, ch = img.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
        dst = cv2.warpAffine(img, m, (cols, rows))
        return dst

    @staticmethod
    def image_contrast(img):
        """
        Args:
            img:
        """
        p = randrange(82, 100) / 2 - 40
        params = 1 + p * 0.2
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))
        return new_img

    @staticmethod
    def image_blur(img):
        return cv2.blur(img, (5, 5))

    @staticmethod
    def image_brightness(img):
        """
        Args:
            img:
        """
        p = randrange(102, 120) / 2 - 50
        beta = p * 10
        b, g, r = cv2.split(img)
        b = cv2.add(b, beta)
        g = cv2.add(g, beta)
        r = cv2.add(r, beta)
        new_img = cv2.merge((b, g, r))
        return new_img

    @staticmethod
    def compose(f, g):
        return lambda x: f(g(x))

    @classmethod
    def get_transformations(cls):
        return [
            ("Original", None),
            ("Translation", cls.image_translation),
            ("Shearing", cls.image_shear),
            ("Rotation", cls.image_rotation),
            ("Contrast", cls.image_contrast),
            ("Blurring", cls.image_blur),
            ("Brightness", cls.image_brightness),
        ]
