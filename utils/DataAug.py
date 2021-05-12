import cv2
import numpy as np

class DataAug:
    """
        A basic class for data augmentation

        This class must have primitives to augment. Using the class you can
        construct more complex and sophisticated modules.
    """
    def resize(self, image, shapes, interpolation=cv2.INTER_AREA):
        """
        :param image: a numpy, HWC
        :param shapes: a list or tuple, [H,W]
        :param interpolation:
        :return resized image: a numpy, HWC
        """
        return cv2.resize(image, (shapes[1], shapes[0]), interpolation=interpolation)

    def flip(self, image, code):
        """

            https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#flip

        :param image: a numpy, HWC
        :param code: a integer, 0: horizontal, 1: vertical, -1: both
        :return flipped image: a numpy, HWC
        """
        return cv2.flip(image, code)

    def translate(self, image, offsets):
        """
        :param image: a numpy, HWC
        :param offsets: a list or tuple, [dx,dy]
        :return translated image: a numpy, HWC
        """
        M = np.asarray([[1, 0, offsets[0]],
                        [0, 1, offsets[1]]], dtype=np.float32)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def rotate(self, image, degrees, axes):
        """
        :param image: a numpy, HWC
        :param degrees: a integer
        :param axes: a list or tuple, [H,W]
        :return rotated image: a numpy, HWC
        """
        M = cv2.getRotationMatrix2D((axes[1], axes[0]), degrees, 1)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def crop(self, image, starts, sizes):
        """
        :param image: a numpy, HWC
        :param starts: a list or tuple, [y,x]
        :param sizes: a list or tuple, [H,W]
        :return cropped image: a numpy, HWC
        """
        return image[starts[0]:starts[0]+sizes[0], starts[1]:starts[1]+sizes[1]]

    def reflect(self, image):
        """
        :param image: a numpy, HWC
        :return tiled image: a numpy, (3H)(3W)C
        """
        syms = []
        h, w, _ = image.shape
        # 1. 원점
        M = np.asarray([[-1, 0, 512],
                        [0, -1, 512]], dtype=np.float32)
        syms.append(cv2.warpAffine(image, M, (w, h)))
        # 2. x축
        M[:] = [[-1, 0, 512],
                [0,  1, 0]]
        syms.append(cv2.warpAffine(image, M, (w, h)))
        # 3. y축
        M[:] = [[1, 0,  0],
                [0, -1, 512]]
        syms.append(cv2.warpAffine(image, M, (w, h)))
        tiled = np.tile(np.zeros(image.shape, dtype=np.uint8), (3, 3, 1))
        frags = [
            syms[0], syms[2], syms[0],
            syms[1], image,   syms[1],
            syms[0], syms[2], syms[0]
        ]
        for r in range(3):
            for c in range(3):
                tiled[h*r:h*(r+1), w*c:w*(c+1)] = frags[3*r+c]
        return tiled