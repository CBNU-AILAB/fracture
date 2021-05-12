import numpy as np

class Preproc:
    def get_means(self, images):
        """
            Get a mean vector

            Get a mean vector from images. The element in the vector
            is calculated in corresponding to each channel. If you
            will process RGB images then the dimensions of the vector
            are 3-D.
        :param images: a numpy, NHWC
        :return means: a numpy, C
        """
        means = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        for n in range(images.shape[0]):
            for h in range(images.shape[1]):
                for w in range(images.shape[2]):
                    means += images[n, h, w]
        means /= images.shape[0] * images.shape[1] * images.shape[2]
        return means

    def subtract_in_each_dim(self, images, means):
        """
            Apply a mean subtracted operation to images with the mean vector

        :param images: a numpy, NHWC
        :param means: a numpy, C
        :return mean subtracted images: a numpy, NHWC
        """
        subtracted = np.zeros(images.shape, dtype=np.float32)
        for n in range(images.shape[0]):
            for h in range(images.shape[1]):
                for w in range(images.shape[2]):
                    subtracted[n, h, w] = images[n, h, w] - means
        return subtracted

    def subtract_in_all_dim(self, images, mean):
        """
            Apply a mean subtracted operation to images with the mean scalar

        :param images: a numpy, NHWC
        :param mean: a numpy, C
        :return mean subtracted images: a numpy, NHWC
        """
        subtracted = np.zeros(images.shape, dtype=np.float32)
        for n in range(images.shape[0]):
            for h in range(images.shape[1]):
                for w in range(images.shape[2]):
                    for c in range(images.shape[3]):
                        subtracted[n, h, w, c] = images[n, h, w, c] - mean
        return subtracted