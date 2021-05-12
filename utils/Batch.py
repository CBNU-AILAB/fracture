import numpy as np

class Batch:
    def __init__(self, x, y, batch_size):
        """
            Batch class to process data in mini batch

            In a case for 2D image processing, the x will be a 4D tensor
            have a size of [batch_size, height, width, channels] and the
            y will be a little different from a specific settings. But,
            it is usually [batch_size, number_of_classes].

        :param x: a numpy
        :param y: a numpy
        :param batch_size: an integer
        :return: None
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.index = 0

    def _reach_to_last(self):
        return self.x.shape[0] < self.index + self.batch_size

    def _shuffle(self):
        new_indexes = [i for i in range(self.x.shape[0])]
        np.random.shuffle(new_indexes)
        new_x = np.zeros(self.x.shape)
        new_y = np.zeros(self.y.shape)
        for i in range(self.x.shape[0]):
            new_x[i] = self.x[new_indexes[i]]
            new_y[i] = self.y[new_indexes[i]]
        self.x = new_x
        self.y = new_y

    def next_to(self):
        """
        :return batch_x: a numpy
                batch_y: a numpy
        """
        if self._reach_to_last():
            self.index = 0
        if self.index == 0:
            self._shuffle()
        batch_x = self.x[self.index:self.index + self.batch_size]
        batch_y = self.y[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch_x, batch_y