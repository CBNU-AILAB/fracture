import os
import cv2

class Load:
    def load(self, path):
        """
        :param path: a string
        :return self.images: a list of numpy
                self.labels: a list of list of integer
        """
        self.images = []
        self.labels = []
        # path내에 labels.txt가 있다는 것을 보장해야함
        for label in os.listdir(path) :
            for file in os.listdir(path+'/'+label):
                img = cv2.imread(os.path.join(path+'/'+label, file))
                self.images.append(img)
                self.labels.append([int(i) for i in label.split('-')])
    def get(self):
        """
        :return self.images: a list of numpy
                self.labels: a list of list of integer
        """
        return self.images, self.labels