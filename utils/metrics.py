import numpy as np

def accuracy_with_two_labels(probs, labels, threshold):
    """
        Get an accuracy with two labels

        Get an accuracy from comparing two labels that have
        largest values in the sequence

    :param probs: a numpy, NC
    :param labels: a numpy, NC
    :param threshold: a float
    :return:
    """
    first = []
    for n in range(probs.shape[0]):
        i = -1
        v = -1
        for c in range(probs.shape[1]):
            if v < probs[n, c]:
                i = c
                v = probs[n, c]
        first.append(i)
    second = []
    for n in range(probs.shape[0]):
        i = -1
        v = -1
        for c in range(probs.shape[1]):
            if c != first[n] and v < probs[n, c]:
                i = c
                v = probs[n, c]
        second.append(i)
    result = []
    for i in range(len(first)):
        if probs[i, first[i]] - probs[i, second[i]] < threshold:
            result.append([first[i], second[i]])
        else:
            result.append([first[i]])
    for i in range(len(result)):
        result[i].sort()
    label = []
    for n in range(labels.shape[0]):
        ones = []
        for c in range(labels.shape[1]):
            if labels[n, c] > 0.5:
                ones.append(c)
        label.append(ones)
    cors = 0
    for i in range(len(result)):
        if len(result[i]) == len(label[i]):
            valid = True
            for j in range(len(result[i])):
                valid &= label[i][j] == result[i][j]
            if valid == True:
                cors += 1
    return cors / len(result)

if __name__ == '__main__':
    probs = np.asarray([[0, 1, 0.999, 0, 0], [0, 0.999, 1, 0, 0], [1, 0, 0.999, 0, 0]])
    labels = np.asarray([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    print("probs: {}\nlabels: {}\n".format(probs, labels))
    print(accuracy_with_two_labels(probs, labels, 0.05))