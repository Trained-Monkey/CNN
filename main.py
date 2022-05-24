from sklearn.datasets import load_digits
import os
import sys
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math

from node import Node
from CNN import CNN
from tester import Tester

random.seed(200)

data = load_digits()
digit_data = data['data']
digit_label = data['target']

def show(data, title = "", shape=(8,8)):
    plt.imshow(np.reshape(data, shape))
    plt.title(title)
    plt.show()

class MNIST_Handler():
    def __init__(self):

        index = 0
        with open("images", "rb") as FILE:

            data = FILE.readline()
            curr = 0
            for i in range(4):
                curr *= 16 * 16
                curr += data[i]
            index += 4

            self.magic_number = curr

            if self.magic_number != 2051:
                raise ValueError("Magic Number differs, data is potentially corrupted")

            for i in range(3):
                curr = 0
                for a in range(4):
                    curr *= 16 * 16
                    curr += data[index + a]

                index += 4

            self.image_index = index

        index = 0
        with open("labels", "rb") as FILE:

            data = FILE.readline()
            curr = 0
            for i in range(4):
                curr *= 16 * 16
                curr += data[i]
            index += 4

            self.magic_number = curr

            if self.magic_number != 2049:
                raise ValueError("Magic Number differs, data is potentially corrupted")

            for i in range(1):
                curr = 0
                for a in range(4):
                    curr *= 16 * 16
                    curr += data[index + a]
                index += 4

            self.label_index = index

    def get_images(self, num, fname = "images", vector = True):
        with open(fname, "rb") as FILE:
            data = FILE.readline()
            result = []

            if vector == True:
                for i in range(num):
                    temp = []

                    for a in range(28 * 28):

                        try:
                            temp.append(data[self.image_index])
                        except:
                            data = FILE.readline()
                            self.image_index = 0
                            temp.append(data[self.image_index])

                        self.image_index += 1

                    result.append(temp)

            else:
                for i in range(num):
                    temp = []
                    for row in range(28):
                        new_row = []
                        for col in range(28):
                            new_row.append(data[self.image_index])
                            self.image_index += 1
                        temp.append(new_row)

                    result.append(temp)

        return result

    def get_labels(self, num, fname = "labels"):
        with open(fname, "rb") as FILE:
            data = FILE.readline()
            result = []

            for i in range(num):
                result.append(data[self.label_index])
                self.label_index += 1

            return result

if __name__ == "__main__":
    database = MNIST_Handler()
    samples = 1001

    digit_data = database.get_images(samples, "images")
    digit_label = database.get_labels(samples, "labels")

    test_data = database.get_images(samples, "test_images")
    test_label = database.get_labels(samples, "test_labels")

    evaluator = Tester(test_data, test_label)

    # for i in range(min(len(digit_data), 10)):
    #     show(digit_data[i], str(digit_label[i]), (28,28))

    # model = CNN.load_from_file("28_hidden")
    # model = CNN.load_from_file("28_hidden_true")
    model = CNN([784, 28, 10, 0])

    for i in range(samples):


        if i % 100 == 0:
            print(i)

        model.train(digit_data[i], digit_label[i])
        model.update()


    # print(model.test_batch(digit_data, digit_label, 1000, 10))

    # model.save("qwerty")

    print(evaluator.test(model, False))

    incorrect = [7,8,9]
    #
    for i in incorrect:
        show(digit_data[i], model.predict(digit_data[i]), (28, 28))

    # print(model.predict(digit_data[0]))
# [64, 96, 10, 0]
