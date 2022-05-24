class Tester():
    def __init__(self, test_data, test_labels):
        self.test_data = test_data
        self.test_labels = test_labels

        self.predictions = [0] * 10
        self.result = [0] * 10
        self.correct = 0

    def test(self, CNN, shuffle = True):

        if shuffle:
            order = []
            for i in range(len(self.test_data)):
                order.append(i)

            reordered_data = []
            reordered_labels = []

            for i in order:
                reordered_data.append(self.test_data[i])
                reordered_labels.append(self.test_labels[i])
        else:
            reordered_data = self.test_data
            reordered_labels = self.test_labels

        for i in range(len(self.test_data)):
            ans = CNN.predict(reordered_data[i])
            expected = reordered_labels[i]
            if ans == expected:
                self.correct += 1

            self.predictions[ans] += 1
            self.result[expected] += 1



        return (self.correct/len(self.test_data), self.predictions, self.result)
