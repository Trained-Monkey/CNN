from layer import Layer
import pickle

class CNN():
    def __init__(self, num_nodes):
        num_layers = len(num_nodes)

        self.predictions = [0] * 10

        self.layers = []
        self.num_layers =  num_layers
        self.num_nodes = []

        for i in range(num_layers - 1):
            new_layer = Layer(num_nodes[i], num_nodes[i + 1], True)
            new_layer.construct()

            if i > 0:
                self.layers[-1].connect_nodes(new_layer)

            self.layers.append(new_layer)

    def forward_pass(self, data):
        self.layers[0].load(data)
        for i in self.layers:
            i.forward_propogate()

    def backward_pass(self, answer):
        expected = [0] * 10
        expected[answer] = 1


        # For each node in the final layer, their partial is
        for i, node in enumerate(self.layers[-1].nodes):
            node.partial_node = 2 * (node.activation - expected[i])
            # print(node.partial_node)

        for i in range(len(self.layers) - 2, -1, -1):
            # print(self.layers[i])
            self.layers[i].node_backpropogate()

    def predict(self, data):
        self.reset()
        self.forward_pass(data)

        answer = self.layers[-1].get_max_node()
        self.predictions[answer] += 1

        return answer

    def train(self, data, answer):
        self.reset()
        self.forward_pass(data)
        self.backward_pass(answer)

    def update(self):
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].node_update()

    def check_result(self):
        for i, node in enumerate(self.layers[-1].nodes):
            print(i, node.activation)

    def show_layers(self):
        print(self.num_layers)
        for i in range(self.num_layers - 1):
            self.layers[i].get_changes(i, self.num_layers)

    def reset(self):
        for i in self.layers:
            i.reset()

    def get_prediction_count(self):
        return self.predictions

    def test_batch(self, data, labels, num_tests, batch_size = 10):
        for a in range(1000):
            for z in range(batch_size):
                x = a * batch_size + z
                self.train(data[x], labels[x])
            if a % 100 == 0:
                print("Batch:", a)

            self.update()

        correct = 0
        fb_correct = 0
        for i in range(num_tests):
            answer = self.predict(digit_data[i])
            if answer == digit_label[i]:
                correct += 1
                if i < 200:
                    fb_correct += 1

        print(fb_correct/200)

        return correct/num_tests

    def save(self, fname = "a"):
        result = []
        for i in self.layers:
            result.append(i.save())

        with open(fname, "w+b") as FILE:
            pickle.dump(result, FILE)

    @staticmethod
    def load_from_file(fname):
        with open(fname, "rb") as FILE:
            data = pickle.load(FILE)

        num_nodes = []
        for i in data:
            num_nodes.append(len(i))

        num_nodes.append(0)
        model = CNN(num_nodes)

        for i, layer in enumerate(data):
            model.layers[i].set(layer)

        return model
