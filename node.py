import random
from utility import sigmoid

class Node():
    def __init__(self, num_weights, bias):
        self.weights = []
        self.bias = 0
        self.activation = 0

        self.partial_bias = 0
        self.partial_weight = []
        self.partial_node = 0

        self.batch = 0

        self.next = []

    # Connect node to the next layer also sets up required weights and biases
    def connect(self, nodes):
        self.next = nodes
        self.partial_weight = [0] * len(nodes.nodes)

        for i in range(self.next.num_nodes):
            self.weights.append(random.randint(0, 100)/2550 - 50/2550)

        self.bias = random.randint(0, 100)/2550 - 50/2550

    # Loads up activation level for input nodes
    def load(self, activation):
        self.activation = float(activation)

    # Propogates activation values forward
    def propogate(self):
        self.activation += self.bias
        self.activation = sigmoid(self.activation)

        if self.next == []:
            return

        else:
            for i, node in enumerate(self.next.nodes):
                node.add(self.weights[i] * self.activation)

    def update(self):
        if self.batch == 0:
            return

        for i in range(len(self.weights)):
            self.weights[i] -= (self.partial_weight[i]/(self.batch))
        self.bias -= self.partial_bias/(self.batch)

        self.partial_bias = 0
        self.partial_weight = [0] * len(self.next.nodes)
        self.partial_node = 0
        self.batch = 0

    # Propogates error backwards
    def backpropogate(self):
        self.batch += 1
        for i, node in enumerate(self.next.nodes):
            # if node.activation > 0:

            self.partial_node += sigmoid(node.activation) * (1 - sigmoid(node.activation)) * self.weights[i] * node.partial_node
            self.partial_weight[i] += sigmoid(node.activation) * (1 - sigmoid(node.activation)) * self.activation * node.partial_node
        self.partial_bias = self.partial_node

    # Adds activation values
    def add(self, val):
        self.activation += val

    def save(self):
        return [self.weights, self.bias]

    def set(self, val):
        self.bias = val[1]
        self.weights = val[0]
