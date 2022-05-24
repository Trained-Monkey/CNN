from node import Node

class Layer():
    def __init__(self, nodes, num_nodes_next_layer, start = False, end = False):
        self.nodes = []
        self.num_nodes = nodes
        self.num_nodes_next_layer = num_nodes_next_layer

        self.start = start
        self.end = end

    def construct(self):
        for i in range(self.num_nodes):
            new_node = Node(self.num_nodes_next_layer, 0)

            self.nodes.append(new_node)

    def connect_nodes(self, next_layer):
        for i in self.nodes:
            i.connect(next_layer)

    def load(self, data):
        for i in range(len(data)):
            self.nodes[i].load(data[i])

    def forward_propogate(self):
        for i in self.nodes:
            i.propogate()

    def node_backpropogate(self):
        for i in self.nodes:
            i.backpropogate()

    def node_update(self):
        for i in self.nodes:
            i.update()

    def get_max_node(self):
        maximum_node = 0
        maximum_value = self.nodes[0].activation

        for i, node in enumerate(self.nodes):

            if node.activation > maximum_value:
                # print(node.activation, i)
                maximum_node = i
                maximum_value = node.activation

        return maximum_node

    def reset(self):
        for i in self.nodes:
            i.activation = 0

    def get_changes(self, i, num_layers):
        if i + 1 == num_layers:
            print("Final Layer")
        else:
            print("Layer: " + str(i))

        print("Bias: ", end = "")
        for i in self.nodes:
            print(str(i.bias) + " ", end = "")

        print()
        print("Weights: ")
        for i in self.nodes:
            print(str(i.weights) + " ", end = "")

        print("")

    def save(self):
        result = []

        for i in self.nodes:
            result.append(i.save())

        return result

    def set(self, layer):
        for i, val in enumerate(layer):
            self.nodes[i].set(val)
