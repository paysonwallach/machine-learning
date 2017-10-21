import numpy as np
import activations as sigma
import costs as cost


class Layer:
    def __init__(self, size, minibatch_size, input_layer=False,
                 output_layer=False, activation=sigma.prelu):
        self.size = size
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.activation = activation
        self.z = np.zeros((minibatch_size, size[0]))

        if not input_layer:
            self.s = np.zeros((minibatch_size, size[1]))
            self.delta = np.zeros((minibatch_size, size[1]))
            # Initilize weights with gaussian distribution proportional to the
            # number of inputs of the input neuron
            self.weights = np.random.normal(size=size,
                                            scale=1/np.sqrt(size[0]))
            self.del_w = np.zeros_like(self.weights)

        if not input_layer and not output_layer:
            self.f_prime = np.zeros((minibatch_size, size[1]))
            self.biases = np.zeros((1, size[1]))
            # Initilize neurons with no bias, allowing biases to be learned
            # through backpropagation
            self.del_b = np.zeros_like(self.biases)

    def forward_propagate(self):
        if self.input_layer:  # No activation function applied to input layer
            return self.z

        if not self.input_layer and not self.output_layer:
            self.s = np.dot(self.s, self.weights)
            self.s = np.add(self.s, self.biases)
            self.f_prime = self.activation(self.s, deriv=True)

        else:
            self.s = np.dot(self.s, self.weights)

        self.z = self.activation(self.s)

        return self.z


class Network:
    def __init__(self, sizes, minibatch_size, cost=cost.cross_entropy):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.layers = np.empty(self.num_layers, dtype=object)
        self.minibatch_size = minibatch_size
        self.cost = cost

        print("Initilizing network...")
        for i in range(self.num_layers - 1):
            if i == 0:
                print("\tInitilizing input layer of size {0}.".format(
                    sizes[i]))
                self.layers[i] = Layer([sizes[i]], minibatch_size,
                                       input_layer=True)
            else:
                print("\tInitilizing hidden layer of size {0}.".format(
                    sizes[i]))
                self.layers[i] = Layer([sizes[i - 1],
                                        sizes[i]], minibatch_size)

        print("\tInitilizing output layer of size {0}.".format(sizes[-1]))
        self.layers[-1] = Layer([sizes[-2], sizes[-1]], minibatch_size,
                                output_layer=True, activation=sigma.softmax)

        print("Done!")

    def forward_propagate(self, input_data):
        self.layers[0].z = input_data
        for i in range(self.num_layers - 1):
            self.layers[i + 1].s = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, y_hat, label):
        # Calculate derivative of cost function
        self.layers[-1].delta = (self.cost).error(y_hat, label)

        for i in range(self.num_layers - 2, 0, -1):
            self.layers[i].delta = np.dot(self.layers[i + 1].delta,
                                          self.layers[i + 1].weights.T) * \
                self.layers[i].f_prime

    def update_weights(self, n_examples, eta, lmbd):
        for i in range(1, self.num_layers):
            if i < self.num_layers - 1:
                self.layers[i].del_b = \
                    np.dot(np.ones((1, self.minibatch_size)),
                           self.layers[i].delta)

                self.layers[i].biases = self.layers[i].biases - \
                    (eta / self.minibatch_size) * self.layers[i].del_b

            self.layers[i].del_w = np.dot(self.layers[i - 1].z.T,
                                          self.layers[i].delta)

            # Apply L2 regularization to weight updates with regularization
            # rate lambda
            self.layers[i].weights = (1 - eta * (lmbd / n_examples)) * \
                self.layers[i].weights - (eta / self.minibatch_size) * \
                self.layers[i].del_w

    def train(self, data, n_examples, eta, lmbd):
        inputs, labels = data
        output = self.forward_propagate(inputs)
        self.backpropagate(output, labels)
        self.update_weights(n_examples, eta=eta, lmbd=lmbd)
