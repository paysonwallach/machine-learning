import numpy as np


def f_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))


def f_softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z


class Layer:
    def __init__(self, size, minibatch_size, input_layer=False,
                 output_layer=False, activation=f_sigmoid):
        self.size = size
        # self.minibatch_size = minibatch_size
        self.input_layer = input_layer
        self.output_layer = output_layer

        # Externally-defined activation function
        self.activation = activation
        # Input matrix for the layer
        self.s = None
        # Outgoing weight matrix (n x m)
        self.weights = None
        # Bias vector of the layer (n x 1)
        self.biases = None
        # Output matrix for the layer
        self.z = np.zeros((minibatch_size, size[0]))
        # Gradient of outgoing weight matrix (n x m)
        self.del_w = None
        # Gradient of the bias vector of the layer (n x 1)
        self.del_b = None
        # Activation function derivatives for the layer (n x m)
        self.f_prime = None

        if not input_layer:
            self.s = np.zeros((minibatch_size, size[1]))
            self.delta = np.zeros((minibatch_size, size[1]))
            self.weights = np.random.randn(size[0], size[1])

        if not input_layer and not output_layer:
            self.f_prime = np.zeros((minibatch_size, size[1]))
            self.biases = np.random.randn(1, size[1])

    def forward_propagate(self):
        if self.input_layer:  # No activation function applied to input layer
            return self.z

        if not self.input_layer and not self.output_layer:
            self.s = np.dot(self.s, self.weights)
            self.s += self.biases
            self.f_prime = self.activation(self.s, deriv=True)

        else:
            self.s = np.dot(self.s, self.weights)

        self.z = self.activation(self.s)

        return self.z


class Network:
    def __init__(self, sizes, minibatch_size=100):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.layers = np.empty(self.num_layers, dtype=object)
        self.minibatch_size = minibatch_size

        print "Initilizing network..."
        for i in range(self.num_layers-1):
            if i == 0:
                print "Initilizing input layer of size {0}.".format(
                    sizes[i])
                self.layers[i] = Layer([sizes[i]], minibatch_size,
                                       input_layer=True)
            else:
                print "Initilizing hidden layer of size {0}.".format(
                    sizes[i])
                self.layers[i] = Layer([sizes[i-1], sizes[i]], minibatch_size,
                                       activation=f_sigmoid)

        print "Initilizing output layer of size {0}.".format(sizes[-1])
        self.layers[-1] = Layer([sizes[-2], sizes[-1]], minibatch_size,
                                output_layer=True, activation=f_softmax)

        print "Done!"

    def forward_propagate(self, input_data):
        for i in range(self.num_layers-1):
            print "Forward propagating through Layer {0}...".format(i)
            self.layers[i+1].s = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, y_hat, classification):
        # Calculate derivative of cost function
        self.layers[-1].delta = (y_hat - classification)

        for i in range(self.num_layers-2, 0, -1):
            print "Backpropagating through layer {0}...".format(i)

            self.layers[i].delta = np.dot(self.layers[i+1].delta,
                                          self.layers[i+1].weights.T) * \
                self.layers[i].f_prime

    def update_weights(self, eta):
        for i in range(1, self.num_layers):
            print "Updating layer {0}...".format(i)
            """ del_b = -eta * np.dot(np.ones_like(self.layers[i].biases),
                                  self.layers[i].delta) """

            del_w = -eta * np.dot(self.layers[i-1].z.T, self.layers[i].delta)

            # self.layers[i].biases += del_b
            self.layers[i].weights += del_w

    def evaluate(self, training_data, test_data, epochs, eta,
                 eval_training=False, eval_test=True):
        training_examples = training_data[0]
        n_training = len(training_examples)
        n_test = len(test_data[1])

        print "Training for {0} epochs...".format(epochs)
        for t in range(epochs):
            out_str = "Epoch {0}:".format(t+1)

            for i in range(n_training/epochs):
                inputs, labels = create_minibatch(training_data, i,
                                                  minibatch_size=100)
                output = self.forward_propagate(inputs)
                self.backpropagate(output, labels)
                self.update_weights(eta=eta)

            if eval_training:
                errors = 0
                training_inputs, training_labels = training_data
                output = self.forward_propagate(training_inputs)
                y_hat = np.argmax(output, axis=1)
                errors += np.sum(1 - training_labels[np.arange(
                                 len(training_labels)), y_hat])

                out_str = "{0} Training error: {1:.5f}".format(
                    out_str, float(errors) / n_training)

            if eval_test:
                errors = 0
                test_inputs, test_labels = test_data
                output = self.forward_propagate(test_inputs)
                y_hat = np.argmax(output, axis=1)
                errors += np.sum(1 - test_labels[np.arange(
                                 len(test_labels)), y_hat])

                out_str = "{0} Test error: {1:.5f}".format(
                    out_str, float(errors) / n_test)

            print out_str


def create_minibatch(data, i, minibatch_size):
    inputs = data[0]
    labels = data[1]

    n = np.size(inputs[0], axis=0)

    minibatch_inputs = np.zeros((n, minibatch_size))
    minibatch_labels = np.empty((minibatch_size, 10))

    for j in range(minibatch_size):
        minibatch_inputs[:, j] = inputs[:, i+j]
        minibatch_labels[j, :] = labels[i+j, :]

    return minibatch_inputs, minibatch_labels
