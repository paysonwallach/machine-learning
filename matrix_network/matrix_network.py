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
        # Activation function derivatives for the layer (n x 1)
        self.f_prime = None

        if not input_layer:
            self.s = np.zeros((minibatch_size, size[0]))
            self.delta_w = np.zeros((minibatch_size, size[0]))

        if not output_layer:
            self.weights = np.random.randn(size[0], size[1])
            self.biases = np.random.randn(1, size[1])

        if not input_layer and not output_layer:
            self.f_prime = np.zeros((size[0], minibatch_size))

    def forward_propagate(self):
        if self.input_layer:  # No activation function applied to input layer
            self.z = np.dot(self.z, self.weights)
            return self.z

        self.z = self.activation(self.s)

        if not self.input_layer and not self.output_layer:
            self.z = np.dot(self.z, self.weights)
            self.z += self.biases
            self.f_prime = self.activation(self.s, deriv=True).T

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
                self.layers[i] = Layer([sizes[i], sizes[i+1]], minibatch_size,
                                       input_layer=True)
            else:
                print "Initilizing hidden layer of size {0}.".format(
                    sizes[i])
                self.layers[i] = Layer([sizes[i], sizes[i+1]], minibatch_size,
                                       activation=f_sigmoid)

        print "Initilizing output layer of size {0}.".format(sizes[-1])
        self.layers[-1] = Layer([sizes[-1]], minibatch_size, output_layer=True,
                                activation=f_softmax)

        print "Done!"

    def forward_propagate(self, input_data):
        for i in range(self.num_layers-1):
            print "Layer {0}...".format(i)
            self.layers[i+1].s = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, y_hat, classification):
        # Calculate derivative of cost function
        print y_hat, classification
        self.layers[-1].del_w = (y_hat - classification).T
        for i in range(self.num_layers-2, 0, -1):

            self.layers[i].del_b = self.layers[i].del_w * \
                self.layers[i].f_prime

            self.layers[i].del_w = np.dot(self.layers[i].del_w,
                                          self.layers[i].weights)

    def update_weights(self, eta):
        for i in range(self.num_layers-1):
            delta_del_b = -eta * (np.dot(self.layers[i+1].del_b,
                                  self.layers[i].z)).T

            delta_del_w = -eta * (np.dot(self.layers[i+1].del_w,
                                  self.layers[i].z)).T

            self.layers[i].biases += delta_del_b
            self.layers[i].weights += delta_del_w

    def evaluate(self, training_data, test_data, epochs, eta,
                 eval_training=False, eval_test=True):
        n_training = len(training_data)
        n_test = len(test_data)

        print "Training for {0} epochs...".format(epochs)
        for t in range(epochs):
            out_str = "[{0:4d}]".format(t)

            for inputs, labels in training_data:
                output = self.forward_propagate(inputs)
                self.backpropagate(output, labels)
                self.update_weights(eta=eta)

            if eval_training:
                errors = 0
                for training_inputs, training_labels in training_data:
                    output = self.forward_propagate(training_inputs)
                    y_hat = np.argmax(output, axis=1)
                    errors += np.sum(1 - training_labels[np.arrange(
                                     len(training_labels)), y_hat])

                out_str = "{0} Training error: {1:.5f}".format(
                    out_str, float(errors) / n_training)

            if eval_test:
                errors = 0
                for test_inputs, test_labels in test_data:
                    output = self.forward_propagate(test_inputs)
                    y_hat = np.argmax(output, axis=1)
                    errors += np.sum(1 - test_labels[np.arrange(
                                     len(test_labels)), y_hat])

                out_str = "{0} Test error: {1:.5f}".format(
                    out_str, float(errors) / n_test)

            print out_str
