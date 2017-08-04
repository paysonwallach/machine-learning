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
            self.weights = np.random.normal(size=size, scale=1e-4)

        if not input_layer and not output_layer:
            self.f_prime = np.zeros((minibatch_size, size[1]))
            self.biases = np.random.normal(size=[1, size[1]], scale=1)

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
        # print "S matrix: {0}\n{1}".format(np.shape(self.s), self.s[50, :])
        # print "Z matrix: {0}\n{1}".format(np.shape(self.z), self.z[10, :])
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
        self.layers[0].z = input_data
        for i in range(self.num_layers-1):
            self.layers[i+1].s = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, y_hat, classification):
        # Calculate derivative of cost function
        self.layers[-1].delta = (y_hat - classification)

        for i in range(self.num_layers-2, 0, -1):
            self.layers[i].delta = np.dot(self.layers[i+1].delta,
                                          self.layers[i+1].weights.T) * \
                self.layers[i].f_prime

    def update_weights(self, eta):
        for i in range(1, self.num_layers):
            if i < self.num_layers-1:
                self.layers[i].biases = np.add(self.layers[i].biases,
                                               self.layers[i].delta)

                self.layers[i].del_b = -eta * np.dot(np.ones((1, 100)),
                                                     self.layers[i].delta)

            self.layers[i].del_w = -eta * np.dot(self.layers[i-1].z.T,
                                                 self.layers[i].delta)

            self.layers[i].weights = np.add(self.layers[i].weights,
                                            self.layers[i].del_w)

    def evaluate(self, training_data, validation_data, test_data,
                 minibatch_size, epochs, eta, eval_training=False,
                 eval_test=True):
        n_training = len(training_data)
        n_validation = len(validation_data)
        n_test = len(test_data)

        print "Training for {0} epochs...".format(epochs)
        for t in range(epochs):
            np.random.shuffle(training_data)

            out_str = "Epoch {0:2d}:".format(t+1)

            for i in range(n_training/minibatch_size):
                inputs, labels = create_minibatch(training_data, i,
                                                  minibatch_size,
                                                  training_data=True)
                output = self.forward_propagate(inputs)
                self.backpropagate(output, labels)
                self.update_weights(eta=eta)

            if eval_training:
                errors = 0
                for i in range(n_validation/minibatch_size):
                    inputs, labels = create_minibatch(validation_data, i,
                                                      minibatch_size)
                    output = self.forward_propagate(inputs)
                    y_hat = np.argmax(output, axis=1)
                    errors += np.sum(y_hat == labels)

                out_str = "{0} Training error: {1:.5f}".format(
                    out_str, float(errors) / n_training * 100)

            if eval_test:
                errors = 0
                for i in range(n_test/minibatch_size):
                    inputs, labels = create_minibatch(test_data, i,
                                                      minibatch_size)
                output = self.forward_propagate(inputs)
                y_hat = np.argmax(output, axis=1)
                errors += np.sum(y_hat == labels)

                out_str = "{0} Test error: {1:.5f}".format(
                    out_str, float(errors) / n_test * 100)

            print out_str


def create_minibatch(data, i, minibatch_size, training_data=False):
    inputs, labels = zip(*data)

    n = np.size(inputs[0], axis=0)

    minibatch_inputs = np.zeros((minibatch_size, n))
    if training_data:
        minibatch_labels = np.empty((minibatch_size, 10))

        for j in range(minibatch_size):
            minibatch_inputs[j, :] = inputs[i+j]
            minibatch_labels[j, :] = labels[i+j]

    else:
        minibatch_labels = np.empty(minibatch_size)

        for j in range(minibatch_size):
            minibatch_inputs[j, :] = inputs[i+j]
            minibatch_labels[j] = int(labels[i+j])

    return minibatch_inputs, minibatch_labels
