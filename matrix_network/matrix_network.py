import numpy as np
import activation as sigma
import cost
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, size, minibatch_size, input_layer=False,
                 output_layer=False, activation=sigma.sigmoid):
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
            self.del_w = np.empty_like(self.weights)

        if not input_layer and not output_layer:
            self.f_prime = np.zeros((minibatch_size, size[1]))
            self.biases = np.zeros((1, size[1]))
            # Initilize neurons with no bias, allowing biases to be learned
            # through backpropagation
            self.del_b = np.empty_like(self.biases)

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
    def __init__(self, sizes, minibatch_size):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.layers = np.empty(self.num_layers, dtype=object)
        self.minibatch_size = minibatch_size

        print "Initilizing network..."
        for i in range(self.num_layers-1):
            if i == 0:
                print "\tInitilizing input layer of size {0}.".format(
                    sizes[i])
                self.layers[i] = Layer([sizes[i]], minibatch_size,
                                       input_layer=True)
            else:
                print "\tInitilizing hidden layer of size {0}.".format(
                    sizes[i])
                self.layers[i] = Layer([sizes[i-1], sizes[i]], minibatch_size)

        print "\tInitilizing output layer of size {0}.".format(sizes[-1])
        self.layers[-1] = Layer([sizes[-2], sizes[-1]], minibatch_size,
                                output_layer=True, activation=sigma.softmax)

        print "Done!"

    def forward_propagate(self, input_data):
        self.layers[0].z = input_data
        for i in range(self.num_layers-1):
            self.layers[i+1].s = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, y_hat, label):
        # Calculate derivative of cost function
        self.layers[-1].delta = cost.quadratic(y_hat, label)

        for i in range(self.num_layers-2, 0, -1):
            self.layers[i].delta = np.dot(self.layers[i+1].delta,
                                          self.layers[i+1].weights.T) * \
                self.layers[i].f_prime

    def update_weights(self, minibatch_size, n_training, eta, lmbd=0.1):
        for i in range(1, self.num_layers):
            if i < self.num_layers-1:
                self.layers[i].del_b = np.dot(np.ones((1, minibatch_size)),
                                              self.layers[i].delta)

                self.layers[i].biases = self.layers[i].biases - \
                    (eta / minibatch_size) * self.layers[i].del_b

            self.layers[i].del_w = np.dot(self.layers[i-1].z.T,
                                          self.layers[i].delta)

            # Apply L2 regularization to weight updates with regularization
            # rate lambda
            self.layers[i].weights = (1 - eta * (lmbd / n_training)) * \
                self.layers[i].weights - (eta / minibatch_size) * \
                self.layers[i].del_w

    def evaluate(self, training_data, validation_data, test_data,
                 minibatch_size, epochs, eta, lmbd, eval_training=True,
                 eval_test=True):
        n_training = len(training_data)
        n_validation = len(validation_data)
        n_test = len(test_data)

        cost_compare = range(epochs)
        print "Training for {0} epochs...".format(epochs)
        for t in range(epochs):
            np.random.shuffle(training_data)

            out_str = "\tEpoch {0:2d}:".format(t+1)

            for i in range(n_training/minibatch_size):
                inputs, labels = create_minibatch(training_data, i,
                                                  minibatch_size,
                                                  training_data=True)
                output = self.forward_propagate(inputs)
                self.backpropagate(output, labels)
                self.update_weights(minibatch_size, n_training, eta=eta,
                                    lmbd=lmbd)

            if eval_training:
                n_correct = 0
                for i in range(n_validation/minibatch_size):
                    inputs, labels = create_minibatch(validation_data, i,
                                                      minibatch_size)
                    output = self.forward_propagate(inputs)
                    y_hat = np.argmax(output, axis=1)
                    n_correct += np.sum(y_hat == labels)

                out_str = "{0} Training accuracy: {1:.2f}%".format(
                    out_str, float(n_correct)/n_validation * 100)

            if eval_test:
                n_correct = 0
                for i in range(n_test/minibatch_size):
                    inputs, labels = create_minibatch(test_data, i,
                                                      minibatch_size)
                    output = self.forward_propagate(inputs)
                    y_hat = np.argmax(output, axis=1)
                    n_correct += np.sum(y_hat == labels)

                out_str = "{0} Test accuracy: {1:.2f}%".format(
                    out_str, float(n_correct)/n_test * 100)

            print out_str
            cost_compare[t] = n_test - n_correct
        plt.plot(cost_compare)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.show()


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
