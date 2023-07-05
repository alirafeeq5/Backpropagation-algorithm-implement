import logging
import random

class Model:

    def __init__(self, x_train, y_train, x_test, y_test, labels, eta=0.4, epochs=750, bias_flag=True, mse_threshold=0.01):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.eta = eta # Learning rate
        self.epochs = epochs
        self.x0 = 1 if bias_flag else 0
        self.mse_threshold = mse_threshold

        self.label_map = {labels[0]: -1, labels[1]: 1} # Representing classes as -1 and 1
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] # Initializing weights

        self.accuracy = -1
        self.confusion_matrix = [[0, 0], [0, 0]]

        logging.debug(f'Model initialized with: eta={eta}, epochs={epochs}, bias_flag={bias_flag}')
        logging.debug(f"Label Map: {self.label_map}")
        logging.debug(f'Starting weights: {self.weights}')


    def train(self):
        for i in range(self.epochs):
            mse = 0

            for [x1, x2], label in zip(self.x_train.values, self.y_train.values):
                t = self.label_map[label] # Set the target to -1 or 1 according to the class

                # Calculate the net value (W^T * X)
                y = (self.x0 * self.weights[0]) + (x1 * self.weights[1]) + (x2 * self.weights[2])

                # Calculate error
                e = t - y
                
                # Update the weights
                self.weights[0] += self.eta * e * self.x0
                self.weights[1] += self.eta * e * x1
                self.weights[2] += self.eta * e * x2
                
                # Calculate MSE
                mse += e**2

            mse *= 1 / len(self.y_train)
                
            logging.debug(f'Weights after epoch {i + 1}: {self.weights}, MSE: {mse}')

            if mse < self.mse_threshold:
                break

    def test(self):
        correct = 0 # A counter for the correct prediction made by the model

        matrix_index = lambda x: 0 if x == -1 else 1

        for [x1, x2], label in zip(self.x_test.values, self.y_test.values):
            t = self.label_map[label] # Set the target to -1 or 1 according to the class

            # Calculate the net value (W^T * X)
            net = (self.x0 * self.weights[0]) + (x1 * self.weights[1]) + (x2 * self.weights[2])

            # Activation function
            if net < 0:
                y = -1
            else:
                y = 1

            # Calculate error
            if t == y:
                correct += 1

            self.confusion_matrix[matrix_index(t)][matrix_index(y)] += 1

        # Return the accuracy as a percentage
        self.accuracy =  (correct / len(self.y_test)) * 100

        return self.accuracy
