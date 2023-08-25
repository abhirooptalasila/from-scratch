import mnist
import numpy as np

train_images = mnist.train_images()  # [:1000]
train_labels = mnist.train_labels()  # [:1000]
test_images = mnist.test_images()  # [:1000]
test_labels = mnist.test_labels()  # [:1000]


class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # dividing by 9 to reduce variance of initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        """
        Generate all 3x3 image regions using valid padding
        """
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                imr = image[i : (i + 3), j : (j + 3)]
                yield imr, i, j

    def forward(self, inp):
        """
        Slide filter over the generated image regions
        Returns a 3D array with dims (h, w, num_filters)
        """
        self.last_inp = inp
        h, w = inp.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for imr, i, j in self.iterate_regions(inp):
            output[i, j] = np.sum(imr * self.filters, axis=(1, 2))

        return output

    def backward(self, dL_dout, lr):
        """
        Perform backward pass of conv layer
        """
        dL_dfilters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_inp):
            for f in range(self.num_filters):
                dL_dfilters[f] += dL_dout[i, j, f] * im_region

        # Update filters
        self.filters -= lr * dL_dfilters
        return None


class MaxPool2x2:
    def iterate_regions(self, image):
        """
        Generate 2x2 image regions to pool over
        """
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                imr = image[(i * 2) : (i * 2 + 2), (j * 2) : (j * 2 + 2)]
                yield imr, i, j

    def forward(self, inp):
        """
        Slide maxpool layer over given input
        Returns a 3d array with dims (h / 2, w / 2, num_filters)
        """
        self.last_inp = inp
        h, w, num_filters = inp.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for imr, i, j in self.iterate_regions(inp):
            output[i, j] = np.amax(imr, axis=(0, 1))

        return output

    def backward(self, dL_dout):
        """
        Perform backward pass
        """
        dL_dinput = np.zeros(self.last_inp.shape)

        for imr, i, j in self.iterate_regions(self.last_inp):
            h, w, f = imr.shape
            amax = np.amax(imr, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it
                        if imr[i2, j2, f2] == amax[f2]:
                            dL_dinput[i * 2 + i2, j * 2 + j2, f2] = dL_dout[i, j, f2]

        return dL_dinput


class Softmax:
    # fully-connected layer with softmax activation

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, inp):
        """
        Softmax layer is applied after dense layer
        Returns a 1d array containing respective probability values
        """
        self.last_inp_shape = inp.shape

        inp = inp.flatten()
        self.last_inp = inp
        input_len, nodes = self.weights.shape

        totals = np.dot(inp, self.weights) + self.biases
        exp = np.exp(totals)
        self.last_totals = totals
        return exp / np.sum(exp, axis=0)

    def backward(self, dl_dout, lr):
        """
        Perform backward pass on input and return loss gradient
        """
        for (
            i,
            gradient,
        ) in enumerate(dl_dout):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)

            # Gradient of output[i] against totals
            dout_dt = -t_exp[i] * t_exp / (S**2)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S**2)

            # Gradients of totals against weights/biases/input
            dt_dw = self.last_inp
            dt_db = 1
            dt_dinputs = self.weights

            # Gradients of loss against totals
            dL_dt = gradient * dout_dt

            # Gradients of loss against weights/biases/input
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt

            # Update weights / biases
            self.weights -= lr * dL_dw
            self.biases -= lr * dL_db
            return dL_dinputs.reshape(self.last_inp_shape)


def forward_pass(image, label):
    """
    Carry out a forward pass of the rough CNN
    Calculate the cross-entropy loss and accuracy
    """
    # rescale
    image = (image / 255) - 0.5
    output = conv.forward(image)
    output = pool.forward(output)
    output = softmax.forward(output)

    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0

    return output, loss, acc


def train(im, label, lr=0.005):
    """
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    """
    # Forward
    out, loss, acc = forward_pass(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backward(gradient, lr)
    gradient = pool.backward(gradient)
    gradient = conv.backward(gradient, lr)
    return loss, acc


conv = Conv3x3(8)
pool = MaxPool2x2()
softmax = Softmax(13 * 13 * 8, 10)

print("MNIST CNN initialized!")


for epoch in range(3):
    print("--- Epoch %d ---" % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                f"[Step {i + 1}] Past 100 steps: Average Loss {loss / 100:.3f} | Accuracy: {num_correct}%"
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print("\n--- Testing the CNN ---")
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward_pass(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print("Test Loss:", loss / num_tests)
print("Test Accuracy:", num_correct / num_tests)
