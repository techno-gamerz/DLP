# Practical 2 — Implement a Feedforward Neural Network Using Keras and TensorFlow

## Aim
To implement a feedforward neural network (FNN) using TensorFlow/Keras and classify handwritten digits from the MNIST dataset.

## 1. Theory
A feedforward neural network is an artificial neural network in which information moves from the input layer through one or more hidden layers to the output layer. In a standard feedforward network, connections do not form cycles. Each neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function.

For neuron \(j\), the operation can be represented as:
\[
z_j = \sum_i w_{ij}x_i + b_j,\qquad a_j = f(z_j)
\]
where \(x_i\) are inputs, \(w_{ij}\) are weights, \(b_j\) is the bias, and \(f\) is an activation function.

### Layers
- **Input layer:** Receives the features. In this practical, each image is 28 × 28 pixels.
- **Flatten layer:** Converts the 2D pixel grid into a one-dimensional vector of 784 values.
- **Hidden Dense layers:** Learn combinations of input features. ReLU is used to introduce non-linearity.
- **Output layer:** Contains ten neurons, one for each digit from 0 through 9. Softmax produces class probabilities.

### MNIST dataset
MNIST is a benchmark dataset of grayscale handwritten digits. It contains 60,000 training images and 10,000 test images, each with dimensions 28 × 28 pixels. The label for each image is an integer from 0 to 9.

### Data normalization
Image pixels are normally stored as integers from 0 to 255. Dividing by 255 scales the values to the interval [0, 1], which generally helps numerical stability and training. Normalization is applied exactly once to both training and test images.

### Loss, optimizer and metric
- **Sparse categorical cross-entropy:** Appropriate when the target is an integer class label, rather than a one-hot vector.
- **Stochastic Gradient Descent (SGD):** Updates model weights in the direction that reduces the loss. The learning rate controls the update step size.
- **Accuracy:** The fraction of examples for which the predicted class matches the true class.
- **Epoch:** One complete pass through the training dataset.
- **Batch size:** Number of training examples processed before a parameter update.

Training uses 90% of the training set for weight updates and reserves 10% for validation. The separate MNIST test set is used for final evaluation.

## 2. Code explanation
1. **Imports:** NumPy handles arrays, Matplotlib visualizes images and graphs, and TensorFlow/Keras supplies the dataset and model APIs.
2. **Loading data:** `mnist.load_data()` returns training and test images with corresponding labels.
3. **Visualization:** `imshow` displays a sample image and its correct label.
4. **Normalization:** `astype("float32") / 255.0` converts pixel values to floating-point numbers between 0 and 1.
5. **Model construction:** `Input(shape=(28, 28))` defines the image dimensions. `Flatten()` converts each image to 784 values. Dense layers with 128 and 64 ReLU neurons learn features. The final ten-neuron softmax layer predicts digit classes.
6. **Compilation:** SGD, sparse categorical cross-entropy and accuracy are specified.
7. **Training:** `fit` trains for five epochs with batches of 32 images and a 10% validation split.
8. **Evaluation:** `evaluate` calculates loss and accuracy on the untouched test dataset.
9. **Plots:** Training and validation accuracy/loss curves help inspect learning and possible overfitting.
10. **Prediction:** `predict` returns ten scores for each image; `argmax` selects the index of the largest score as the predicted digit.

## 3. Algorithm
1. Import required libraries.
2. Load the MNIST training and test datasets.
3. Display one sample image.
4. Normalize image pixels by dividing by 255.
5. Construct an FNN using Flatten, two hidden Dense layers and a ten-neuron output layer.
6. Compile using SGD, sparse categorical cross-entropy and accuracy.
7. Train the network and validate on a held-out portion of training data.
8. Evaluate the trained network on the test set.
9. Plot training/validation accuracy and loss.
10. Predict labels for sample test images and compare them with actual labels.

## 4. Result
A feedforward neural network is implemented and trained to classify handwritten digits. The test loss, test accuracy, learning curves and sample predictions are obtained.

## 5. Viva questions
1. **What is a feedforward neural network?** A network in which information moves from input to output without recurrent cycles.
2. **Why is Flatten used?** It converts each 28 × 28 image into a one-dimensional vector of 784 features.
3. **Why use ReLU?** It introduces non-linearity and is computationally simple.
4. **Why are there ten output neurons?** MNIST has ten digit classes.
5. **What does softmax do?** It converts output logits into normalized class probabilities.
6. **Why normalize images?** It scales pixel values and often improves training stability.
7. **What is an epoch?** One full pass through the training examples.
8. **What is batch size?** The number of examples processed per training step.
9. **What is SGD?** An optimizer that updates parameters using gradient information from batches.
10. **Why use test data?** To estimate performance on examples not used for fitting the model.
