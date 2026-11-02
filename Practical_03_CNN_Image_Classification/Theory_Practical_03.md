# Practical 3 — Image Classification Using a Convolutional Neural Network

## Aim
To implement a Convolutional Neural Network (CNN) for image classification using TensorFlow/Keras and the CIFAR-10 dataset.

## 1. Theory
A Convolutional Neural Network is a deep-learning architecture designed to process grid-like data, particularly images. Unlike a conventional fully connected network, a CNN uses local receptive fields and shared convolution filters. This allows it to learn spatial patterns while using fewer parameters than a fully connected network operating directly on every pixel.

### Convolution layer
A convolution filter (kernel) slides over an image or feature map. At each position, it computes a weighted combination of nearby values. Multiple filters learn different patterns such as edges, textures and shapes. `Conv2D(32, (3, 3))` creates 32 filters, each with a 3 × 3 kernel.

### Activation function
ReLU is defined as \(f(x)=\max(0,x)\). It introduces non-linearity so that stacked layers can learn more complex relationships.

### Pooling
Max pooling downsamples feature maps by selecting the maximum value within each local window. A 2 × 2 max-pooling layer approximately halves the feature-map height and width, reducing computation and retaining strong activations.

### Flatten and Dense layers
After convolution and pooling, `Flatten` transforms the 3D feature maps into a vector. Dense layers then use the extracted features for classification.

### Dropout
Dropout randomly disables a proportion of activations during training. This regularization technique can reduce overfitting.

### CIFAR-10
CIFAR-10 contains 60,000 32 × 32 colour images across ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. The standard split has 50,000 training images and 10,000 test images.

### CNN versus FNN
A fully connected network generally flattens an image immediately and loses the explicit spatial arrangement of pixels. A CNN processes local image patterns with shared filters, preserving spatial structure through convolutional stages.

## 2. Code explanation
1. **Imports:** TensorFlow/Keras provides the model layers and CIFAR-10 dataset; NumPy and Matplotlib support labels and visualizations.
2. **Dataset loading:** CIFAR-10 returns training and test arrays. A subset is used to keep the demonstration manageable.
3. **Normalization:** Dividing pixel values by 255 scales them to [0, 1].
4. **Model input:** Each input has shape `(32, 32, 3)`.
5. **First convolution:** 32 filters detect low-level patterns. `padding="same"` preserves the feature-map dimensions at this layer.
6. **First pooling:** Reduces the spatial dimensions.
7. **Second convolution and pooling:** Learns richer features and downsamples them again.
8. **Flatten:** Converts the feature maps into a vector.
9. **Dense layer:** Learns combinations of extracted features.
10. **Dropout:** Adds regularization during training.
11. **Output:** Ten softmax units predict the ten classes.
12. **Compilation and training:** Adam updates weights to reduce sparse categorical cross-entropy. Accuracy is monitored.
13. **Evaluation and visualization:** The test set provides an independent performance estimate; plots show learning behaviour and example predictions.

## 3. Algorithm
1. Import required libraries.
2. Load the CIFAR-10 dataset.
3. Display a sample image and its label.
4. Normalize training and test images.
5. Define a CNN with convolution, ReLU, max pooling, flattening, dense and softmax output layers.
6. Compile the model using Adam, sparse categorical cross-entropy and accuracy.
7. Train the model using training data and a validation split.
8. Evaluate the model on test data.
9. Plot training and validation accuracy/loss.
10. Predict the classes of sample images and compare them with the actual labels.

## 4. Result
A CNN image-classification model is built and trained on CIFAR-10. Its test performance, learning curves and sample predictions are observed.

## 5. Viva questions
1. **What is a CNN?** A neural network that learns spatial features using convolution operations.
2. **What is a filter/kernel?** A small trainable matrix that detects patterns in local regions.
3. **Why use convolution?** It exploits local spatial structure and shares parameters across the image.
4. **What is ReLU?** An activation function \(f(x)=\max(0,x)\).
5. **What is max pooling?** A downsampling operation that selects the maximum value in a window.
6. **Why use Flatten?** To convert feature maps into a vector for Dense layers.
7. **What is dropout?** A regularization method that randomly disables activations during training.
8. **Why are there ten output neurons?** CIFAR-10 has ten classes.
9. **What is padding?** A method of adding border values to control convolution output dimensions.
10. **How is CNN different from FNN?** CNNs exploit local spatial patterns with shared filters; FNNs typically use fully connected connections.
