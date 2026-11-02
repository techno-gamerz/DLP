# Practical 1 — Study of Deep Learning Packages

## Aim
To study the commonly used deep-learning packages TensorFlow, Keras, PyTorch and Theano, and understand their roles in constructing, training and evaluating neural networks.

## 1. Introduction
Deep learning is a branch of machine learning that uses neural networks with multiple processing layers to learn representations from data. Implementing a deep-learning model requires numerical computation, tensor operations, automatic differentiation, optimization algorithms, data-handling utilities and hardware acceleration. Deep-learning frameworks provide these facilities and reduce the amount of low-level code a developer must write.

A **tensor** is a multidimensional array. A scalar is a zero-dimensional tensor, a vector is one-dimensional, a matrix is two-dimensional, and image batches or neural-network activations can be represented by higher-dimensional tensors. Neural-network parameters are adjusted during training to reduce a loss function.

## 2. TensorFlow
TensorFlow is an open-source machine-learning framework that provides tensor operations, automatic differentiation, neural-network layers, optimizers, model deployment and hardware acceleration. It supports CPU and compatible GPU execution. TensorFlow is used for research and production systems, including image classification, natural-language processing and time-series analysis.

Important features:
- Tensor operations and automatic differentiation.
- Model-building and training APIs.
- Support for CPU and supported accelerators.
- Tools for saving, loading and deploying models.
- Data input pipelines and evaluation utilities.

In this notebook, TensorFlow is checked for availability and used through its Keras API to create a small sequential neural network.

## 3. Keras
Keras is a high-level neural-network API. In current TensorFlow-based workflows, `tf.keras` provides convenient abstractions for layers, model construction, compilation, training and evaluation. Keras makes neural-network code easier to read by allowing models to be built from reusable layers.

Typical workflow:
1. Define a model and its layers.
2. Compile the model with an optimizer, loss function and metrics.
3. Train using `fit`.
4. Evaluate using `evaluate`.
5. Generate predictions using `predict`.

The example uses an input with four numerical features, a hidden Dense layer with ReLU activation and a three-neuron softmax output. The output represents scores/probabilities for three classes.

## 4. PyTorch
PyTorch is an open-source deep-learning framework popular in research and application development. It provides tensor computation, automatic differentiation, neural-network modules and GPU support on compatible hardware. Models are commonly defined using `torch.nn` modules. Training usually involves a forward pass, loss computation, gradient calculation using automatic differentiation, and an optimizer update.

In this notebook, `nn.Sequential` connects two linear layers with a ReLU activation. A sample input is passed through the model, and softmax converts the output logits into a probability distribution. This is a forward-pass demonstration; a full PyTorch training loop would additionally calculate a loss, call `backward()`, and update parameters with an optimizer.

## 5. Theano
Theano was an influential Python library for defining, optimizing and evaluating mathematical expressions, particularly those involving multidimensional arrays. It supported symbolic computation and automatic differentiation and influenced later deep-learning tools. However, Theano is discontinued and is not generally suitable for new projects using modern Python versions. For this reason, the notebook only attempts an optional import and reports whether it is available. A failed Theano import does not mean the other practical sections have failed.

## 6. TensorFlow/Keras example explained
- **Imports and version check:** The notebook reports the Python and package versions. Optional imports are wrapped in exception handling so that one missing library does not stop the entire notebook.
- **Synthetic data:** Random data are generated solely to demonstrate the model API. The matrix `X` contains 60 samples and four features per sample. The array `y` contains an integer class label from 0 to 2 for each sample.
- **Model architecture:** `Input(shape=(4,))` specifies four features. `Dense(8, activation="relu")` creates a hidden layer with eight neurons. `Dense(3, activation="softmax")` produces three class scores as probabilities.
- **Compilation:** Adam is the optimizer, sparse categorical cross-entropy is the loss for integer class labels, and accuracy is the metric.
- **Training:** `fit` performs two short epochs to demonstrate the workflow. The tiny random dataset is illustrative and is not intended to produce a meaningful classifier.
- **Prediction:** `predict` returns the model's output for one sample.

## 7. PyTorch example explained
- `nn.Linear(4, 8)` maps four input features to eight outputs.
- `nn.ReLU()` applies a non-linear activation.
- `nn.Linear(8, 3)` produces three output logits.
- `torch.randn(1, 4)` creates one synthetic input sample.
- `torch.softmax(logits, dim=1)` converts logits into probabilities across the three classes.

## 8. Comparison
| Package | Main strengths | Typical use |
|---|---|---|
| TensorFlow | Tensor computation, training and deployment ecosystem | Production ML and deep learning |
| Keras | High-level, readable model API | Rapid prototyping and model development |
| PyTorch | Flexible tensor/autograd system and neural-network modules | Research and application development |
| Theano | Historical symbolic mathematical computation | Legacy study; discontinued |

## 9. Algorithm
1. Import Python and package utilities.
2. Display the Python version.
3. Attempt to import TensorFlow, Keras, PyTorch and Theano.
4. Display available package versions and safely report missing packages.
5. If TensorFlow is available, define a small Keras model.
6. Generate a small synthetic dataset and train the model briefly.
7. If PyTorch is available, define a small feed-forward network and perform a forward pass.
8. Record the observations and compare the packages.

## 10. Result
The major deep-learning packages are studied. The notebook reports the installed versions and, where available, demonstrates basic neural-network operations using TensorFlow/Keras and PyTorch. Theano is treated as a legacy package whose availability depends on the environment.

## 11. Viva questions
1. **What is deep learning?** A machine-learning approach that learns representations through neural networks with multiple layers.
2. **What is a tensor?** A multidimensional numerical array.
3. **What is TensorFlow?** A framework for tensor computation, automatic differentiation, neural networks and deployment.
4. **What is Keras?** A high-level API for constructing and training neural networks.
5. **What is PyTorch?** A deep-learning framework with tensor operations, automatic differentiation and neural-network modules.
6. **What was Theano used for?** Symbolic mathematical computation and gradient calculation for machine learning.
7. **What is an optimizer?** An algorithm that updates trainable parameters to reduce the loss.
8. **What is a loss function?** A numerical measure of prediction error.
9. **What does ReLU do?** It returns `max(0, x)` element-wise and introduces non-linearity.
10. **Why is softmax used for multiclass classification?** It converts logits into normalized class probabilities.
