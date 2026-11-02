# Practical 4 — Anomaly Detection Using an Autoencoder

## Aim
To understand autoencoders and demonstrate reconstruction-error-based anomaly detection using a neural network trained on normal synthetic signals.

## 1. Theory
An **anomaly** is an observation that differs significantly from the expected pattern of data. Anomaly detection is used in applications such as equipment monitoring, network security, fraud analysis, quality control and health monitoring.

### Autoencoder
An autoencoder is a neural network trained to reconstruct its input. It contains:
- **Encoder:** Transforms the input into a lower-dimensional or otherwise compact representation.
- **Latent representation:** The internal compressed representation of the input.
- **Decoder:** Uses the representation to reconstruct the input.

For input \(x\), the autoencoder produces a reconstruction \(\hat{x}\). Training aims to minimize the difference between \(x\) and \(\hat{x}\).

### Reconstruction error
A common reconstruction metric is mean squared error (MSE):
\[
MSE(x,\hat{x}) = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2
\]
A low error indicates that the model reconstructed the observation well. If the model is trained mostly on normal data, unusual patterns may be reconstructed poorly and receive larger errors.

### Threshold-based anomaly detection
A threshold \(T\) is chosen from training reconstruction errors. A simple decision rule is:
- If reconstruction error \(e \leq T\), classify the sample as normal.
- If reconstruction error \(e > T\), flag it as an anomaly.

The notebook uses the 95th percentile of the training errors as a demonstration threshold. In real applications, this percentile should be selected based on validation data, expected false-alarm rates and the cost of missed anomalies. A percentile threshold does not guarantee any particular precision or recall.

### Synthetic data used in this practical
The notebook creates periodic sine-like signals with small random noise to represent normal behaviour. Artificial spikes are added to some test signals to represent anomalies. This makes the experiment reproducible and avoids external dataset downloads. It is a teaching demonstration, not a validated medical, industrial or financial detector.

## 2. Code explanation
1. **Imports:** NumPy generates signals and calculates errors; Matplotlib plots signals and error distributions; TensorFlow/Keras builds the autoencoder.
2. **Synthetic data generation:** `make_normal_signal` creates a noisy periodic signal with small variations. `make_anomalous_signal` adds a large spike to a normal signal.
3. **Dataset split:** The model trains only on normal signals. The test set contains both normal and anomalous examples.
4. **Visualization:** Example curves show the difference between normal and spiked signals.
5. **Architecture:** Dense layers reduce the representation from 50 input values to 32 and then 16, then expand it back through 32 units to 50 reconstructed values.
6. **Compilation:** Adam is the optimizer and MSE is the reconstruction loss.
7. **Training:** Both the input and target are `x_train`, because the task is to reconstruct each input. The model is not given anomaly labels during training.
8. **Threshold:** The model reconstructs the normal training signals; the 95th percentile of their errors becomes the threshold.
9. **Test scoring:** The model reconstructs each test signal and calculates its MSE. Errors above the threshold are labelled anomalies.
10. **Plots and predictions:** The notebook shows training loss, test reconstruction errors, the threshold and a few predicted labels.

## 3. Algorithm
1. Import the required libraries.
2. Generate normal periodic signals and anomalous signals containing artificial spikes.
3. Split normal signals into training and test portions.
4. Combine normal and anomalous test examples and retain their known labels for evaluation.
5. Build an autoencoder with an encoder, compressed representation and decoder.
6. Compile the model using Adam and mean squared error.
7. Train the autoencoder to reconstruct normal training signals.
8. Calculate reconstruction errors for the normal training signals.
9. Select an anomaly threshold from the training errors.
10. Reconstruct test signals and calculate their errors.
11. Classify samples above the threshold as anomalies.
12. Plot loss and reconstruction errors and inspect sample predictions.

## 4. Result
An autoencoder is trained to reconstruct normal synthetic signals. A threshold is derived from reconstruction errors, and unusual test signals are identified using the threshold rule.

## 5. Viva questions
1. **What is an autoencoder?** A neural network trained to reconstruct its input.
2. **What is the encoder?** The part that maps the input into a latent representation.
3. **What is the decoder?** The part that reconstructs the input from the representation.
4. **What is reconstruction error?** A numerical measure of the difference between an input and its reconstruction.
5. **Why use MSE?** It averages the squared differences between original and reconstructed values.
6. **How can an autoencoder detect anomalies?** It is trained on normal examples and flags examples with unusually high reconstruction errors.
7. **What is a threshold?** A decision boundary separating normal and anomalous scores.
8. **Why train on normal examples?** The model learns the normal data patterns, so unusual patterns may be reconstructed less accurately.
9. **What is a false positive?** A normal example incorrectly classified as anomalous.
10. **Is this synthetic demonstration ready for real-world use?** No. Real applications require representative data, validation, threshold tuning and domain-specific assessment.
