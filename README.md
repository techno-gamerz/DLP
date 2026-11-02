# SPPU BE IT — Deep Learning Practicals 1–6

This archive contains six Jupyter notebooks and six corresponding detailed academic theory notes.

## Folder structure
Each practical folder contains:
- `Practical_XX_....ipynb` — executable notebook with structured code cells and explanatory headings.
- `Theory_Practical_XX.md` — academic theory, aim, code walkthrough, algorithm, result and viva questions.

## Practical list
1. Study of Deep Learning Packages — TensorFlow, Keras, PyTorch and Theano.
2. Feedforward Neural Network — MNIST handwritten-digit classification.
3. Image Classification Using CNN — CIFAR-10.
4. Anomaly Detection Using Autoencoder — self-contained synthetic signals.
5. Continuous Bag of Words (CBOW) — context-based word prediction.
6. Transfer Learning Using VGG16 — CIFAR-10 classification with a pretrained feature extractor.

## Suggested environment
Python 3.10–3.12 is a practical choice for broad TensorFlow compatibility. Install the libraries in the environment used by Jupyter:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow matplotlib numpy
```

Practical 1 also demonstrates PyTorch if it is installed. Install the correct PyTorch build for your system from the official selector: https://pytorch.org/get-started/locally/

## Run
1. Extract the ZIP.
2. Open the folder in Jupyter Notebook or JupyterLab.
3. Open a practical notebook.
4. Run cells from top to bottom.
5. Read the matching Markdown theory file for the academic write-up.

## Important runtime notes
- MNIST and CIFAR-10 are downloaded by Keras on first use, so an internet connection is required the first time.
- Practical 6 also downloads pretrained ImageNet VGG16 weights on first use.
- Practical 4 generates its own synthetic dataset and does not need a dataset download.
- Practical 5 uses a small built-in corpus.
- Practical 1 treats Theano as a legacy/optional package because it is discontinued and may not import on modern Python. Its unavailability does not block the rest of the notebook.
- The notebook examples use modest subsets/epoch counts where possible to reduce runtime. Results will vary with hardware and random initialization.
- Practical 6 is explicitly a transfer-learning image-classification demonstration. VGG16 alone does not perform bounding-box object detection.
