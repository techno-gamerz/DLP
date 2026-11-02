# Practical 6 — Object Detection Using Transfer Learning of CNN Architectures

## Aim
To study transfer learning with a pretrained CNN architecture and implement a transfer-learning workflow using VGG16 and CIFAR-10.

## Important scope clarification
The assignment title refers to object detection, while the commonly listed workflow of loading VGG16, removing its original classification head, freezing the pretrained layers, adding a new classifier and training on CIFAR-10/Caltech-101 is an **image-classification transfer-learning workflow**. VGG16 by itself does not predict bounding boxes. The notebook therefore demonstrates transfer learning for image classification and clearly distinguishes it from full object detection. If a faculty-specific requirement explicitly demands bounding boxes, use a dedicated detector such as Faster R-CNN or SSD with a detection head and suitable annotations.

## 1. Theory

### Transfer learning
Transfer learning is a machine-learning technique in which a model trained on a source task is reused for a related target task. Training a deep CNN from scratch can require a large labelled dataset, considerable computation and long training times. A pretrained CNN already contains useful visual features learned from a large source dataset. These features can be reused and adapted to a smaller target dataset.

Typical transfer-learning steps:
1. Load a pretrained model and its weights.
2. Remove or bypass the original task-specific output layers.
3. Freeze the pretrained feature extractor initially.
4. Add new layers suitable for the target classes.
5. Train the new layers on the target dataset.
6. Evaluate the adapted model.
7. Optionally unfreeze selected deeper layers and fine-tune with a small learning rate.

### VGG16 architecture
VGG16 is a CNN architecture developed by the Visual Geometry Group at the University of Oxford. The original architecture has 13 convolutional layers and three fully connected layers with trainable weights, giving 16 weight-bearing layers. It uses small 3 × 3 convolution filters, ReLU activations and max pooling. The pretrained ImageNet version has learned features useful for visual recognition.

With `include_top=False`, the original fully connected classification head is omitted. The remaining convolutional base returns feature maps that can be connected to a new classifier.

### Freezing and fine-tuning
Freezing a model means preventing its weights from being updated during training. This preserves the learned feature extractor and reduces the number of trainable parameters. Fine-tuning means unfreezing some pretrained layers and updating their weights on the target data. Fine-tuning can improve adaptation but requires careful learning-rate selection to avoid damaging useful pretrained features.

### Image classification versus object detection
Image classification assigns a label to an image, such as “cat” or “truck.” Object detection identifies object categories and their locations, usually with bounding boxes and confidence scores. A classifier alone does not produce bounding boxes. Detection systems require an additional localization/detection architecture, such as a region-proposal network or a single-shot detector.

### CIFAR-10
CIFAR-10 contains 60,000 small colour images, each measuring 32 × 32 pixels, divided into ten classes. It is convenient for illustrating a new classification head. In this notebook, only a subset is used to keep runtime manageable.

### Input preprocessing
The pretrained VGG16 model expects inputs prepared using its specific preprocessing function. `preprocess_input` converts RGB ordering to BGR and subtracts ImageNet channel means. Using the correct preprocessing is important because the pretrained weights were learned under that input convention. The notebook resizes images to 96 × 96 and applies VGG16 preprocessing.

## 2. Code explanation
1. **Imports:** TensorFlow/Keras provides CIFAR-10, VGG16, layers and training utilities.
2. **Dataset:** CIFAR-10 is loaded and reduced to a subset of 5,000 training and 1,000 test images for a practical demonstration.
3. **Pretrained model:** `VGG16(weights="imagenet", include_top=False, input_shape=(96, 96, 3))` loads pretrained ImageNet weights without the original classifier. The first run downloads weights and therefore requires an internet connection.
4. **Freeze:** `base_model.trainable = False` prevents updates to the VGG16 feature-extraction weights.
5. **Resize:** `Resizing(96, 96)` converts 32 × 32 CIFAR-10 images to the configured VGG16 input size.
6. **Preprocessing:** `preprocess_input` performs VGG16-specific channel and mean preprocessing.
7. **Feature extraction:** The pretrained convolutional base converts the input image into feature maps.
8. **Global average pooling:** `GlobalAveragePooling2D` turns each feature map into a single value, creating a compact feature vector.
9. **Custom head:** A 128-unit ReLU layer learns target-specific combinations of features. Dropout regularizes the classifier. Ten softmax units output scores for the ten CIFAR-10 classes.
10. **Compilation:** Adam optimizes the trainable classifier weights. Sparse categorical cross-entropy is used because labels are integers from 0 to 9.
11. **Training and evaluation:** The custom head is trained for a few epochs; a separate test subset is used for evaluation.
12. **Prediction:** The model returns ten scores for a sample image, and `argmax` selects the highest-scoring class.

## 3. Algorithm
1. Import the required libraries.
2. Load CIFAR-10 and define the ten class names.
3. Select manageable training and test subsets.
4. Load VGG16 with ImageNet pretrained weights and exclude its original classifier.
5. Freeze the VGG16 convolutional base.
6. Construct a new model that resizes and preprocesses images, extracts pretrained features and applies a custom classifier.
7. Compile the model using Adam, sparse categorical cross-entropy and accuracy.
8. Train the custom classification layers on the training subset.
9. Evaluate the model on the test subset.
10. Plot training and validation accuracy/loss.
11. Predict the class of a sample image.
12. State clearly that the implementation performs image classification; full object detection requires a localization/detection head.

## 4. Result
Transfer learning is demonstrated using ImageNet-pretrained VGG16 as a frozen feature extractor. A new classifier is trained for CIFAR-10, and its performance and sample predictions are observed.

## 5. Viva questions
1. **What is transfer learning?** Reusing a model trained on one task as a starting point for another related task.
2. **What is VGG16?** A CNN with 16 weight-bearing layers, commonly used with pretrained ImageNet weights.
3. **What does `include_top=False` do?** It omits VGG16's original fully connected classification head.
4. **Why freeze the pretrained base?** To preserve learned features and reduce the number of trainable parameters.
5. **What is fine-tuning?** Unfreezing some pretrained layers and training them on target data.
6. **Why preprocess input for VGG16?** The pretrained weights expect inputs prepared according to VGG16's channel-order and mean-subtraction convention.
7. **What is ImageNet?** A large labelled image dataset used to train many image-recognition models.
8. **What is global average pooling?** It averages each feature map spatially to produce a compact vector.
9. **How does classification differ from detection?** Classification predicts image labels; detection also predicts object locations, usually bounding boxes.
10. **Is VGG16 alone a complete object detector?** No. It can be a feature extractor, but detection requires an appropriate detection head and localization mechanism.
