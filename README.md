# Oral Cancer Detection Using Deep Learning

This repository contains a deep learning-based project for detecting oral cancer from images using various CNN architectures. The project processes a small dataset provided by a hospital, performs image augmentation, handles class imbalance, and uses pre-trained models for classification. The final model is converted to TensorFlow Lite format for deployment in a mobile app built with Flutter.

## Overview

The project implements a deep learning pipeline for oral cancer detection, including:

- **Data preprocessing** (duplicate image detection, augmentation, and splitting)
- **Model training** using various pre-trained models (ResNet101, VGG16, InceptionV3, etc.)
- **GradCAM** for model explainability
- **Deployment** of the trained model to a Flutter mobile app

## Dataset

The dataset used in this project consists of images categorized into three classes:

- **Normal**
- **Pre-cancer**
- **Oral cancer**

Note: The dataset is very small, which may cause overfitting and less generalization in models. To mitigate this, image augmentation techniques were applied.

## Data Preprocessing

The preprocessing step involves:

- **Removing duplicate images**: Duplicate images across classes were detected later in the process using image hashing (e.g., `imagehash`), and these images were moved to a separate folder for further annotation.
- **Data Augmentation**: Given the small size of the dataset, data augmentation is applied using Keras' `ImageDataGenerator` to generate additional training samples.

### Data Augmentation

Various transformations are applied, including:

- Rotation
- Width and height shifting
- Shearing
- Zooming
- Horizontal flipping

These augmentations help increase the diversity of training data.

## Model Training

The model uses pre-trained CNN architectures, including:

- **ResNet101**
- **VGG16**
- **VGG19**
- **InceptionV3**
- **DenseNet121**
- **DenseNet169**
- **DenseNet201**
- **MobileNetV2**

The models are fine-tuned on the augmented dataset. After training, the models are evaluated using accuracy and loss metrics.

## Training the Model
Models are trained on the new augmented dataset. Additionally a script is made for custom training using pre-trained models. The `train_pretrained_model.py` script allows you to train a pre-trained deep learning model. It supports model training, dataset splitting, and evaluation using various pre-trained models from Keras applications (such as ResNet, VGG, Inception, etc.).

### How to Use

To train a model, run the following command:

```bash
python train/train_pretrained_model.py --model_name <MODEL_NAME> --epochs <EPOCHS> --original_dir <ORIGINAL_DATASET_DIR> --output_dir <OUTPUT_DIR>
```

or 

```bash
python train/train_pretrained_model.py --model_name <MODEL_NAME> --epochs <EPOCHS> --dataset_dir <DATASET_DIR>
```

Arguments
- `--model_name`: Required. The name of the pre-trained model (from keras applications) to use (e.g., ResNet101, VGG16, InceptionV3). This model will be fine-tuned on your dataset.
- `--epochs`: Required. The number of epochs for training.
- `--original_dir`: The directory containing the original dataset to split into train, validation, and test sets.
- `--output_dir`: The directory where the dataset will be saved after splitting and the trained model will be stored.
- `--dataset_dir`: If you already have a dataset split into train, validation, and test directories, specify the path to this directory instead of --original_dir and --output_dir.
- `--split_ratio`: Optional. A list of two float values specifying the train-validation split ratio (default is [0.3, 0.5]).
- `--batch_size`: Optional. The batch size for training and evaluation (default is 16).
- `--learning_rate`: Optional. The learning rate for the optimizer (default is 0.0005).
- `--l2_reg`: Optional. L2 regularization factor (default is 0.005).
- `--dropout_rate`: Optional. Dropout rate for the model (default is 0.5).
- `--classes`: Optional. The number of output classes in your dataset (default is 3).

The best model (checkpoint) is saved as `<model_name>.keras` based on validation performance.

#### Example Usage
To train a ResNet101 model for 20 epochs with a dataset split ratio of 70% for training, 15% for validation, and 15% for testing:

```bash
python train/train_pretrained_model.py --model_name ResNet101 --epochs 20 --original_dir ./new_aug_dataset --output_dir ./oral_cancer_train_test_val_split --split_ratio 0.7 0.15
```

### GradCAM for Model Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which parts of the image the model focuses on when making predictions. This provides insights into model decision-making, especially important in medical applications.

## Model Deployment

### TensorFlow Lite Conversion

The trained models are converted to TensorFlow Lite format for mobile deployment. This is essential for integrating the model into a Flutter mobile application for real-time oral cancer detection.

### Flutter Mobile App

The converted `.tflite` model is used in a mobile app built with Flutter. The app provides a user-friendly interface to upload an image and get predictions (normal, pre-cancer, or oral cancer).

**To use the model in the Flutter app:**

1. Move the `model.tflite` file to the `flutter_app/assets/` folder.
2. Build and run the app using Flutter.

## Results

After training and evaluating the models, the performance is measured using accuracy and loss metrics. Each model's test accuracy is displayed along with a confusion matrix showing the true vs predicted labels. The final model, DenseNet201, provides the best performance for oral cancer detection with a test accuracy of about 91%.
