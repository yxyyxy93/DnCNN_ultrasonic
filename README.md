# Deep Image Denoising with DnCNN

This project implements a deep learning approach for image denoising using the DnCNN (Deep Convolutional Neural Network) architecture. It focuses on reducing noise from images, which is a crucial step in improving image quality in various applications such as medical imaging and photography.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Torchvision
- Scikit-learn

### Installation

Clone the repository and install the required dependencies:


## Model

The project uses the DnCNN model, which is a deep convolutional neural network known for its effectiveness in image denoising tasks. The model is defined in the `train.py` script and trained using PyTorch.

## Training the Model

To train the model, run the `train.py` script. The script includes K-fold cross-validation to ensure the robustness of the model and uses a combination of data augmentation techniques for better generalization.


### Features

- **K-Fold Cross-Validation**: Ensures model reliability by training on different subsets of the data.
- **Data Augmentation**: Includes random horizontal flips, vertical flips, and rotations for robustness.
- **Learning Rate Scheduling**: Uses step LR for adaptive learning rate during training.
- **Loss Function**: Utilizes Mean Squared Error (MSE) for training the model.

## Results

The training process outputs the model's performance metrics, including loss over epochs. These results are crucial for evaluating the effectiveness of the model in denoising images.

