# Apple Leaf Disease Classification using YOLO

This project implements a deep learning solution for classifying different diseases in apple leaves using the YOLO11 (You Only Look Once) architecture. The system is designed to identify 9 distinct classes of apple leaf health conditions with high accuracy.

## Project Overview

The project fine-tunes a pre-trained YOLO11m-cls model on a custom dataset of apple leaf images. It includes a complete pipeline for:
- Data extraction and preprocessing
- Automatic dataset restructuring for YOLO classification format
- Model training with advanced data augmentation
- Comprehensive evaluation and visualization of results

## Features

- **Automated Data Preparation**: Scripts to unzip and reorganize dataset into Train/Val/Test splits.
- **Advanced Augmentation**: Implementation of HSV adjustments, rotation, translation, scaling, mixup, and random erasing to improve model robustness.
- **Training Visualization**: Real-time generation of loss curves, accuracy metrics, and confusion matrices.
- **Detailed Evaluation**: Per-class accuracy reporting and visualization of misclassified samples.

## Requirements

The project requires Python 3.x and the following libraries:

- `ultralytics` (YOLO framework)
- `torch` & `torchvision`
- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`
- `Pillow`

## Installation

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install ultralytics matplotlib seaborn numpy pandas scikit-learn opencv-python
   ```

## Usage

The primary workflow is contained within the `unzip.ipynb` notebook.

1. **Setup**: Ensure your dataset zip file (e.g., `apple leaf.v2i.yolo26.zip`) is available. Update the `zip_path` variable in the notebook if necessary.
2. **Execution**: Open and run `unzip.ipynb` cell by cell.
   - **Step 1**: Extracts the dataset.
   - **Step 2**: Reorganizes files into the required YOLO directory structure (`train`/`val`/`test`).
   - **Step 3**: Checks for GPU availability.
   - **Step 4**: Loads the YOLO11m-cls model and starts training.
   - **Step 5**: Evaluates the model on the test set and generates performance plots.

## Project Structure

- **`unzip.ipynb`**: The main Jupyter Notebook containing the end-to-end pipeline.
- **`dataset/`**: (Generated) Directory where the dataset is extracted and organized.
- **`apple_leaf_results_runs/`**: Contains training outputs, including trained weights (`best.pt`, `last.pt`) and logs.
- **`Performance_graphs/`**: Generated visualizations of model metrics.
- **`datasecription/`**: Contains loss and metric curves.

## Results

The model outputs various metrics to evaluate performance:
- **Confusion Matrix**: To visualize classification performance across all classes.
- **Accuracy Curves**: Top-1 and Top-5 accuracy over epochs.
- **Loss Curves**: Training and validation loss trends.
