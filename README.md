
----


# **`Kidney Disease Classification Using MLFlow and DVC`**

This project demonstrates the implementation of a machine learning pipeline for classifying kidney diseases. The pipeline leverages **MLflow** for experiment tracking and **DVC (Data Version Control)** for efficient data and model management. By employing state-of-the-art tools and techniques, this project aims to streamline the development and deployment of machine learning models for healthcare applications.



## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Technologies Used](#technologies-used)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)



## **Introduction**

Kidney disease is a significant health issue worldwide. Early and accurate classification of kidney diseases can help in better treatment planning and patient management. This project focuses on building a robust classification model, ensuring reproducibility and scalability.

**Key objectives:**
- Train a machine learning model to classify kidney diseases.
- Track experiments, metrics, and artifacts using MLflow.
- Pipeline tracking using DVC to ensure reproducibility.


## Features

- **Data Processing:** Preprocesses raw kidney disease datasets for model training.
- **Model Training:** Trains and evaluates classification models.
- **Experiment Tracking:** Tracks metrics, parameters, and models using MLflow.
- **Data and Model Versioning:** Ensures reproducibility with DVC.
- **Visualization:** Provides metrics visualization to analyze model performance.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/muhammadadilnaeem/Kidney-Disease-Classification-Using-MLFlow-And-DVC.git
   cd Kidney-Disease-Classification-Using-MLFlow-And-DVC
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install DVC:
   ```bash
   pip install dvc
   ```

5. Set up MLflow:
   ```bash
   pip install mlflow
   ```


## Usage

1. **Data Versioning:**
   - Initialize DVC in the project:
     ```bash
     dvc init
     ```
   - Add data to DVC:
     ```bash
     dvc repro
     ```

2. **Tracking Experiments:**
   - Run the pipeline:
     ```bash
     python src/train.py
     ```
   - Track experiments using MLflow:
     ```bash
     mlflow ui
     ```

3. **Model Deployment:**
   - Export the trained model:
     ```bash
     mlflow models export -m runs:/<run_id>/model -o output_path

## Project Structure

```
Kidney-Disease-Classification-Using-MLFlow-And-DVC/
├── data/                    # Raw and processed datasets
├── src/                     # Source code for data preprocessing, training, and evaluation
├── models/                  # Trained models
├── reports/                 # Performance reports and visualizations
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── dvc.yaml                 # DVC pipeline configuration
```

## Technologies Used

- **Programming Language:** Python
- **Frameworks and Libraries:** pandas, numpy, tenserflow, matplotlib
- **Experiment Tracking:** MLflow
- **Data Version Control:** DVC
- **Deployment:** Streamlit (optional)


## Results

The project achieved the following performance metrics on the test dataset:
- **Accuracy:** `88%`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

------
