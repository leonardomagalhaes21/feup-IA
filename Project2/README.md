# Diabetes Prediction Project

This project implements a complete machine learning pipeline for multiclass diabetes prediction (non-diabetic, pre-diabetic, and diabetic) based on various health indicators. The pipeline includes data loading and preprocessing, problem definition, model selection and parameter tuning, model training and testing, advanced model analysis, and comprehensive evaluation of results.

## Project Structure

- `proj2.ipynb`: Jupyter notebook containing the complete machine learning pipeline
- `Dataset of Diabetes .csv`: Dataset used for training and testing the models
- `README.md`: This file with instructions and project information
- `requirements.txt`: File containing all the required Python packages

## Requirements

To run this project, you need the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- imbalanced-learn (for SMOTE implementation)

You can install all required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow imbalanced-learn eli5
```

Alternatively, you can install all dependencies at once using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## How to Execute the Program

1. Open the `proj2.ipynb` file in the Jupyter interface

2. You can execute the notebook in two ways:
   - **Cell by cell**: Click on each cell and press `Shift+Enter` to execute one cell at a time
   - **Run all**: Select "Kernel" > "Restart & Run All" to execute the entire notebook

## Machine Learning Pipeline

The notebook follows a complete machine learning pipeline:

1. **Data Loading and Preprocessing**
   - Loading the diabetes dataset
   - Handling missing values and duplicates
   - Encoding categorical variables
   - Feature scaling

2. **Problem Definition and Target Identification**
   - Defining the problem as diabetes prediction
   - Creating multiclass targets (non-diabetic, pre-diabetic, and diabetic classes)

3. **Model Selection and Parameter Tuning**
   - Implementation of multiple models:
     - Decision Tree
     - k-Nearest Neighbors (k-NN)
     - Support Vector Machine (SVM)
     - Logistic Regression
     - Neural Network
   - Hyperparameter tuning using GridSearchCV

4. **Model Training and Testing**
   - Splitting data into training (80%) and testing (20%) sets
   - Training each model on the training data
   - Making predictions on the test set

5. **Evaluation and Comparison of Results**
   - Evaluating models using multiple metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC AUC
   - Visualizing model performance
   - Addressing class imbalance with SMOTE

6. **Advanced Model Analysis**
   - Learning curves analysis (underfitting/overfitting detection)
   - ROC curves and AUC analysis for each class
   - Precision-recall curves 
   - Model calibration analysis
   - Decision boundary visualization using PCA
   - Cross-validation analysis

## Results

The notebook includes comprehensive evaluation and comparison of all implemented models, allowing you to select the best performing model for diabetes prediction. The results section includes:

- Performance metrics for each model
- Confusion matrices
- ROC curves and AUC analysis
- Precision-recall curves
- Model calibration charts
- Decision boundaries visualization
- Cross-validation results with model stability analysis
- Learning curves

## Additional Notes

- Make sure the dataset file `Dataset of Diabetes .csv` is in the same directory as the notebook
- Each section of the notebook is clearly documented with explanatory markdown cells
- You can modify the hyperparameter grid searches to explore different model configurations
- The notebook includes group information and academic context
