# Patronising and Condescending Language Detection (PCL)

This repository contains the implementation of a transformer-based model for detecting **Patronising and Condescending Language (PCL)** in text. The project was completed as part of a Natural Language Processing coursework assignment based on **SemEval 2022 Task 4: PCL Detection**.

The task is a **binary classification problem** where the goal is to determine whether a paragraph contains patronising or condescending language.

0 = No PCL  
1 = PCL

---

# Model Overview

The baseline model provided for the task uses **RoBERTa-base** with an F1 score of approximately **0.48** on the development set.

To improve performance, this project uses **DeBERTa-v3-base**, a transformer model with improved attention mechanisms. The model is fine-tuned on the PCL dataset using techniques designed to address the heavy class imbalance present in the data.

The training pipeline includes:

- Class weighting to emphasise the minority class
- Oversampling of PCL examples to approximately 30% of the training data
- Training two models using different random seeds
- Ensemble averaging of predicted probabilities
- Threshold optimisation on the development set to maximise F1 score

---

# Repository Structure

```
pcl-detection-nlp/
│
├── BestModel/
│   ├── pcl_model.ipynb
│
├── dev.txt
├── test.txt
└── README.md
```

The `BestModel` folder contains the notebook used to train the final model.

---

# Running the Model

To reproduce the model and generate predictions, run the notebook:

```
BestModel/pcl_model.ipynb
```

The notebook trains two DeBERTa models, ensembles their predictions, and selects the best classification threshold.

---

# Output Files

Running the notebook produces the following files:

```
dev.txt
test.txt
```

Each file contains one prediction per line corresponding to the model's output.

```
0 = No PCL
1 = PCL
```

These files are required for evaluation and leaderboard submission.

---

# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas / NumPy
