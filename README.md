# Projects-2023-24/bxa321

## Enhancing the Robustness of Neural Network-Based Intrusion Detection Systems Against DDoS Adversarial Attacks: A Study Using CICIDS2017


# Project Overview
This project aims to enhance the robustness of neural network-based intrusion detection systems (IDS) against distributed denial of service (DDoS) attacks using adversarial techniques such as FGSM, PGD, and BIM. 
The study utilises the CICIDS2017 dataset, simulating real-world network traffic with normal and malicious activity.

# Objectives
1. Evaluate the vulnerability of various deep learning architectures (MLP, CNN, RNN) to DDoS adversarial attacks in NIDS.
2. Implement and compare the effectiveness of several adversarial DDoS attack techniques \\(FGSM, PGD, and BIM) against NIDS.
3. Explore and implement adversarial training defence mechanism to enhance model robustness.

# Prerequisites
To run this project ensure all the following libraries are installed:


| **Library**   | **Purpose**                                                                                         | **Explanation**                                                                                                                                               |
|---------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[NumPy](https://numpy.org/)**     | Numerical Computations                                                                              | NumPy is used for performing efficient numerical operations, such as array manipulations and mathematical calculations. |
| **[Pandas](https://pandas.pydata.org/)**    | Data Manipulation and Analysis                                                                      | Pandas is utilized for loading, cleaning, and manipulating tabular data.|
| **[Seaborn](https://seaborn.pydata.org/)**   | Data Visualization                                                                                  | Seaborn builds on top of Matplotlib and is used for creating visually appealing statistical plots. In this project, it is used to visualize data distributions and correlations. |
| **[Matplotlib](https://matplotlib.org/)**| Plotting and Visualization                                                                          | Matplotlib is used for creating static, interactive, and animated visualizations in Python. In this project, it is utilized for plotting graphs, confusion matrices, and correlation matrices. |
| **[TensorFlow](https://www.tensorflow.org/)**| Deep Learning and Neural Networks                                                                   | TensorFlow is the core library used to build, train, and evaluate neural network models. It provides high-level APIs like Keras, making it easier to implement complex models such as CNNs and RNNs. |
| **[scikit-learn (sklearn)](https://scikit-learn.org/)**   | Machine Learning and Data Preprocessing                                                             | Scikit-learn (sklearn) is used for data preprocessing, model evaluation, and generating adversarial examples. It provides various tools for splitting datasets, scaling features, and generating classification reports. |


# The Implementation Workflow
### **Data Loading, Preprocessing, and Visualization**
The CICIDS2017 dataset is loaded in chunks to handle its large size efficiently. Preprocessing steps include handling missing values, normalizing features using **StandardScaler**, and balancing class distributions to ensure sufficient samples for training and testing.

### **Label Encoding and Data Splitting**
Categorical labels are encoded using **LabelEncoder**, and the dataset is split into training and testing sets using an **80/20 split**. This ensures that the models have sufficient data for both training and evaluation.

### **Neural Network Models**
In this project, three neural network architectures were implemented:
* **MLP (Multi-Layer Perceptron)**: A basic fully connected neural network.
* **CNN (Convolutional Neural Network)**: A more advanced model with four convolutional layers, designed to capture spatial relationships in the data.
* **RNN (Recurrent Neural Network - LSTM)**: An LSTM-based RNN to capture temporal dependencies in the data.

Each model is trained on the processed dataset and evaluated on the testing set to measure accuracy.

# **Adversarial Attack Generation and Evaluation**
The robustness of the CNN model is tested using DDoS adversarial attack techniques:
* **FGSM (Fast Gradient Sign Method):** This attack perturbs the input data based on the gradient of the loss function under different values of ϵ `0.05`, `0.1`, and `0.2`.
* **PGD (Projected Gradient Descent):** A more advanced attack that iteratively perturbs the input data. We implemented the PGD attack with ϵ values of `0.05`, `0.1`, and `0.2`, and a step size α of `0.3`, applied over 40 iterations.
* **BIM (Basic Iterative Method):** Similar to PGD but with specific tweaks to enhance the attack's effectiveness. The BIM attack was implemented with ϵ values of `0.05`, `0.1`, and `0.2`, and a step size α of `0.3`, applied.

For each attack, DDoS adversarial examples are generated, and the CNN model was evaluated to determine the accuracy and robustness against these attacks.

# **Adversarial Training**
The CNN model was retrained using the DDoS adversarial examples generated from FGSM, PGD, and BIM attacks to enhance the robustness. The effectiveness of adversarial training is evaluated by testing the CCNN model on DDoS adversarial examples after training.

# **Model Performance Analysis**
* **Training, Validation, and Test Accuracy:** The project includes detailed visualizations comparing the accuracy of the models during training and evaluation.
* **Confusion Matrices and Classification Reports:** For each model, confusion matrices and detailed classification reports are generated to provide insights into how well the models perform across different classes.

# Results
The results of the project are stored in the following files:

### Plots:
* **Malicious_Traffic_after_Sampling.png**: Distribution of traffic after sampling.
* **correlation_matrix.png**: Correlation matrix of features.
* **train_accuracy_comparison.png**: Comparison of training accuracy for different adversarial techniques.
* **val_accuracy_comparison.png**: Comparison of validation accuracy.
* **test_accuracy_comparison.png**: Comparison of test accuracy.

### CSV Files:
* **fgsm_adversarial_training_results.csv**
* **pgd_adversarial_training_results.csv**
* **bim_adversarial_training_results.csv**

# How to Run the Project

### 1. Importing Libraries
Before executing the project, import the necessary Python libraries in the Jupyter Notebook file.

### 2. Load Dataset
Ensure that the CICIDS2017 dataset is available in the same directory as the Final_MSc_Project.ipynb. [Click here to Download CICIDS2017 Dataset](https://drive.google.com/file/d/1fWLk9Yci_PtLCIzehvdPOrgaVhBtmhgi/view?usp=sharing)

### 3. Run the Project
Final_MSc_Project.ipynb

### 4. View Results
Check the generated plots and CSV files for model evaluation metrics.

# Future Perspectives
* **Additional Attack Methods:** Explore other adversarial attack techniques to further test the robustness of the models.
* **Model Optimization:** Experiment with different neural network architectures and hyperparameters for improved performance.
* **Combination of Various Defence Mechanisms:** Adversarial training combined with defensive distillation or other regularisation techniques.

# Project Repository

You can find the project repository on GitLab using the following link:

[Click here for GitLab Repository - Projects 2023-24/bxa321](https://git.cs.bham.ac.uk/bxa321/projects-2023-24-bxa321)





