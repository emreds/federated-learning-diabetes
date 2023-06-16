# Federated Learning in Medical Applications 
In this repository we have one example of federated learning in medical applications.

# Dataset 
The example uses the "Diabetes Health Indicators Dataset" from Kaggle. 
For more information about the dataset:
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

# How to run the example? 
1. Clone the repository
2. Install the the environment with the following command: 
`poetry install `
3. Run the example with the following command: 
`poetry ./run.sh`

# Methodology
- Example uses logistic regression model on 10 different clients. 
- It uses `flower` framework to simulate federated learning on local machine.
- Each client has different size of imbalanced dataset points. For more information check the `dirichlet_dist.py`
- The dataset has 2 classes and 21 features. Each client has same number of features. 
- It uses `Federated Averaging` strategy. 