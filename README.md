🧪 Practical 1: Linear Regression using Normal Equation
📌 Objective

To implement Linear Regression using the Normal Equation method on the Iris dataset and visualize the relationship between features.

📊 Dataset Used

Iris Dataset (First 50 samples)

Features:

Sepal Length

Sepal Width

🧠 Concept Used

We used the Normal Equation:

𝑤
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑦
w=(X
T
X)
−1
X
T
y

Where:

X → Design matrix

y → Target vector

w → Weight vector (slope & intercept)

📈 Output

Slope (m): 0.7985

Intercept (c): -0.5694

Scatter plot of Sepal Length vs Sepal Width

Regression Line visualization

🛠 Libraries Used

NumPy

Matplotlib

Scikit-Learn (for dataset only)




🧪 Practical 2: Linear Regression using Scikit-Learn
📌 Objective

To perform Linear Regression using built-in Scikit-Learn model on a real-world dataset.

📊 Dataset Used

California Housing Dataset

🧠 Steps Performed

Load dataset

Split into training & testing set

Train LinearRegression model

Predict on test data

Evaluate model

📊 Evaluation Metrics

Mean Squared Error (MSE): 0.5304

R² Score: 0.6057

📈 Visualization

Scatter plot of Actual vs Predicted values

Diagonal reference line

🛠 Libraries Used

Scikit-Learn

Matplotlib

NumPy



🧪 Practical 3: Logistic Regression (Classification)
📌 Objective

To classify whether a user will purchase a product based on features like Age and Estimated Salary.

📊 Dataset Used

log.csv (Social Network Ads dataset)

Features:

Gender

Age

Estimated Salary

Target:

Purchased (0 or 1)

🧠 Steps Performed

Data preprocessing

Label Encoding

One Hot Encoding

Feature Scaling (StandardScaler)

Train Logistic Regression model

Model Evaluation

Confusion Matrix

ROC-AUC Score

Decision Boundary visualization

📊 Model Performance

Accuracy: 0.80

Precision: 0.76

Recall: 0.59

F1-Score: 0.66

ROC-AUC Score: 0.90

📈 Visualization

Confusion Matrix

Logistic Regression Decision Boundary


Practicle -4 date = 24 feb. 2026

📂 Project Description

In this project, I built a simple Neural Network using TensorFlow (Keras API) to solve the XOR problem.

The XOR problem is not linearly separable, so it requires hidden layers to learn nonlinear decision boundaries.


🧠 XOR Neural Network – TensorFlow Practice
📌 What I Learned

Built a Neural Network using TensorFlow (Keras)

Solved XOR classification problem

Used hidden layers (ReLU activation)

Applied Binary Crossentropy loss

Evaluated model using Accuracy & MSE

Plotted training loss graph

🛠 Tech Stack

Python

TensorFlow / Keras

Matplotlib

⚙️ Model Architecture

Input: 2 features

Hidden Layers: 100 → 30 neurons (ReLU)

Output: 1 neuron (Sigmoid)

📊 Training

Optimizer: Adam

Loss: Binary Crossentropy

Epochs: 200
