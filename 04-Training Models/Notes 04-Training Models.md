# CHAPTER 4: TRAINING MODELS

## Intro
- Many ways to train models to be covered in this book
- Will start by looking at a simple linear regression and 2 ways to train it
  - Using a closed form equation that directly calculates parameters that are a best fit
  - Using an iterative calculation approach called Gradient Descent (GD) that gradually tweaks input parameters to minimize the cost function.  Multiple types of Gradient descent:
    - Batch GD
    - Mini batch GD
    - Stochastic GD

## Linear Regression Training
- Simple version, converted into multiple features: 
  - y = m*x + b
  - y = b + m*x
  - y = C0 + (W1*C1 + W2*C2 + Wx*Cx)
- example from chapter 1:
  - life_satisfaction = θ 0 + θ 1 × GDP_per_capita  
- Linear Regression formula
![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/b1b58dd75017ceb4621f9272c9701169a53d918f/04-Training%20Models/Images/Eq%204-01%20Linear%20Regression%20Formula.png)
- xx
