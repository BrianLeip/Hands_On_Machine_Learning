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

![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/e54ecd884a0bb716ad40ce02ac55dfbceb8133a9/04-Training%20Models/Eq%204.1%20Linear%20Regression%20Formula.png "Linear Regression Formula")
- xx
