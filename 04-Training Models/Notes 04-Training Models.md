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
  - y = C0 + (W1*C1 + W2*C2 + Wn*Cn)
  - *note: b is also known as the y-intercept and somtimes referred to as bias in ML*
- example from chapter 1:
  - life_satisfaction = θ 0 + θ 1 × GDP_per_capita  
- Linear Regression formula
![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/b1b58dd75017ceb4621f9272c9701169a53d918f/04-Training%20Models/Images/Eq%204-01%20Linear%20Regression%20Formula.png)
- *note on vectorized equations:*
![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/c99e801a5d8984159ceee2bc2dbb50683a52643a/04-Training%20Models/Images/Eq%204-02%20Note%20on%20vectorized%20Linear%20regression%20model%20equation.png)

### How do we train a model?
- Training means to set model parameters so that the line best fits the model training data
- For linear regression, the most common performance measure is root mean square error (RSME), so the goal of training linear regression model would be to find inputs that minimize the RSME
- In practice, it's simpler to calculate the Mean Squared Error (MSE)
- MSE equation:
![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/c99e801a5d8984159ceee2bc2dbb50683a52643a/04-Training%20Models/Images/Eq%204-03%20MSE%20Cost%20equation%20for%20linear%20regression.png)
- To simplify notations, we will just write MSE( θ ) instead of MSE( X , h θ )

### The Normal Equation
- To find the value of theta that minimizes the cost function, we will need to use a closed-form solution.  The mathematical equation for this is called the Normal Equation
- Normal Equation:
![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/c99e801a5d8984159ceee2bc2dbb50683a52643a/04-Training%20Models/Images/Eq%204-04%20Normal%20Equation.png)
- We will use the inv() function from NumPy’s Linear Algebra module ( np.linalg ) to compute the inverse of a matrix, and the dot() method for matrix multiplication
```
import numpy as np 
X = 2 * np . random . rand ( 100 , 1 ) 
y = 4 + 3 * X + np . random . randn ( 100 , 1 )

X_b = np . c_ [ np . ones (( 100 , 1 )), X ] # add x0 = 1 to each instance 
theta_best = np . linalg . inv ( X_b . T . dot ( X_b )) . dot ( X_b . T ) . dot ( y )
```
- Let’s see what the equation found:   
`>>> theta_best array([[ 4.21509616], [ 2.77011339]]) `
- we would have hoped to get 4, 3 but the added noise made it impossible to narrow down to the exact parameters
- Now we can make predictions:
```
>>> X_new = np . array ([[ 0 ], [ 2 ]]) 
>>> X_new_b = np . c_ [ np . ones (( 2 , 1 )), X_new ] # add x0 = 1 to each instance 
>>> y_predict = X_new_b . dot ( theta_best ) 
>>> y_predict 

array([[ 4.21509616], [ 9.75532293]]) 
```
### Performing linear regression using sci-kit learn
```
>>> from sklearn.linear_model import LinearRegression 
>>> lin_reg = LinearRegression () 
>>> lin_reg.fit ( X , y ) 
>>> lin_reg.intercept_ , lin_reg.coef_ 
(array([ 4.21509616]), array([[ 2.77011339]])) 

>>> lin_reg.predict ( X_new ) 
array([[ 4.21509616], [ 9.75532293]]) 
```
### Or to call least squares function directly, use scipi.linalg.lstsq:
```
>>> theta_best_svd, residuals, rank, s = np.linalg.lstsq ( X_b, y, rcond = 1e-6 )
>>> theta_best_svd
array([[4.21509616], [2.77011339]])
```

## Gradient Descent
- general idea is to tweak parameters iteratively in order to minimize cost function
- Starts in a random place, and takes a step in a random direction 
- the size of step determined by the learning rate
- Picture a bowl, and startign at the edge of the bowl, taking steps along the way until reaching the bottom of the bowl
### Gradient Descent Pitfalls
- Learning rate pitfalls:
  - learning rate too large - will bounce back and forth across the curve and never find the minimum
  - learning rate too small - will take tiny steps and never reach the minimum
- Local minimum vs global min:
  - Not all cost curves are simple bowls, some are like mountain terrain with multiple hills and valleys and peaks
  - Can potentially get stuck in a local minimum when there is a global minimum elsewhere but would have to "climb" out of the local minimum valley first
  - *note - for linear regression, the MSE function is convex, so in that case there is no risk of local vs global min*
- Scaling of features unequal
    - when using gradient descent, you should ensure that all features have a similar scale (use scikit-learns StandardScalar class) or it will take a lot longer to converge
    - to visualize, it would stretch out the terrain of gradient descent in x, y, or z directions and will take a long time to find the minimum
- Too many parameters
    - the more parameters, the harder the search is
    - with 2 parameters, it would be a 3D terrain.  With 300 parameters it's impossible to visualize 300 dimensions, but imagine trying to find the global minimum in that case

### Calculating change to cost function
- Need to calculate change to cost function for each change in weights
- To do this, you need to calculate partial derivative
- Partial derivative of the cost function (done individually):
![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/c99e801a5d8984159ceee2bc2dbb50683a52643a/04-Training%20Models/Images/Eq%204-05%20Partial%20derivative%20of%20the%20cost%20function.png)
- Vectorized partial derivative of the cost function (done on all at once):
![](https://raw.githubusercontent.com/BrianLeip/Hands_On_Machine_Learning/c99e801a5d8984159ceee2bc2dbb50683a52643a/04-Training%20Models/Images/Eq%204-06%20Gradient%20vector%20of%20the%20cost%20function.png)

### Batch Gradient Descent
- Calculates the change to cost function on the ENTIRE training set at once in batches
- can be very slow as a result

### Stochatic Gradient Descent (SGD)
- Picks a random place in the training set at every step and calculates cost using only 1 instance
- By convention we iterate by rounds of m iterations; each round is called an epoch
##### Advantages of SGD
- Makes the algorithm much faster and possible to train on huge training sets
- Has a much better chance of finding a global minimum due to the randomness, since it is less likely to get trapped in a local minimum.  
  - But at the same time it will never settle at a global minimum because it will keep bouncing around
##### Disadvantages of SGD
- SGD will never converge exactly on the minimum due to the randomness, but will be very close
- Since it picks random instances, it can be less regular / uniform / consistent compared to Batch gradient descent
  - i.e. instead of gradually moving toward the minimum, it bounces around a bit randomly, each time getting a bit closer to the min
##### Implementing SGD
- To perform Linear Regression using SGD with Scikit-Learn, you can use the SGDRegressor class, which defaults to optimizing the squared error cost function. 

## Mini-Batch Gradient Descent
- x
## TODO - CONTINUE HERE