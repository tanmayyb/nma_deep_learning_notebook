# w1d5t1:Optimization techniques

Why Optimize? What to optimize? How to Optimize?

**About Sections 3 - 7:**

| Section | Challenge (What to Optimize) | Solution (How to Optimize) |
| --- | --- | --- |
| 3 | High dimensional search | Gradient descent |
| 4 | Poor conditioning | Momentum |
| 5 | Non-convexity | Overparameterization |
| 6 | Full gradients are expensive to compute | Stochastic gradient descent, mini-batches |
| 7 | Hyperparameter tuning | Adaptive methods |

## Section 2: **Case study: successfully training an MLP for image classification**

We do a case study on MNIST to optimise our:

- 2.1 Data

- 2.2 Model

- 2.3 Loss

- 2.4 Empirical Risk Minimization\

**About 2.1: Data**

Training set, S, of n examples $S$

All Examples drawn independently and identically (iid) from same data distribution $(x_i, y_i) \sim D$

**About 2.2: Model**

-784 input neurons

-1 Hidden Layer containing 10 neurons

-10 output neurons

-Non-Linearity?

-Softmax Units in the last layer

![Fig: Our simple case study model.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled.png)

Fig: Our simple case study model.

![Fig: Contribution of each hidden layer neuron (filter) to the output after having trained the weights and the biases.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%201.png)

Fig: Contribution of each hidden layer neuron (filter) to the output after having trained the weights and the biases.

**About 2.3: Loss**

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%202.png)

Fig: Risk Function Expects loss of model on UNSEEN data distribution

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%203.png)

Fig: Empirical Surrogate (Surrogate Objective Function). Dependant on training set data. For n large enough, good approximation.

**Sections 2.1 to 2.4** 

**need TO BE CODED**

## **Section 3: High dimensional search**

*Solution: Gradient Descent*

Local Random Search algorithm

![Fig: Simple case where objective function depends only on one weight.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%204.png)

Fig: Simple case where objective function depends only on one weight.

In this simple case, we can just choose the direction with the lowest derivative. (simple heuristic). But Look at this:

![When Higher-Dimensional cases are considered, it becomes exponentially hard to minimize objective functions.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%205.png)

When Higher-Dimensional cases are considered, it becomes exponentially hard to minimize objective functions.

Here we cannot do that so we implement a random search algorithm! Due to the ‘curse of dimensionality’.

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%206.png)

But this takes forever to reach the target. Computing $J$ (objective function) is significantly cheaper than calculating $\triangledown{J(w_t)}$ (Del of $J$), but it can more efficiently handle models of many parameters. Therefore, we use gradient descent:

![Fig: gradient descent algorithm.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%207.png)

Fig: gradient descent algorithm.

![Fig: Statquest Video with Josh](https://i.ytimg.com/vi/iyn2zdALii8/maxresdefault.jpg)

Fig: Statquest Video with Josh

![Fig: Gradient Descent on our Empirical Risk function.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%208.png)

Fig: Gradient Descent on our Empirical Risk function.

![Fig: Objective function, gradient descent in 2-dimensional weight space.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%209.png)

Fig: Objective function, gradient descent in 2-dimensional weight space.

## **Section 4: Poor conditioning**

*Solution: Momentum*

> Find good step size how?
> 

[Why Momentum Really Works](https://distill.pub/2017/momentum/)

**Relation of Curvature and Learning Step Size:**

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2010.png)

**Conditioning of Multidimensional Objective:**

![Fig: Multi-Dimensionality causes problems with choosing learning rate when curvatures are different for the objective function dimensions.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2011.png)

Fig: Multi-Dimensionality causes problems with choosing learning rate when curvatures are different for the objective function dimensions.

We can only have one $\eta$ (learning rate) for our model. This is what can happen:

![Fig: Violent oscillations on the ‘sharper’ (high curvature) directions , and slow movement along the ‘flatter’ (low curvature) directions. This impacts learning progress because we never reach the optimum.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2012.png)

Fig: Violent oscillations on the ‘sharper’ (high curvature) directions , and slow movement along the ‘flatter’ (low curvature) directions. This impacts learning progress because we never reach the optimum.

**Momentum fixes this:**

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2013.png)

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2014.png)

## **Section 5: Non-convexity**

*Solution: Overparameterization*

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2015.png)

[Loss Landscape | A.I deep learning explorations of morphology & dynamics](https://losslandscape.com/)

![Fig: Surface of an Objective Function (”loss landscape”). Seriously non-convex, really hard to find the global optimum.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2016.png)

Fig: Surface of an Objective Function (”loss landscape”). Seriously non-convex, really hard to find the global optimum.

**Overparameterization:**

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2017.png)

## Section 6: Full gradients are expensive to compute

*Solution: Mini-Batches/Stochastic Gradient Descent*

![Mini-Batching: Use a few examples per step](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2018.png)

Mini-Batching: Use a few examples per step

![How should we choose the mini-batch size?](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2019.png)

How should we choose the mini-batch size?

## **Section 7: Hyperparameter Tuning**

*Solution: Adaptive methods*

### Learning Rate

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/neural-networks-3/#baby)

![Higher learning rates will decay the loss faster, but they get stuck at worse values of loss (green line). This is because there is too much "energy" in the optimization and the parameters are bouncing around chaotically, unable to settle in a nice spot in the optimization landscape.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2020.png)

Higher learning rates will decay the loss faster, but they get stuck at worse values of loss (green line). This is because there is too much "energy" in the optimization and the parameters are bouncing around chaotically, unable to settle in a nice spot in the optimization landscape.

![Examples of visualized weights for the first layer of a neural network. **Left**
: Noisy features indicate could be a symptom: Unconverged network, improperly set learning rate, very low weight regularization penalty. **Right:**
 Nice, smooth, clean and diverse features are a good indication that the training is proceeding well.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2021.png)

Examples of visualized weights for the first layer of a neural network. **Left**
: Noisy features indicate could be a symptom: Unconverged network, improperly set learning rate, very low weight regularization penalty. **Right:**
 Nice, smooth, clean and diverse features are a good indication that the training is proceeding well.

*Babysitting the learning process*

### Bulk specific guidelines for Learning Rate

**If you learn too fast:**
-you see wild variations on your loss curve
-you converge towards solutions with huge ( + or - ) weights (or NaN values)

**If you learn too slowly:**
-convergence takes forever

**A partial solution:**
-decrease the rate if your loss varies wildly;
-otherwise increase it
-Might need to go faster initially, slower later

### Weight specific Learning Rate Tuning

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2022.png)

**Adaptive Methods for Individual Learning Rates**

![‘Momentum’ (?) Optimisers](https://machinelearningmastery.com/wp-content/uploads/2017/05/Comparison-of-Adam-to-Other-Optimization-Algorithms-Training-a-Multilayer-Perceptron.png)

‘Momentum’ (?) Optimisers

Research more into how these optimisers work

**Adagrad**

**RMSprop**

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2023.png)

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2024.png)

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2025.png)

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2026.png)

## Section 9: Putting it all together (Bonus)

**Pytorch functions**

![Untitled](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2027.png)

## Bonus resources:

![Fig: [distill.pub](http://distill.pub) focuses on clarity of explaination.](w1d5t1%20Optimization%20techniques%201545e64051634a4aad88227520e5a0dc/Untitled%2028.png)

Fig: [distill.pub](http://distill.pub) focuses on clarity of explaination.