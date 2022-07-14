# w1d3: Deep MLPs (Multi Layer Perceptron)

## Section1: Wider vs Deeper Networks

### Question of Expressivity

Universal Approximation Theorem says we can approximate a function to some accuracy with a one hidden layer neural network. But what is the difference in expressivity for shallow and deep nets?

Intuition is: more ‘wiggles’ in the function requires more neurons in 1 hidden layer.

**Sawtooth Function example:**

(Telgarsky 2015)

![s1_sawtooth.png](w1d3%20Deep%20MLPs%20(Multi%20Layer%20Perceptron)%20ccc7d60f3ff941ffaf5c2fa0bd881638/s1_sawtooth.png)

![s1_sawtooth1.png](w1d3%20Deep%20MLPs%20(Multi%20Layer%20Perceptron)%20ccc7d60f3ff941ffaf5c2fa0bd881638/s1_sawtooth1.png)

2^n linear pieces can be expressed by 3^n neurons of 2^n depth for a deep implementation, but exponential neurons for a shallow implementation.  

**Number of Monomials in sum product nets example:**

![Untitled](w1d3%20Deep%20MLPs%20(Multi%20Layer%20Perceptron)%20ccc7d60f3ff941ffaf5c2fa0bd881638/Untitled.png)

This function’s number of linear regions are exponential in depth & need exponentially many neurons. Same as for:

**Number of Monomials in sum-products nets example:**

![sec1_spn.png](w1d3%20Deep%20MLPs%20(Multi%20Layer%20Perceptron)%20ccc7d60f3ff941ffaf5c2fa0bd881638/sec1_spn.png)

refer to (Delalleau and Yoshua Bengio, Shallow vs. deep sum-product networks, NeurIPS 2011) for the math for this.

**Chaos in Deep Nets example for expressivity:**

Can any function computed by a deep neural network be efficiently approximated by a shallow network? No, not efficiently. Because:

![s1_chaos.png](w1d3%20Deep%20MLPs%20(Multi%20Layer%20Perceptron)%20ccc7d60f3ff941ffaf5c2fa0bd881638/s1_chaos.png)

**Deep Nets leverage chaos:**

(Exponential expressivity in deep neural networks through transient chaos, NeurIPS 2016)

There are examples that Randomly initialised Deep nets can compute that shallow nets cannot efficiently compute. A manifold gets heavily distorted when it passes through a high random variance-weight initialised deep net. To get the same effect on a shallow net would require Exponential number of Neurons. 

Working explanation:

small change in input results in drastic change of the output. This is the ‘signature of Chaos’.

When the variance of weights is low, the manifold passes untouched because the sigmoid units operate in the linear region. 

When the variance is high, the sigmoid units act non-linearly and the manifold gets crumbled and tangled.

 

**Deep Nets have Chaotic transition:**

![s1_weights.png](w1d3%20Deep%20MLPs%20(Multi%20Layer%20Perceptron)%20ccc7d60f3ff941ffaf5c2fa0bd881638/s1_weights.png)

Optimal tuning operates on the middle line.

(Exponential expressivity in deep neural networks through transient chaos, NeurIPS 2016)

### Disentanglement

???

something about Manifolds in the human vision (DiCarlo and Cox, Untangling invariant object recognition, Trends in Cognitive Sciences, 2007)

### Expressivity vs Learning

Training a wide shallow network is not ideal, because, a wide, shallow network can be trained to mimic a deep network, attaining significantly greater accuracy than training the shallow network directly on the data.

Ba & Caruana (2014)