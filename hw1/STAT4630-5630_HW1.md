# Homework #1

**See Canvas for this assignment due date and submission**. Complete all of the following problems. Ideally, the theoretical problems should be answered in a Markdown cell directly underneath the question. If you don't know LaTex/Markdown, you may submit separate handwritten solutions to the theoretical problems, but please see the [class scanning policy](https://docs.google.com/document/d/17y5ksolrn2rEuXYBv_3HeZhkPbYwt48UojNT1OvcB_w/edit?usp=sharing). Please do not turn in messy work. Computational problems should be completed in this notebook (using the R kernel). Computational questions may require code, plots, analysis, interpretation, etc. Working in small groups is allowed, but it is important that you make an effort to master the material and hand in your own work. 

**This assignment contains some problems that are meant to assess whether you have the background knowledge to be successful in the course. Specifically, probability and linear algebra concepts - including joint probability distributions, random vectors, conditional distributions, expectation, variance, and matrix/vector operations - are considered prerequisite concepts. We may briefly review some of these topics in class, but if you have never seen them before, and struggle with this assignment, please reach out to me so that we can assess whether you have the background knowledge to be successful in the course. Future assignments will follow the concepts covered in lecture more closely.**

## A. Theoretical Problems

### A.1 Mixture distributions

Suppose that if $$\theta = 0$$, then $$Y$$ has a normal distribution with mean $$\theta = 0$$ and standard deviation $$\sigma$$, and if $$\theta = 5$$, then $$Y$$ has a normal distribution with mean $$\theta = 5$$ and standard deviation $$\sigma$$. Also, suppose $$P(\theta = 0) = 0.5$$ and $$P(\theta = 5) = 0.5$$.


**A.1 (a) [12 points] In the first cell, for $$\sigma = 2$$, write the formula for the marginal probability density for $$Y$$. In the second (code) cell, plot the marginal probability density for $$Y$$.**

$$f_Y(y) = f_{Y, \theta = 0}(y)P(\theta = 0) + f_{Y, \theta = 5}(y)P(\theta = 5)$$, where $$P(\theta = 0) = 0.5$$ and $$P(\theta = 5) = 0.5$$.

hence, for $$\sigma = 2$$:

$$f_Y(y) = .5f_{Y, \theta = 0}(y) + .5f_{Y, \theta = 5}(y)  = .5{\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {y-0 }{\sigma }}\right)^{2}} + .5{\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {y-5 }{\sigma }}\right)^{2}} $$

$$= .5{\frac {1}{2 {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {y-0 }{2 }}\right)^{2}} + .5{\frac {1}{2 {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {y-5 }{2 }}\right)^{2}}$$,


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
f_Y = function(y) {
  0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((y - 0) / 2)^2) + 0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((y - 5) / 2)^2)
}
# https://www.educative.io/answers/how-to-use-curve-in-rd
curve(f_Y, -20, 20, xlab = "y", ylab = "f_Y(y)", col = "red")  
```


    
![png](STAT4630-5630_HW1_files/STAT4630-5630_HW1_4_0.png)
    


**A.1 (b) [10 points] What is $$P(\theta = 0|y = 1)$$, again supposing $$\sigma = 2$$?**

Use the first cell for the derivation and the subsequent cell for code to evalate.

###### YOUR ANSWER HERE:
Using Baysesian theorem:

$$P(\theta  = 0| y = 1) = \dfrac{P(y = 1 | \theta  = 0) P(\theta  = 0) }{P(y = 1)} $$

$$ = \dfrac{.5{\frac {1}{2 {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {1-0 }{2 }}\right)^{2}}}{.5{\frac {1}{2 {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {1-0 }{2 }}\right)^{2}} + .5{\frac {1}{2 {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {1-5 }{2 }}\right)^{2}}}$$


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
theta_0 =  0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((1 - 0) / 2)^2)
theta_5 =  0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((1 - 5) / 2)^2)

result = theta_0/ (theta_0 + theta_5)
print(result)
```

    [1] 0.8670358


**A.1 (c) [6 points] In the code cell, plot the posterior density of $$\theta$$, $$P(\theta = 0|y = 1)$$, for values of $$\sigma \in \{1,2,3,...,100\}$$.  Describe how the posterior density of $$\theta$$ changes in shape as $$\sigma$$ is increased and as it is decreased.**


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
posterior_density  =  function(sigma) {
  numerator =  0.5 * (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((1 - 0) / sigma)^2)
  denominator  =  0.5 * (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((1 - 0) / sigma)^2) + 
  0.5 * (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((1 - 5) / sigma)^2)
  result  =  numerator / denominator
  return(result)
}


sigma_values  = 1:100

# https://r-coder.com/sapply-function-r/
posterior_densities  = sapply(sigma_values, posterior_density)

plot(sigma_values, posterior_densities, col = 'blue')
```


    
![png](STAT4630-5630_HW1_files/STAT4630-5630_HW1_9_0.png)
    


the posterior density of $$\theta$$'s shape spread out as $$\sigma$$ is increased.
But when $$\sigma$$ decreases, the posterior density of $$\theta$$ narrowed.

### A.2 (STAT 5630 Only) Spell Check

For a six-word dictionary, this problem pretends we are dealing with a language with a 

$$$$W = \{\text{fun, sun, sit, sat, fan, for}\}.$$$$ 

An extensive study of literature written in this language reveals that all words are equally likely except that “for” is $$\alpha$$ times as likely as the other words. Further study reveals that:
- Each keystroke is an error with probability $$\theta$$.
- All letters are equally likely to produce errors.
- Given that a letter is typed incorrectly, it is equally likely to be any other letter.
- Errors are independent across letters.

For example, the probability of correctly typing “fun” (or any other word) is $$(1 − \theta)^3$$, the probability of typing “pun” or “fon” when intending to type is “fun” is $$\theta(1−\theta)^2$$, and the probability of typing “foo” or “nnn” when intending to type “fun” is $$\theta^2(1−\theta)$$. 

**A.2 (a) [2 points] Write down the prior probability of each word in $$W$$.**

Since all words are equally likely except that “for” is $$\alpha$$ times as likely as the other words. Let $$p$$ be the probability of each word except "for", then the total probability for these 6 words should be 1. 

$$5p + \alpha p  = 1 \rightarrow p  = \dfrac{1}{5 + \alpha}$$

Hence:

$$p(\text{fun}) = p(\text{sun}) =  p(\text{sit}) = p(\text{sat})  = p(\text{fan})= \dfrac{1}{5 + \alpha}$$


$$p(\text{for})= \dfrac{\alpha}{5 + \alpha}$$

**A.2 (b) [2 points] Given that someone intended to type `fun`, what is the probability that they actually typed:** 

- `sun`

- `the`


we know that the probability of correctly typing “fun” (or any other word) is $$(1 − \theta)^3$$, the probability of typing “sun” when intending to type is “fun” is $$\theta(1−\theta)^2$$, and the probability of typing “the” when intending to type “fun” is $$\theta^3$$. 
Therefore, by conditional probability,

$$p(sun|fun) =\theta(1−\theta)^2$$

$$p(the|fun) =\theta^3$$



**A.2 (c) [15 points] Given that someone typed `sun`, what is the probability that they intended to type $$w \in W$$?** 

HINT: The denominator of Bayes' theorem should be $$\sum_{w \in W}P(\text{sun} \, | \, I({w}))P(I(w))$$. $$w$$ denotes each word in $$W$$.

Leave your answer in terms of $$\theta$$ and $$\alpha$$.

 using Bayes' theorem: 
 ``hint: do the Bayes six times: fun, sun, sit, sat, fan, for``
 
 $$p(I(fun)|sun) = \dfrac{p(sun|I(fun))p(I(fun))}{p(sun)} = \dfrac{\theta(1−\theta)^2\dfrac{1}{5 + \alpha}}{p(sun)}$$
 
 $$\color{red} p(I(sun)|sun) = \dfrac{p(sun|I(sun))p(I(sun))}{p(sun)} = \dfrac{(1−\theta)^3\dfrac{1}{5 + \alpha}}{p(sun)}$$

 ``this was my previous derivation, but after I plot it in the last problem, I've realized it has exceeded the 1 thredhold, which should be impossible. therefore I changed it to the following, but I am not sure though, I need help on where I did wrong``
 
  $$\color{blue} p(I(sun)|sun) = (1-\theta)^3$$
 
 $$p(I(sit)|sun) = \dfrac{p(sun|I(sit))p(I(sit))}{p(sun)}  = \dfrac{\theta^2(1−\theta)\dfrac{1}{5 + \alpha}}{p(sun)}$$
 
 $$p(I(sat)|sun) = \dfrac{p(sun|I(sat))p(I(sat))}{p(sun)} = \dfrac{\theta^2(1−\theta)\dfrac{1}{5 + \alpha}}{p(sun)}$$

 $$p(I(fan)|sun) = \dfrac{p(sun|I(fan))p(I(fan))}{p(sun)} = \dfrac{\theta^2(1−\theta)\dfrac{1}{5 + \alpha}}{p(sun)}$$

 $$p(I(for)|sun) = \dfrac{p(sun|I(for))p(I(for))}{p(sun)} = \dfrac{\theta^3\dfrac{\alpha}{5 + \alpha}}{p(sun)}$$

 where
  $p(sun) = {\sum_{w \in W}P(\text{sun} \, | \, I({w}))P(I(w))} = {\theta(1−\theta)^2\dfrac{1}{5 + \alpha}} + {(1−\theta)^3\dfrac{1}{5 + \alpha}} + 
  {\theta^2(1−\theta)\dfrac{1}{5 + \alpha}} + {\theta^2(1−\theta)\dfrac{1}{5 + \alpha}} +{\theta^2(1−\theta)\dfrac{1}{5 + \alpha}} +  {\theta^3\dfrac{\alpha}{5 + \alpha}}$

$$ = {\dfrac{\theta(1−\theta)^2 + 3\theta^2(1−\theta) + (1−\theta)^3  + \alpha\theta^3}{5 + \alpha}}$$

**A.2 (d)  [6 points] Given that someone typed `the`, what is the probability that they intended to type $$w \in W$$?**

`question: Does this mean the prior word 'the' do not change the posterior distribution? since there is no 
theta involved`

$$p(I(fun)|the) = \dfrac{p(the|I(fun))p(I(fun))}{p(the)} = \dfrac{\theta^3\dfrac{1}{5 + \alpha}}{p(the)} = \dfrac{1}{5 + \alpha}$$
 
 $$p(I(sun)|the) = \dfrac{p(the|I(sun))p(I(sun))}{p(the)} = \dfrac{\theta^3\dfrac{1}{5 + \alpha}}{p(the)}  =\dfrac{1}{5 + \alpha}$$
 
 $$p(I(sit)|the) = \dfrac{p(the|I(sit))p(I(sit))}{p(the)}  = \dfrac{\theta^3\dfrac{1}{5 + \alpha}}{p(the)} =\dfrac{1}{5 + \alpha}$$
 
 $$p(I(sat)|the) = \dfrac{p(the|I(sat))p(I(sat))}{p(the)} = \dfrac{\theta^3\dfrac{1}{5 + \alpha}}{p(the)}=\dfrac{1}{5 + \alpha}$$

 $$p(I(fan)|the) = \dfrac{p(the|I(fan))p(I(fan))}{p(the)} = \dfrac{\theta^3\dfrac{1}{5 + \alpha}}{p(the)}=\dfrac{1}{5 + \alpha}$$

 $$p(I(for)|the) = \dfrac{p(the|I(for))p(I(for))}{p(the)} = \dfrac{\theta^3\dfrac{\alpha}{5 + \alpha}}{p(the)}=\dfrac{\alpha}{5 + \alpha}$$

 where
  $$ \cancel{p(the) = {\sum_{w \in W}P(\text{the} \, | \, I({w}))P(I(w))}   = \dfrac{(5+\alpha)\theta^3}{(5+\alpha)} = \theta^3}$$

### A.3 (STAT 4630 Only) Joint, marginal, and conditional distributions

Suppose $$X_1$$ and $$X_2$$ have the following joint pmf:

| $$x_1 \,\,\,\,  x_2$$    | $$p_{1,2}$$                       |
| ---------------------- | ------------------------------- |
| $$0 \,\,\,\, 0$$         | 0.15                            |
| $$1 \,\,\,\, 0$$         | 0.15                            |
| $$2 \,\,\,\, 0$$         | 0.15                            |
| $$0 \,\,\,\, 1$$         | 0.15                            |
| $$1 \,\,\,\, 1$$         | 0.20                            |
| $$2 \,\,\,\, 1$$         | 0.20                            |

**A.3 (a) [6 points] Find the marginal distribution of $$X_1$$.**

YOUR ANSWER HERE

**A.3 (b) [2 points] Find the marginal distribution of $$X_2$$.**

YOUR ANSWER HERE

**A.3 (c) [8 points] Find the following distributions:**

- $$X_1 \, \big| \, (X_2 = 0)$$
- $$X_1 \, \big| \, (X_2 = 1)$$
- $$X_2 \, \big| \, (X_1 = 1)$$
- $$X_2 \, \big| \, (X_1 = 2)$$

YOUR ANSWER HERE

**A.3 (d) [5 points] Are $$X_1$$ and $$X_2$$ independent? Justify your answer.**

YOUR ANSWER HERE

### A.4 Bivariate normal: inference on the correlation coefficient

Assume that $$(X,Y)$$ follow a bivariate normal distribution with $$E(X) = E(Y) = 0$$ and $$Var(X) = Var(Y) = 1$$. The pdf of $$(X,Y)$$ is given as

\begin{align*}
f(x,y \, | \, \rho) = \frac{1}{2\pi\sqrt{1-\rho^2}}\exp{\left\{-\frac{x^2 - 2\rho xy + y^2}{2(1-\rho^2)} \right\} },
\end{align*}

where $$\rho$$ is the correlation coefficient for $$X$$ and $$Y$$. Suppose that we observe $$n$$ iid realizations of $$(X,Y)$$. That is, we observe:

| X | Y |
|---|---|
| $$x_1$$ | $$y_1$$ |
| $$x_2$$ | $$y_2$$ |
| $$\vdots$$ | $$\vdots$$ |
| $$x_n$$ | $$y_n$$ |

**A.4 (a) [8 points] Find the joint pdf and the likelihood function for these observations.**

 joint pdf:
 
\begin{align*}
f(x_1,y_1,x_2,y_2,\dots, x_n,y_n \, | \, \rho) = \Pi_{i = 1}^{n}\frac{1}{2\pi\sqrt{1-\rho^2}}\exp{\left\{-\frac{x_i^2 - 2\rho x_iy_i  + y_i^2}{2(1-\rho^2)} \right\} },
\end{align*}

 likelihood function (because of iid):
\begin{align*}
L(x_1,y_1,x_2,y_2,\dots, x_n,y_n \, | \, \rho) = \Pi_{i = 1}^{n}\frac{1}{2\pi\sqrt{1-\rho^2}}\exp{\left\{-\frac{x_i^2 - 2\rho x_iy_i  + y_i^2}{2(1-\rho^2)} \right\} },
\end{align*}
 

**A.4 (b) [5 points] Assume that $$\rho \sim U(-1,1)$$. Up to a constant with respect to $$\rho$$, write down the posterior distribution for $$\rho$$ given the data $$(x_i, y_i)$$, $$i = 1,...,n$$.**

HINT: You do not have to compute the integral denominator of Bayes' theorem here!

$\pi( \rho| \, x_1,y_1,x_2,y_2,\dots, x_n,y_n ) = \dfrac{L(x_1,y_1,x_2,y_2,\dots, x_n,y_n \, | \, \rho)\pi(\rho)}{\pi( x_1,y_1,x_2,y_2,\dots, x_n,y_n )} = \begin{cases}
\dfrac{\Pi_{i = 1}^{n}\frac{1}{2\pi\sqrt{1-\rho^2}}\exp{\left\{-\frac{x_i^2 - 2\rho x_iy_i  + y_i^2}{2(1-\rho^2)} \right\} }U(\text{constant})}{\int( x_1,y_1,x_2,y_2,\dots, x_n,y_n )}, & \text{when} -1 < \rho < 1 \\
0, & \text{otherwise}
\end{cases}$

### Basic posterior inference [Extra Practice, Not Graded]

**Suppose you have a Beta$$(4, 4)$$ prior distribution on the probability $$\theta$$ that a coin will yield a ‘head’ when flipped in a specified manner. The coin is independently flipped ten times, and ‘heads’ appear fewer than $$3$$ times. You are not told how many heads were seen, only that the number is less than $$3$$. Calculate your exact posterior density (up to a proportionality constant) for $$\theta$$.**






## B. Computational Problems

### B.1 Bivariate normal: inference on the correlation coefficient revisited

Consider problem **A.4** from above. Suppose that the $$(X,Y)$$ observations are given as:


```R
#note the updated/corrected simulation
library(MASS)
set.seed(7309)

mu = c(0,0)
rho = 0
Sig = matrix(c(1,rho,rho,1), ncol = 2)
    
n = 15
X = mvrnorm(n, mu, Sig)
x = X[,1]; y = X[,2]

```

**B.1 (a) [6 points] Plot the data. Do you notice a strong correlation?**


```R
#YOUR CODE HERE
plot(x, y, main = "B.1 (a)", xlab = "X", ylab = "Y", pch = 16, col = "blue")

# fail() # No Answer - remove if you provide an answer
```


    
![png](STAT4630-5630_HW1_files/STAT4630-5630_HW1_38_0.png)
    


I do not see a strong correlation, or at all. (same as assumption: iid) 

**B.1 (b) [15 points] Plot the posterior distribution (up to a normalization constant, meaning, you can drop the denominator of Bayes' theorem), following these steps:**

1. Create a grid of $$\rho$$ values between $$-0.99$$ and $$0.99$$. Store these in `r`. 

2. Compute the prior density (uniform between $$-1$$ and $$1$$, as above) at each value of $$\rho$$.

3. Program the likelihood function for the observations given in part (a) of this problem. Remember, the likelihood function should be a function of $$\rho$$ (in this case, it will be evaluated at each value of $$\rho$$ created in step 1).

4. Compute the posterior distribution at each value of $$\rho$$.

4. Plot the posterior as a function of $$\rho$$.

 ``likelihood function``
 
\begin{align*}
L(x_1,y_1,x_2,y_2,\dots, x_n,y_n \, | \, \rho) = \Pi_{i = 1}^{n}\frac{1}{2\pi\sqrt{1-\rho^2}}\exp{\left\{-\frac{x_i^2 - 2\rho x_iy_i  + y_i^2}{2(1-\rho^2)} \right\} },
\end{align*}


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
library(MASS) # MASS: Support Functions and Datasets for Venables and Ripley's MASS
set.seed(7309)

mu = c(0,0)
rho = 0
Sig = matrix(c(1,rho,rho,1), ncol = 2)
    
n = 15
X = mvrnorm(n, mu, Sig)
x = X[,1]; y = X[,2]
#TODO: uniform constant for rho?
U_constant = 2

# 1. Create a grid of $$\rho$$ values between $$-0.99$$ and $$0.99$$. Store these in `r`. 
r = seq(-0.99, 0.99, by = 0.01)

# 2. Compute the prior density (uniform between $$-1$$ and $$1$$, as above) at each value of $$\rho$$.
prior = rep(U_constant, length(r))  

# 3. Program the likelihood function for the observations given in part (a) of this problem. 
#TODO: no product? why
likelihood = function(rho, x, y) {
  prod(1 / (2 * pi * sqrt(1 - rho^2)) * exp(-((x^2 - 2 * rho * x * y + y^2) / (2 * (1 - rho^2)))))
}

# 4. Compute the posterior distribution at each value of $$\rho$$.
posterior =  prior * sapply(r, likelihood, x = x, y = y)

# 5. Plot the posterior as a function of $$\rho$$.
plot(r, posterior, col = "blue",
     xlab = 'rho', ylab = "posterior",
)
# adding the likelihood function
lines(r, sapply(r, likelihood, x = x, y = y), col = "red")
legend("topright", legend = c("Posterior", "Likelihood"), col = c("blue", "red"), lty = 1:1)

```


    
![png](STAT4630-5630_HW1_files/STAT4630-5630_HW1_42_0.png)
    



```R
# visulizing likelihood function alone
set.seed(7309)  #42

mu = c(0,0)
rho = 0
Sig = matrix(c(1,rho,rho,1), ncol = 2)
    
n = 15
X = mvrnorm(n, mu, Sig)
x = X[,1]; y = X[,2]

r = seq(-0.99, 0.99, by = 0.01)
likelihood = function(rho, x, y) {
  prod(1 / (2 * pi * sqrt(1 - rho^2)) * exp(-((x^2 - 2 * rho * x * y + y^2) / (2 * (1 - rho^2)))))
}
likelihood_r  = sapply(r, likelihood, x = x, y = y)
plot(r, likelihood_r, col = "red",
     xlab = 'rho', ylab = "likelihood",
)
```


    
![png](STAT4630-5630_HW1_files/STAT4630-5630_HW1_43_0.png)
    


**B.1 (c) [3 points] Does the posterior distribution align with what you would expect, given these data points?**

Not really to be honest. The posterior distribution seems completely behave the same as likelihood (with the same random seed number of course), I was more looking for 'similar' hahavior not 'mirroring'.   Or I just coded it wrong.

### B.2 (STAT 5630 Only) Spell check continued

**[10 points] Refer back to problem A.2. Code the formulas that you derived with the following values:**

- (a) $$\alpha = 2$$ and $$\theta = 0.1$$.

- (b) $$\alpha = 50$$ and $$\theta = 0.1$$.

- (c) $$\alpha = 2$$ and $$\theta = 0.95$$.


**Comment on the changes you observe in these three cases.**

$$ p(sun) = denom= {\dfrac{\theta(1−\theta)^2 + 3\theta^2(1−\theta) + (1−\theta)^3  + \alpha\theta^3}{5 + \alpha}}$$


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
alpha = 2 # 50
theta = 0.1 # .95

# all six words
denom = (theta * (1 - theta)^2 + 3 * theta^2 * (1 - theta) + (1 - theta)^3 + alpha * theta^3) / (5 + alpha)
p_fun_given_sun = (theta * (1 - theta)^2) / denom 
p_sun_given_sun = ((1 - theta)^3) #TODO: why / denom makes it impossible? 
p_sit_given_sun = ((theta^2) * (1 - theta)) / denom
p_sat_given_sun = ((theta^2) * (1 - theta)) / denom
p_fan_given_sun = ((theta^2) * (1 - theta)) / denom
p_for_given_sun =((theta^3) * alpha) /denom

# Create a bar plot
dict = c("fun", "sun", "sit", "sat", "fan", "for")
probs =c(p_fun_given_sun, p_sun_given_sun, p_sit_given_sun, p_sat_given_sun, p_fan_given_sun, p_for_given_sun)

barplot(probs, names.arg = dict, main = "Probabilities given 'sun'",
        xlab = 'words',ylab = "Probability")
```


    
![png](STAT4630-5630_HW1_files/STAT4630-5630_HW1_48_0.png)
    


### comments:
When $$\theta = .95$$ it is more guareented that `for` has the dominant where `for` is always $$\alpha$$ time more likely to happen than other words. Implying when the keystroke error rate is high, `for` is the word. 


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
alpha = 2 # 50
theta = .1 # .95

# all six words
p_fun_given_the = 1 /(5 + alpha)
p_sun_given_the = 1 /(5 + alpha) #TODO: why not ((1 - theta)^3)?
p_sit_given_the = 1 /(5 + alpha)
p_sat_given_the = 1 /(5 + alpha)
p_fan_given_the = 1 /(5 + alpha)
p_for_given_the = alpha /(5 + alpha)

# Create a bar plot
dict = c("fun", "sun", "sit", "sat", "fan", "for")
probs =c(p_fun_given_the, p_sun_given_the, p_sit_given_the, p_sat_given_the, p_fan_given_the, p_for_given_the)

barplot(probs, names.arg = dict, main = "Probabilities given 'the",
        xlab = 'words',ylab = "Probability")
```


    
![png](STAT4630-5630_HW1_files/STAT4630-5630_HW1_50_0.png)
    


### comments
There is no $$\theta$$ in the distribution, implying that the word `the` is not in the dictionary, therefore has no effect on the posterior. The probability will increase when $$\alpha$$ increases, where `for` is always $$\alpha$$ time more likely to happen than others.

### B.3 Basic posterior inference [Extra practice Not Graded]

**Suppose you have a Beta(4, 4) prior distribution on the probability $$\theta$$ that a coin will yield a ‘head’ when flipped in a specified manner. The coin is independently flipped ten times, and ‘heads’ appear fewer than $$3$$ times. You are not told how many heads were seen, only that the number is less than $$3$$. Calculate your exact posterior density (up to a proportionality constant) for $$\theta$$ and sketch it.**


```R
#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer
```


    Error in fail(): could not find function "fail"
    Traceback:




```R

```
