# Homework #2

**See Canvas for this assignment and due date**. Complete all of the following problems. Ideally, the theoretical problems should be answered in a Markdown cell directly underneath the question. If you don't know LaTex/Markdown, you may submit separate handwritten solutions to the theoretical problems, but please see the [class scanning policy](https://docs.google.com/document/d/17y5ksolrn2rEuXYBv_3HeZhkPbYwt48UojNT1OvcB_w/edit?usp=sharing). Please do not turn in messy work. Computational problems should be completed in this notebook (using the R kernel). Computational questions may require code, plots, analysis, interpretation, etc. Working in small groups is allowed, but it is important that you make an effort to master the material and hand in your own work. 

## A. Theoretical Problems

## A.1 [8 points] Gamma-Poisson

**Show that the Gamma distribution is a conjugate prior for data coming from the Poisson model $$X_1,...,X_n \overset{iid}{\sim} \text{Poisson}(\lambda)$$.**

Note: 

The convention in this course will be to interpret $$\Gamma(\alpha, \beta)$$ as the "shape/rate" [parameterization](https://en.wikipedia.org/wiki/Gamma_distribution): If $$Y \sim \Gamma(\alpha, \beta)$$ then the pdf of $$Y$$ is given by 

\begin{align*}
f(y \, | \, \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}y^{\alpha -1}e^{-\beta y}, \,\,\,\,\, y >0, \,\, \alpha > 0, \,\, \beta >0.
\end{align*}

Above, $$\Gamma(\alpha)$$ is the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function) of $$\alpha$$. In the Gamma distribution, $$\Gamma(\alpha)$$ is just a normalizing constant, and so you do not need to compute it explicitly (just leave it as $$\Gamma(\alpha)$$).

Gamma given:
\begin{align*}
f(y \, | \, \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}y^{\alpha -1}e^{-\beta y}, \,\,\,\,\, y >0, \,\, \alpha > 0, \,\, \beta >0.
\end{align*}
The likelihood of a sample [Poisson model](https://en.wikipedia.org/wiki/Gamma_distribution) $$X_1,...,X_n \overset{iid}{\sim} \text{Poisson}(\lambda)$$
 is 
 $$f(\mathbf{x}| \lambda) = \prod_{i = 1}^n\dfrac{e^{-\lambda}\lambda^{x_i}}{x_i!}$$
 Hence the posterior 
 $$f(\lambda | X1,\cdots, X_n) \propto f(\lambda \, | \, \alpha, \beta) f( X1,\cdots, X_n| \lambda)  $$
 
$$ = \frac{\beta^\alpha}{\Gamma(\alpha)}\lambda^{\alpha -1}e^{-\beta \lambda}\prod_{i = 1}^n\dfrac{e^{-\lambda}\lambda^{X_i}}{X_i!}$$
 $$\propto \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{(\alpha -1 + \sum_{i =1}^{n}X_i)}e^{(-\beta \lambda - n\lambda)}$$
 
 which is a gamma by similar parameters

## A.2 [12 points] Predictive Distributions

Consider two coins, $$C_1$$ and $$C_2$$, with the following characteristics: 

- $$P(H\,| \, C_1) = 0.6$$ 
- $$P(H\, |\, C_2) = 0.4$$ 

**Choose one of the coins at random and imagine flipping it repeatedly. Given that the first two flips from the chosen coin are tails, what is the expectation of the number of additional flips until a head shows up?**

Some useful hints:

- It may be useful at some point to note that the geometric random variable, $$X = $$ "the number of flips until one H" has expectation $$E(X \, | \, p) = 1/p$$, where $$p$$ is the probability of $$H$$. 

- The [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation) implies that, for a partition $$A_1$$ and $$A_2$$, and any random variable $$X$$, 

$$$$E(X) = E(X \, | \, A_1)P(A_1) + E(X \, | \, A_2)P(A_2).$$$$

- Further, for any other event $$E$$: 

$$$$E(X \, | \, E) = E(X \, | \,E, A_1)P(A_1 \, | \, E) + E(X \, | \, E,A_2)P(A_2 \, | \, E).$$$$

we know from $$X = $$ "the number of flips until one H":

$$P(H\,| \, C_1) = 0.6$$  
$$P(H\, |\, C_2) = 0.4$$ 

$$P(C_1) = P(C_2) = .5$$

Also from $$E(X \, | \, E) = E(X \, | \,E, A_1)P(A_1 \, | \, E) + E(X \, | \, E,A_2)P(A_2 \, | \, E).$$

 we get: $$E(X \, | \, TT) = E(X \, | \,TT, C_1)P(C_1 \, | \, TT) + E(X \, | \, TT,C_2)P(C_2 \, | \, TT)  = E(X \, | \, C_1)P(C_1 \, | \, TT) + E(X \, | \,C_2)P(C_2 \, | \, TT).$$
 
where

$$E(X \, | \, C_1) = 1/P(H\,| \, C_1) = 1/.6$$

$$E(X \, | \, C_1) = 1/P(H\,| \, C_1) = 1/.4$$

and

$$P(C_1 \, | \, TT) = \dfrac{P(TT \, | \, C_1)P(C_1)}{P(TT \, | \, C_1)P(C_1) + P(TT \, | \, C_2)P(C_2)}  = \dfrac{(1-.6)^2 .5}{(1-.6)^2 .5 + (1-.4)^2 .5} \approx .3$$

$$P(C_2 \, | \, TT) = \dfrac{P(TT \, | \, C_2)P(C_2)}{P(TT \, | \, C_1)P(C_1) + P(TT \, | \, C_2)P(C_2)}  = \dfrac{(1-.4)^2 .5}{(1-.6)^2 .5 + (1-.4)^2 .5} \approx .7$$

Plug all in:

$$E(X \, | \, TT)  = 1/.6*.3 + 1/.4*.7 \approx 2.3$$

## A.3 [5 points] Negative binomial conjugate prior

**Let $$X$$ represent the number of failures observed before $$r$$ successes are observed in a set of independent Bernoulli trials. That is, let $$X \sim NB(r, p)$$ with $$r$$ known and $$p$$ unknown. In the first classwork assignment (the problem on stopping rules), we saw that, if $$p \sim beta(\alpha, \beta)$$, then the posterior distribution for $$p \, | \, x$$ is $$beta(\alpha + r, \beta + x)$$. Prove it!**

We know from Unit 1
\begin{align*}
f(x^* \, | \, \mathbf{x}) = \frac{\Gamma(A,B)}{\Gamma(A)\Gamma(B)} {n^* \choose x^* } \frac{\Gamma(A + x^*)\Gamma(B + n^* - x^*)}{\Gamma(A + B + n^*)}
\end{align*}

[PDF](https://en.wikipedia.org/wiki/Beta_distribution) $$ = {\displaystyle {\frac {p^{\alpha -1}(1-p)^{\beta -1}}{\mathrm {B} (\alpha ,\beta )}}\!}$$

Likelihood $$ = {x + r -1 \choose x }p^r(1-p)^x$$

Therefore Posterior $$ \propto {x + r -1 \choose x }p^r(1-p)^x {\displaystyle {\frac {p^{\alpha -1}(1-p)^{\beta -1}}{\mathrm {B} (\alpha ,\beta )}}\!}  \propto {x + r -1 \choose x }{\displaystyle {\frac {p^{\alpha  +r -1 }(1-p)^{\beta + x-1}}{\mathrm {B} (\alpha ,\beta )}}\!}  = beta(\alpha + r, \beta + x)$$

## B. Computational Problems

## B.1 Rate of wildfires <span style="color: #CFB87C;">(STAT 5630 Students Only, 20 points)</span>

Imagine that, over the past $$50$$ years Colorado has experienced an average of $$\lambda_0 = 150$$ large wildfires per year. For the next $$10$$ years you will record the number of large fires in Colorado and then fit a Poisson/gamma model to these data. Let the rate of large fires in this future period, $$\lambda$$, have prior $$\lambda ∼ \Gamma(a,b)$$. 

**B.1 (a) [10 points] Select $$a$$ and $$b$$ so that the prior is "uninformative" with prior variance around $$100$$ and gives equal probability on the hypotheses**

$$$$H_0: \lambda \le 150, \,\,\,\,\,\,\,\, H_1: \lambda > 150.$$$$


Note that, for $$\lambda \sim \Gamma(a,b)$$, $$Var(\lambda) = \frac{a}{b^2}$$, but that the median of $$\lambda$$ does not have a nice closed form formula; so, you may do a grid search to find $$a$$ and/or $$b$$. 


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
prior_variance  = 100

```


    Error in fail(): could not find function "fail"
    Traceback:



**B.1 (b) [10 points] Ten years have gone by and the vector `x` below represents the number of large fires in Colorado each year for these ten years. Plot the posterior distribution for these data and the prior computed in B.1 (a). Also, compute a $$90\%$$ credible interval. Do you have reason to believe that the rate of wildfires is different across these ten years when compared to past data?**


```R
x = c(145, 142, 158, 143, 185, 164, 141, 148, 151, 150)
```


```R
#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer
```

YOUR ANSWER HERE

## B.2 Normal conjugate example

Suppose Company B is developing a new type of LED light bulb and wants to estimate the average lifespan of these bulbs. The company believes that the true average lifespan, denoted as $$ \mu $$, of these new LED bulbs is different from the industry standard of $$10,000$$ hours. Company B's initial estimates about $$ \mu $$ can be represented by a normal distribution, centered at $$ \theta = 10,000 $$ hours, with a standard deviation $$ \tau = 1,000 $$. That is, $$ \mu \sim N(10,000, 1,000^2) $$.

Company B decides to test $$ n = 10 $$ randomly selected LED bulbs to gather more data. Denote the lifespan of bulbs in a possible sample as $$ X_1, ..., X_n $$. Assume that the standard deviation of the bulb lifespans is $$ \sigma = 200 $$ hours. For this particular sample, imagine that the data $$x_1,...,x_n$$ were measured as `x` below (which I am simulating).


```R
#This cell simulates data with known parameters. We'll imagine that we don't know the true values 
# of the parameters that generate the data. We'll try to infer them from Bayesian inference.

set.seed(1181)
n = 10; mu_true = 12000; sig_true = 200; #sample size and true parameters for the data distribution
x = round(rnorm(n,mu_true,sig_true),1) #simulated data, centered at 12,000


mean(x)
var(x)
```


12011.71



48986.6098888888



```R
x
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>11940.8</li><li>11795.5</li><li>11954.5</li><li>12419.6</li><li>12367.9</li><li>12058.7</li><li>11847.2</li><li>11766.7</li><li>11954.4</li><li>12011.8</li></ol>



**B.2 (a) [6 points] Compute the posterior distribution mean and variance of $$\mu \, | \, \mathbf{x}$$. What would be a reasonable point estimate for $$\mu$$ given the data?**


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
set.seed(1181)
n = 10; mu_true = 12000; sig_true = 200; #sample size and true parameters for the data distribution
x = round(rnorm(n,mu_true,sig_true),1) #simulated data, centered at 12,000
x_bar = mean(x)
tau = 100
theta = 10000

posterior_mean = (theta / tau^2 + n * x_bar / sig_true^2) / (1 / tau^2 + n / sig_true^2)
posterior_variance  = 1 / (1 / tau^2 + n / sig_true^2)  #TODO: too small

# posterior_mean
cat("Posterior Mean is :", posterior_mean, "\n")
cat("Posterior Variance is :", posterior_variance, "\n")
```

    Posterior Mean is : 11436.94 
    Posterior Variance is : 2857.143 


[posterior](https://www.statlect.com/fundamentals-of-statistics/normal-distribution-Bayesian-estimation) = 
$$\prod\dfrac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x_i-\mu)^2}{2\sigma^2}} \times \dfrac{1}{\sqrt{2\pi \tau^2}}e^{-\frac{(\mu-\theta)^2}{2\tau^2}} \propto e^{-\frac{(x_i-\mu)^2}{2\sigma^2}-\frac{(\mu-\theta)^2}{2\tau^2}}?$$

Reasonable estimate is somewhere around 11000

**B.2 (b) [6 points] Plot the prior distribution and posterior distribution (on the same plot). What do you notice about this plot?**


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
library(ggplot2)
#https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/seq
# x_ = seq(theta - 3*tau, theta + 3*tau, length.out = 100)  TODO:  the range is too small 
x_ = seq(8000, 12000, length.out = 100)
#R
prior_density = dnorm(x_, mean = theta, sd = tau)
posterior_density = dnorm(x_, mean = posterior_mean, sd = sqrt(posterior_variance))

#HW1:
# plot(r, posterior, col = "blue",
#      xlab = 'rho', ylab = "posterior",
# )
# # adding the likelihood function
# lines(r, sapply(r, likelihood, x = x, y = y), col = "red")
# legend("topright", legend = c("Posterior", "Likelihood"), col = c("blue", "red"), lty = 1:1)

plot(x_, prior_density, col = "blue",
     xlab = 'x', ylab = "prior", ylim = c(0, max(c(prior_density, posterior_density)))  # ylim!
)
# adding the likelihood function
lines(x_, posterior_density, col = "red")
legend("topright", legend = c("Prior", "Posterior"), col = c("blue", "red"), lty = 1:1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_22_0.png)
    


They are both normal, which reflects posterior of normal is normal.  I think my posterior_variance was wrong from the previous section, this seems to be too much of a shift.

**B.2 (c) [3 points] Compute a $$95\%$$ credible interval for $$\mu$$ given the data. Interpret this interval. In particular, do we have evidence to suggest that $$\mu > 10,000$$?**

To obtain 95% confidence intervals for a normal distribution with known variance, you take the mean and add/subtract \displaystyle 1.96\times standard\ error. This is because 95% of the values drawn from a normally distributed sampling distribution lie within 1.96 standard errors from the sample mean.



```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
lower_bound = posterior_mean - 1.96 * sqrt(posterior_variance)
upper_bound = posterior_mean + 1.96 * sqrt(posterior_variance)
lower_bound
upper_bound
```


11332.169307456



11541.7021211154


$$[11332, 11541] > 10000$$ with 95 confidence, I say we do have the evidence.

**B.2 (d) [6 points] Now, suppose instead that prior to the experiment, researchers believed that $$\theta = 5000$$ was the center of the prior distribution, with $$\tau = 100$$. Compute the posterior distribution for this prior (for the same data as above), reproduce the plots from above, and describe the difference between this posterior and the one from previous parts.**


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
set.seed(42)
n = 10; mu_true = 12000; sig_true = 200; #sample size and true parameters for the data distribution
x = round(rnorm(n,mu_true,sig_true),1) #simulated data, centered at 12,000
x_bar = mean(x)
tau = 100
theta = 5000

posterior_mean = (theta / tau^2 + n * x_bar / sig_true^2) / (1 / tau^2 + n / sig_true^2)
posterior_variance  = 1 / (1 / tau^2 + n / sig_true^2)  #TODO: too small

# posterior_mean
cat("Posterior Mean is :", posterior_mean, "\n")
cat("Posterior Variance is :", posterior_variance, "\n")

x_ = seq(2000, 12000, length.out = 100)
#R
prior_density = dnorm(x_, mean = theta, sd = tau)
posterior_density = dnorm(x_, mean = posterior_mean, sd = sqrt(posterior_variance))
plot(x_, prior_density, col = "blue",
     xlab = 'x', ylab = "prior", ylim = c(0, max(c(prior_density, posterior_density)))  # ylim!
)
# adding the likelihood function
lines(x_, posterior_density, col = "red")
legend("topright", legend = c("Prior", "Posterior"), col = c("blue", "red"), lty = 1:1)
```

    Posterior Mean is : 10078.2 
    Posterior Variance is : 2857.143 



    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_29_1.png)
    


Basically the same, but since $$\theta$$ is 5000 instead of 10000, we can see the posetior is also shifted to the left.

**B.2 (e) [3 points] Compute a credible interval for this new posterior distribution. Interpret this interval. In particular, do we have evidence to suggest that $$\mu = 10,000$$?**


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
lower_bound = posterior_mean - 1.96 * sqrt(posterior_variance)
upper_bound = posterior_mean + 1.96 * sqrt(posterior_variance)
lower_bound
upper_bound
```


9973.43359317033



10182.9664068297


$$[9973, 10182]$$ with 95 confidence, we have few evidence to sugguest that $$\mu = 10,000$$?.

**B.2 (f) [3 points] What might be worrisome about the difference between the credible interval inferences made above? Does Bayesian inference suffer from issues related to [researcher degrees of freedom](https://en.wikipedia.org/wiki/Researcher_degrees_of_freedom)?**

Your initial guess has a lot to do with your posterior,  with only one change from $$\theta$$, we have 2 complete inference for $$\mu = 10,000$$. Bayesian for me (at least for now) is all about not enough data (making asumption with fewer data point), so yes, it does suffer from issues related to Researcher degrees of freedom

(Quote)``However, researcher degrees of freedom can lead to data dredging and other questionable research practices where the different interpretations and analyses are taken for granted [5][6] Their widespread use represents an inherent methodological limitation in scientific research, and contributes to an inflated rate of false-positive findings.[1] They can also lead to overestimated effect sizes.
``

## B.3 The negative binomial revisited

Let $$X$$ represent the number of failures observed before $$r$$ successes are observed in a set of independent Bernoulli trials. That is, let $$X$$ be negative binomial$$(r, p)$$ with $$r$$ known and $$p$$ unknown. Set $$r = 5$$, $$X = 15$$, and $$\alpha = 1$$ and $$\beta = 4$$.

**B.3 (a) [6 points] Plot the prior distribution, the likelihood function, and posterior distribution of $$p$$ given the data.**

Prior $$= {\displaystyle {\frac {x^{\alpha -1}(1-x)^{\beta -1}}{\mathrm {B} (\alpha ,\beta )}}\!} = Beta(p; 1,4)$$

Likelihood $$ = p^r(1-p)^X$$

Posterior (also Beta) $$ = p^r(1-p)^XBeta(p; 1,4) = p^5(1-p)^15Beta(p; 1,4)$$


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
alpha = 1
beta = 4
r = 5
X = 15
#range  = \in (0,1)
p_ = seq(0, 1, length.out = 100)

prior  = dbeta(p_, alpha, beta)
likelihood = p_^r * (1 - p_)^X
posterior = likelihood*prior 
#TODO: normalize, otherwise wont work, why
posterior <- posterior / sum(posterior) * length(p_) 

plot(p_, prior, col = "blue",
     xlab = 'x', ylab = "prior", ylim = c(0, max(c(prior, likelihood, posterior))) # ylim!
)
lines(p_, likelihood, col = "red")
lines(p_, posterior, col = "green")

legend('topright', legend = c('Prior', 'Likelihood', 'Posterior'), col = c('blue', 'red', 'green'), lty = 1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_38_0.png)
    


 **B.3 (b) [2 points] Give a $$95\%$$ credible interval based on the posterior in the previous part.**


```R
#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
# posterior no longer normal 
# lower_bound
# upper_bound
```

**B.3 (c) [21 points] Suppose we'd like to make a prediction about the number of failures that we are likely to observe before the *next* $$r = 5$$ trials. Estimate the posterior predictive distribution using Monte Carlo simulations. Plot this distribution.**

(There are three cells below to break up your code a bit. In the first, simulate the prediction values. In the second, calculate the relative frequences. In the third, plot the distribution.)


```R
#YOUR CODE HERE
alpha = 1
beta = 4
r = 5
X = 15
alpha_posterior  = alpha + r
beta_posterior = beta + X

mc_sim  = 1000
# Syntax: rbeta(N, shape1, shape2 ) Parameters: vec: Vector to be used shape1, shape2: beta density of input values
p_posterior_mc = rbeta(mc_sim, alpha_posterior, beta_posterior)
# Syntax: rnbinom(N, size, prob)
prediction_value = rnbinom(mc_sim, size = r, prob = p_posterior_mc)
prediction_value
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1</li><li>39</li><li>14</li><li>28</li><li>20</li><li>11</li><li>45</li><li>12</li><li>35</li><li>24</li><li>4</li><li>18</li><li>14</li><li>87</li><li>8</li><li>11</li><li>13</li><li>9</li><li>9</li><li>23</li><li>7</li><li>42</li><li>10</li><li>18</li><li>75</li><li>22</li><li>18</li><li>26</li><li>14</li><li>26</li><li>38</li><li>20</li><li>5</li><li>19</li><li>13</li><li>6</li><li>21</li><li>5</li><li>17</li><li>14</li><li>21</li><li>13</li><li>10</li><li>21</li><li>14</li><li>11</li><li>20</li><li>4</li><li>9</li><li>17</li><li>9</li><li>3</li><li>48</li><li>29</li><li>8</li><li>23</li><li>15</li><li>10</li><li>4</li><li>19</li><li>44</li><li>19</li><li>17</li><li>10</li><li>14</li><li>16</li><li>9</li><li>25</li><li>23</li><li>17</li><li>3</li><li>3</li><li>10</li><li>23</li><li>31</li><li>42</li><li>34</li><li>6</li><li>26</li><li>20</li><li>3</li><li>12</li><li>26</li><li>41</li><li>30</li><li>4</li><li>7</li><li>11</li><li>21</li><li>23</li><li>19</li><li>14</li><li>13</li><li>20</li><li>22</li><li>34</li><li>23</li><li>39</li><li>52</li><li>27</li><li>46</li><li>15</li><li>5</li><li>20</li><li>27</li><li>11</li><li>3</li><li>13</li><li>18</li><li>3</li><li>32</li><li>17</li><li>4</li><li>13</li><li>5</li><li>26</li><li>15</li><li>21</li><li>4</li><li>30</li><li>11</li><li>12</li><li>17</li><li>12</li><li>32</li><li>2</li><li>35</li><li>11</li><li>13</li><li>19</li><li>42</li><li>22</li><li>4</li><li>42</li><li>5</li><li>51</li><li>12</li><li>7</li><li>38</li><li>8</li><li>14</li><li>4</li><li>5</li><li>13</li><li>55</li><li>7</li><li>46</li><li>6</li><li>19</li><li>2</li><li>3</li><li>11</li><li>21</li><li>5</li><li>36</li><li>55</li><li>12</li><li>17</li><li>39</li><li>42</li><li>14</li><li>19</li><li>7</li><li>21</li><li>9</li><li>14</li><li>16</li><li>10</li><li>47</li><li>16</li><li>8</li><li>10</li><li>6</li><li>23</li><li>34</li><li>27</li><li>20</li><li>14</li><li>21</li><li>29</li><li>12</li><li>12</li><li>9</li><li>3</li><li>15</li><li>43</li><li>9</li><li>23</li><li>21</li><li>23</li><li>5</li><li>14</li><li>8</li><li>26</li><li>11</li><li>25</li><li>12</li><li>20</li><li>17</li><li>5</li><li>⋯</li><li>16</li><li>13</li><li>14</li><li>22</li><li>2</li><li>7</li><li>26</li><li>28</li><li>10</li><li>6</li><li>32</li><li>21</li><li>44</li><li>8</li><li>11</li><li>8</li><li>37</li><li>24</li><li>53</li><li>5</li><li>15</li><li>5</li><li>6</li><li>22</li><li>8</li><li>80</li><li>23</li><li>19</li><li>14</li><li>55</li><li>27</li><li>32</li><li>13</li><li>27</li><li>18</li><li>22</li><li>9</li><li>9</li><li>22</li><li>7</li><li>50</li><li>29</li><li>9</li><li>4</li><li>13</li><li>18</li><li>24</li><li>23</li><li>6</li><li>8</li><li>93</li><li>58</li><li>1</li><li>14</li><li>35</li><li>14</li><li>5</li><li>10</li><li>10</li><li>10</li><li>21</li><li>3</li><li>16</li><li>1</li><li>36</li><li>71</li><li>0</li><li>3</li><li>14</li><li>10</li><li>5</li><li>25</li><li>7</li><li>23</li><li>24</li><li>46</li><li>6</li><li>24</li><li>19</li><li>2</li><li>8</li><li>16</li><li>9</li><li>4</li><li>19</li><li>15</li><li>14</li><li>30</li><li>5</li><li>45</li><li>21</li><li>27</li><li>35</li><li>21</li><li>9</li><li>43</li><li>8</li><li>2</li><li>6</li><li>23</li><li>11</li><li>18</li><li>8</li><li>15</li><li>19</li><li>7</li><li>14</li><li>17</li><li>13</li><li>2</li><li>13</li><li>15</li><li>40</li><li>15</li><li>24</li><li>28</li><li>228</li><li>6</li><li>12</li><li>20</li><li>21</li><li>14</li><li>12</li><li>5</li><li>17</li><li>10</li><li>14</li><li>11</li><li>14</li><li>42</li><li>6</li><li>14</li><li>18</li><li>8</li><li>42</li><li>5</li><li>21</li><li>93</li><li>5</li><li>33</li><li>6</li><li>10</li><li>17</li><li>7</li><li>7</li><li>15</li><li>25</li><li>27</li><li>20</li><li>6</li><li>31</li><li>24</li><li>11</li><li>9</li><li>28</li><li>46</li><li>35</li><li>2</li><li>47</li><li>22</li><li>22</li><li>22</li><li>11</li><li>8</li><li>13</li><li>12</li><li>38</li><li>11</li><li>19</li><li>3</li><li>10</li><li>10</li><li>21</li><li>24</li><li>21</li><li>4</li><li>25</li><li>11</li><li>14</li><li>2</li><li>25</li><li>44</li><li>19</li><li>26</li><li>1</li><li>38</li><li>2</li><li>9</li><li>21</li><li>36</li><li>13</li><li>15</li><li>18</li><li>72</li><li>15</li><li>15</li><li>12</li><li>20</li><li>14</li><li>22</li></ol>




```R
#YOUR CODE HERE
# unit 1 ppd_sim = table(pred_sim) / m
relative_freq = table(prediction_value) / mc_sim
relative_freq
```


    prediction_value
        0     1     2     3     4     5     6     7     8     9    10    11    12 
    0.001 0.006 0.016 0.021 0.027 0.032 0.028 0.038 0.039 0.040 0.043 0.043 0.029 
       13    14    15    16    17    18    19    20    21    22    23    24    25 
    0.037 0.051 0.034 0.039 0.025 0.028 0.030 0.026 0.040 0.027 0.030 0.017 0.014 
       26    27    28    29    30    31    32    33    34    35    36    37    38 
    0.025 0.018 0.015 0.007 0.010 0.017 0.014 0.005 0.009 0.009 0.008 0.001 0.007 
       39    40    41    42    43    44    45    46    47    48    49    50    51 
    0.008 0.006 0.005 0.009 0.002 0.009 0.002 0.006 0.003 0.003 0.003 0.003 0.001 
       52    53    54    55    57    58    59    60    62    64    65    68    70 
    0.002 0.002 0.004 0.004 0.001 0.001 0.002 0.001 0.001 0.001 0.001 0.001 0.001 
       71    72    75    78    80    87    89    93   228 
    0.001 0.003 0.001 0.001 0.001 0.001 0.001 0.002 0.001 



```R
#YOUR CODE HERE
barplot(relative_freq, names.arg = names(relative_freq),
        xlab = "number of failures", ylab = "relative frequency")
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_44_0.png)
    


**B.3 (d) [6 points] Let $$X^*$$ be the number of failures observed before $$r = 5$$ successes on a future set of Bernoulli trials.  Estimate $$P(0 \le X^* \le 35)$$. Interpret your answer.**


```R
#YOUR CODE HERE
prob_0_le_X_le_35 <- mean(prediction_value <= 35)
prob_0_le_X_le_35
```


0.89


YOUR ANSWER HERE

## B.4 Inference on the number of trials in a binomial <span style="color: #CFB87C;">(STAT 5630 Students Only, 30 points)</span>

Let $$n$$ be the unknown number of customers that visit a store on the day of a sale. The number of customers that make a purchase is $$Y \, | \, n \sim Binomial(n, \theta)$$, where $$\theta$$ is the *known* probability of making a purchase, given that the customer visited the store. The prior is $$n \sim Poisson(10)$$. 

**B.4 (a) [4 points] Plot the prior distribution for $$n$$. Choose values of $$n$$ between $$0$$ and $$30$$.** 


```R
#YOUR CODE HERE
lambda = 10
n = 0:30

prior = dpois(n, lambda)

plot(n, prior, col = "blue", type = 'l', # line
     xlab = 'x', ylab = "prior")
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_49_0.png)
    


**B.4 (b) [6 points] Assuming $$\theta = 0.2$$ is known, and that there were $$y = 5$$ purchases on the day of the sale, compute the posterior distribution for $$n \, | \, y$$. Plot the distribution. How does it compare to the prior distribution?**


```R
#YOUR CODE HERE
theta = .2
y = 5
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$$Y \, | \, n \sim Binomial(n, \theta)$$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
posterior
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0</li><li>0</li><li>0</li><li>0</li><li>0</li><li>0.000335462747040314</li><li>0.00268370197632251</li><li>0.01073480790529</li><li>0.0286261544141068</li><li>0.0572523088282136</li><li>0.0916036941251417</li><li>0.122138258833522</li><li>0.139586581524025</li><li>0.139586581524026</li><li>0.124076961354689</li><li>0.0992615690837515</li><li>0.0721902320609101</li><li>0.0481268213739401</li><li>0.0296165054608862</li><li>0.0169237174062207</li><li>0.00902598261665104</li><li>0.00451299130832552</li><li>0.0021237606156826</li><li>0.000943893606970043</li><li>0.000397428887145281</li><li>0.000158971554858113</li><li>6.05605923269e-05</li><li>2.20220335734182e-05</li><li>7.6598377646672e-06</li><li>2.55327925488906e-06</li><li>8.17049361564501e-07</li></ol>




```R
#YOUR CODE HERE
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(prior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_52_0.png)
    


**B.4 (c) [4 points] Again, assuming $$\theta = 0.2$$ is known, but now assuming that there were $$y = 10$$ purchases on the day of the sale, compute the posterior distribution for $$n \, | \, y$$. Plot the distribution. How does it compare to the prior distribution and the posterior from the previous part?**

We can see with more trials, the likelihood shift right, so is the posterior. 


```R
#YOUR CODE HERE
theta = .2
y = 10
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$$Y \, | \, n \sim Binomial(n, \theta)$$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_55_0.png)
    


**B.4 (d) [4 points] Again, assuming $$\theta = 0.2$$ is known, but now assuming that there were $$y = 15$$ purchases on the day of the sale, compute the posterior distribution for $$n \, | \, y$$. How does it compare to the prior distribution and the posterior from the previous part?**

posterior more shift to the right, while prior stays the same.


```R
#YOUR CODE HERE
theta = .2
y = 15
lambda = 10
n = 0:100 #TODO: why likelihood is far off? 

prior = dpois(n, lambda)

#$$Y \, | \, n \sim Binomial(n, \theta)$$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_58_0.png)
    


**B.4 (e) [12 points] Repeat parts (b)-(d) for $$\theta = 0.8$$. Comment on any changes.**


```R
#YOUR CODE HERE
theta = .8
y = 5
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$$Y \, | \, n \sim Binomial(n, \theta)$$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_60_0.png)
    



```R
#YOUR CODE HERE
theta = .8
y = 10
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$$Y \, | \, n \sim Binomial(n, \theta)$$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_61_0.png)
    



```R
#YOUR CODE HERE
theta = .8
y = 15
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$$Y \, | \, n \sim Binomial(n, \theta)$$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)
```


    
![png](STAT4630-5630_HW2_files/STAT4630-5630_HW2_62_0.png)
    


Immediately, we can see the y axis increase, which means there are more likelihood to purchase, translating to higher posteriors. The shifting still following the previous section: more purchase, more shift on the right.


```R

```
