# Homework #3

**See Canvas for this assignment and due date**. Complete all of the following problems. Ideally, the theoretical problems should be answered in a Markdown cell directly underneath the question. If you don't know LaTex/Markdown, you may submit separate handwritten solutions to the theoretical problems, but please see the [class scanning policy](https://docs.google.com/document/d/17y5ksolrn2rEuXYBv_3HeZhkPbYwt48UojNT1OvcB_w/edit?usp=sharing). Please do not turn in messy work. Computational problems should be completed in this notebook (using the R kernel). Computational questions may require code, plots, analysis, interpretation, etc. Working in small groups is allowed, but it is important that you make an effort to master the material and hand in your own work. 

## A. Theoretical Problems

### A.1 Improper posterior distributions

**[15 points] Let $$f(\mathbf{x} \, | \, \theta)$$ be a likelihood function such that $$f(\mathbf{x} \, | \, \theta) \le C(\mathbf{x})$$ for all values of $$\theta \in \Theta$$. Show that if $$\pi(\theta)$$ is a proper prior (i.e., $$\int \pi(\theta)d\theta = 1$$), then the posterior distribution is also proper.** 

HINT: A distribution is proper if we know it integrates to a finite value (because once it is finite, we can always rescale it to integrate to 1!).

With  Bayes' theorem, we know that the posterior is

$$\pi(\theta | \mathbf{x})   = \dfrac{f(\mathbf{x} \, | \, \theta)\pi(\theta)}{\int_{\Theta}f(\mathbf{x} \, | \, \underline{\theta})\pi(\underline{\theta})d\underline{\theta}}$$, 

given $$f(\mathbf{x} \, | \, \theta) \le C(\mathbf{x})$$, we will have

$$ \int_{\Theta}f(\mathbf{x} \, | \, \underline{\theta})\pi(\underline{\theta})d\underline{\theta} \le \int_{\Theta}C(\mathbf{x})\pi(\underline{\theta})d\theta_i = C(\mathbf{x})\int_{\Theta}\pi(\underline{\theta})d\theta_i = C(\mathbf{x})$$

Therefore, 

$$\int\pi(\theta | \mathbf{x})  = \int \dfrac{f(\mathbf{x} \, | \, \theta)\pi(\theta)}{C(\mathbf{x})} = \dfrac{1}{C(\mathbf{x})}$$, which is a constant.

Hence, it is a proper distribution

### A.2 Jefferys' prior

Let $$X_1,...,X_n \overset{iid}{\sim} \Gamma(1,\lambda)$$.

**A.2 (a) [10 points] Derive the Jeffreys' prior for $$\lambda$$.**

The Log likelihood function for this problem is define as 

$$l(\lambda) = \log(\lambda^n e^{-\lambda \sum_{i=1}^{n}x_i}) = n\log(\lambda) - \lambda \sum_{i=1}^{n}x_i$$

therefore, fisher information 

$$I(\lambda) = -E(\dfrac{d l(\lambda)}{d\lambda^2}) = -E(-\dfrac{n}{\lambda^2}) = \dfrac{n}{\lambda^2}$$

We get 

$$JP = \sqrt{\dfrac{n}{\lambda^2}} = \dfrac{\sqrt{n}}{\lambda} \propto \dfrac{1}{\lambda}$$

**A.2 (b) [2 point] Is the prior proper?**

Due to 

$$\int \dfrac{1}{\lambda}d\lambda = log(\lambda) + C \ne 1$$

therefore, it is not proper

**A.2 (c) [5 points] Derive the posterior and give conditions on $$\mathbf{X} = (X_1,...,X_n)^T$$ to ensure it is proper.**

$$\pi(\lambda)  \propto \lambda^n e^{-\lambda \sum_{i=1}^{n}x_i}\dfrac{1}{\lambda}  = \lambda^{n-1}e^{-\lambda \sum_{i=1}^{n}x_i}$$

And, in order for 

$$\int \lambda^{n-1}e^{-\lambda \sum_{i=1}^{n}x_i}d \lambda = C $$

it is basically [Gamma function](https://en.wikipedia.org/wiki/Gamma_function) when $$\sum_{i=1}^{n}x_i = 1$$

$${\displaystyle \Gamma (n)=(n-1)!\,.} \forall n \in {Z}$$ 

therefore the conditon is 

$$n > 1, \sum_{i=1}^{n}x_i = 1$$

### A.3 The invariance of Jefferys' prior

Let $$X_1,...,X_n \overset{iid}{\sim}f(\mathbf{x} \, | \, \theta)$$ with associated Fisher information number $$I_\theta(\theta)$$. Consider the (continuous, differentiable, invertible) reparameterization according to $$\gamma = g(\theta)$$. Denote the Fisher information for this reparameterization as $$I_\gamma(\gamma)$$. 

**A.3 (a) [16 points] Show that**

\begin{align*}
I_\gamma(\theta) = I_\theta(\theta)\left(\frac{d\theta}{d\gamma} \right)^2.
\end{align*}


From textbook (2.35), we know 
$$I(\theta) = -E\left(\dfrac{d^2\log f(\mathbf{x} \, | \, \theta)}{d \theta^2}\right)$$ 

next, we evaluate $$\pi(\cdot)$$ at $$\theta = g^{-1}(\gamma)$$:

$$[\pi(\theta)]^2 \propto I_{\gamma}(\theta)  = -E\left(\dfrac{d^2\log f(\mathbf{x} \, | \, \theta)}{d \gamma^2}\right) = -E\left(\dfrac{d}{d\gamma}(\dfrac{d\log f(\mathbf{x} \, | \, \theta)}{d \theta}\dfrac{d\theta}{d\gamma})\right)$$

$$= -E\left(\dfrac{d}{d\gamma}(\dfrac{\frac{d f(\mathbf{x} \, | \, \theta)}{d\theta}}{ f(\mathbf{x} \, | \, \theta)}\dfrac{d\theta}{d\gamma})\right)$$

from the class lecture, we also know that
$$\dfrac{d f(\mathbf{x} \, | \, \theta)}{d\theta} = [\dfrac{d\log f(\mathbf{x} \, | \, \theta)}{d\theta}]f(\mathbf{x} \, | \, \theta)$$, plug back in the formula above:

$$= -E\left(\dfrac{d}{d\gamma}(\dfrac{d \log f(\mathbf{x} \, | \, \theta)}{ d\theta}\dfrac{d\theta}{d\gamma})\right)$$, apply product rule 


$$= -E\left[\dfrac{d \log f(\mathbf{x} \, | \, \theta)}{ d\theta}\dfrac{d^2\theta}{d\gamma^2} + \dfrac{d^2 \log f(\mathbf{x} \, | \, \theta)}{ d\theta^2}(\dfrac{d\theta}{d\gamma})^2\right]$$,

where $${\dfrac{d \log f(\mathbf{x} \, | \, \theta)}{ d\theta}\dfrac{d^2\theta}{d\gamma^2}} = 0$$ and $$ -E\left(\dfrac{d^2\log f(\mathbf{x} \, | \, \theta)}{d \theta^2}\right) = I(\theta)$$

therefore:
\begin{align*}
I_\gamma(\theta) = I_\theta(\theta)\left(\frac{d\theta}{d\gamma} \right)^2.
\end{align*}

**A.3 (b) [5 points] Argue that this implies that the Jeffrey's prior is invariant to transformations.** (Nothing mathematical to do here...)

We see that $$J(\gamma)^{1/2} = J(\theta)^{1/2}|\frac{d\theta}{d \gamma}|$$, meaning the choice of parameterization does not influence the objective aspects of bayesian inference.

**Note: The following question is a bit long, and so I am giving it to you on this assignment, with no points assigned, so that you can get a head start if you'd like. It will be included, with points and graded, on the next homework.**

## A.4 <span style="color: #CFB87C;">(STAT 5630 Students Only, to be completed for HW4)</span>  Conjugate normal model

Suppose $$\mathbf{X} = (X_1,...,X_n)^T$$ is an independent and identically distributed sample of size $$n$$ from the distribution $$N(\mu,\sigma^2)$$, where the prior distribution for $$(\mu,\sigma^2)$$ is 

$$$$\pi(\mu, \sigma^2) = \pi(\mu \, | \, \sigma^2) \, \times \, \pi(\sigma^2)$$$$

where 

- $$\pi(\mu \, | \, \sigma^2) = N\left(\mu_0, \,  \sigma^2\big/k_0\right)$$ 

- $$\pi(\sigma^2) = \text{inv-}\Gamma\left(v_0\big/2, \, v_0\sigma_0^2\big/2\right)$$. 

We might say that $$(\mu,\sigma^2) \sim \text{normal-inverse-}\Gamma\left(\mu, \sigma^2 \, | \, \mu_0, \sigma_0^2\big/k_0; v_0, \sigma_0^2\right)$$. The posterior distribution, $$\pi(\mu,\sigma^2\, | \, \mathbf{x})$$, is also normal-inverse-$$\Gamma$$.

**Explicitly derive the posterior parameters in terms of the prior parameters and the sufficient statistics of the data.**

YOUR ANSWER HERE

## B. Computational Problems

## B.1 An objective analysis in R

This problem refers to problem A.3.

**B.1 (a) [10 points] Conduct an "objective" Bayesian analysis using the prior and posterior from A.2 and $$x = (0.4478, 0.6173, 1.1317, 0.9011, 1.9250)$$. That is, compute the posterior distribution for the data and prior above.**

Use R functions for your answer; this will make part (c) easier!

 prior is $$JP = \sqrt{\dfrac{n}{\lambda^2}} = \dfrac{\sqrt{n}}{\lambda} \propto \dfrac{1}{\lambda}$$
posterior is  $$\pi(\lambda)  \propto \lambda^n e^{-\lambda \sum_{i=1}^{n}x_i}\dfrac{1}{\lambda}  = \lambda^{n-1}e^{-\lambda \sum_{i=1}^{n}x_i}$$


```R
 # meaning of lambda: x are gamma.   lambda is rate parameter. 
```


```R
#YOUR CODE HERE
x = c(0.4478, 0.6173, 1.1317, 0.9011, 1.9250)

n = length(x)

sum_x_i = sum(x)

#posterior 
posterior  =  function(lambda) {
  lambda^(n-1) * exp(-lambda * sum_x_i)
}
#prior 
prior  =  function(lambda) {
  1 / lambda  
}

#lambda in [0,2]
lambda_range = seq(0, 2, length.out=100)

posterior_values = sapply(lambda_range, posterior) 
posterior_values

prior_values =  sapply(lambda_range, prior)
# prior_values
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0</li><li>1.50490872028341e-07</li><li>2.17551005992995e-06</li><li>9.95077912893007e-06</li><li>2.84146942649189e-05</li><li>6.26778360448691e-05</li><li>0.000117427531831065</li><li>0.000196556787599966</li><li>0.000302961043580747</li><li>0.000438457832247284</li><li>0.000603793938869782</li><li>0.000798712431811001</li><li>0.00102205823601766</li><li>0.00127190601674856</li><li>0.00154569823040679</li><li>0.00184038446102054</li><li>0.00215255574177413</li><li>0.00247856958435264</li><li>0.00281466300771824</li><li>0.00315705205807004</li><li>0.00350201721440808</li><li>0.00384597473833728</li><li>0.00418553450121388</li><li>0.00451754514650867</li><li>0.00483912765314253</li><li>0.00514769848328515</li><li>0.0054409835473919</li><li>0.00571702421758004</li><li>0.00597417658182744</li><li>0.00621110506705407</li><li>0.00642677147768915</li><li>0.0066204204046693</li><li>0.00679156186320484</li><li>0.00693995192006007</li><li>0.00706557197544797</li><li>0.00716860727304363</li><li>0.00724942512551142</li><li>0.00730855326324364</li><li>0.00734665864123748</li><li>0.00736452697340201</li><li>0.00736304320505943</li><li>0.00734317308278771</li><li>0.00730594593572257</li><li>0.00725243874359206</li><li>0.00718376153364049</li><li>0.00710104412072465</li><li>0.00700542418174259</li><li>0.00689803663669255</li><li>0.0067800042935849</li><li>0.00665242970269025</li><li>0.00651638815677894</li><li>0.00637292176769653</li><li>0.0062230345454647</li><li>0.00606768840376859</li><li>0.00590780001489224</li><li>0.00574423843762775</li><li>0.0055778234431754</li><li>0.00540932446636256</li><li>0.00523946011245761</li><li>0.00506889815328279</li><li>0.00489825595010179</li><li>0.00472810124475805</li><li>0.00455895326467012</li><li>0.00439128409146984</li><li>0.00422552024723023</li><li>0.00406204445631744</li><li>0.00390119754487238</li><li>0.00374328044374773</li><li>0.00358855626436945</li><li>0.00343725242044083</li><li>0.00328956277164769</li><li>0.00314564976854963</li><li>0.0030056465806502</li><li>0.00286965919222874</li><li>0.00273776845289331</li><li>0.00261003207198054</li><li>0.00248648654789473</li><li>0.00236714902525087</li><li>0.00225201907427701</li><li>0.00214108038834863</li><li>0.00203430239678343</li><li>0.00193164179113025</li><li>0.00183304396415123</li><li>0.00173844436153267</li><li>0.00164776974707898</li><li>0.00156093938275426</li><li>0.0014778661254492</li><li>0.0013984574427748</li><li>0.00132261635052925</li><li>0.00125024227475726</li><li>0.00118123184153146</li><li>0.00111547959773879</li><li>0.00105287866625971</li><li>0.000993321338988934</li><li>0.000936699611170536</li><li>0.000882905660511858</li><li>0.000831832274504727</li><li>0.000783373229323697</li><li>0.000737423623592558</li><li>0.000693880170216312</li></ol>



**B.1 (b) [7 points] Plot the prior and posterior distributions. (You may want to multiply the prior by a constant so that you can visualize it with the posterior.)**


```R
#YOUR CODE HERE
# plot(lambda_range, posterior_values, type='l', col='blue', 
#      xlab="Lambda", ylab='posterior_values',
#     )
plot(lambda_range, posterior_values, type='l', col='blue',
 xlab="Lambda", ylab='posterior_values',
#  ylim=c(0, max(posterior_values, prior_values))
    )

# multiply the constant on prior_values
lines(lambda_range, 0.0009*prior_values, type='l', col='red', lty=2)  

legend("topright", legend=c("Posterior", "Prior"), col=c("blue", "red"), lty=c(1, 2))

```


    
![png](STAT4630-5630_HW3_files/STAT4630-5630_HW3_22_0.png)
    


**B.1 (c) [4 points] Numerically verify that the posterior is proper.**

HINT: The `integrate()` function in `R` may be helpful.


```R
#YOUR CODE HERE
#  demoniators.  dont forget about that. 
denom  =  integrate(function(lambda) lambda^(n-1) * exp(-lambda * sum_x_i), lower = 0, upper = Inf)

normal_posterior =  function(lambda) {
  lambda^(n-1) * exp(-lambda * sum_x_i)  / denom$value # TODO: 
} 
                    
integrate(normal_posterior, lower = 0, upper = Inf)
# it is proper since the integration is 1. 
```


    1 with absolute error < 2.6e-08


## B.2 Computing a posterior with a nonconjugate model

Suppose $$y_1, \ldots, y_5$$ are independent samples from a Cauchy distribution with unknown center/location $$\theta$$ and known scale $$1$$:

\begin{align*}
f(y_i|\theta) \propto \frac{1}{1 + (y_i - \theta)^2}
\end{align*}

Assume, for simplicity, that the prior distribution for $$\theta$$ is the continuous uniform on $$[0, 100]$$.

Given the observations $$(y_1, \ldots, y_5) = (53, 54, 57, 60.3, 44.7)$$:

**B.2 (a) Compute the unnormalized posterior density function, $$\pi(\theta \, | \, \mathbf{y}) \propto \pi(\theta)f(\mathbf{y}|\theta)$$, on a dense grid of points $$\theta \in (0, 100)$$. Using the grid approximation, compute and plot the normalized posterior density function, $$\pi(\theta|y)$$, as a function of $$\theta$$.**

HINT: The `integrate()` function in `R` may be helpful. Use the first cell for computations and the second cell for plotting.


```R
#YOUR CODE HERE
y =c(53, 54, 57, 60.3, 44.7)

# grid approx
theta_grid = seq(0, 100, length.out = 1000)

# uniform prior
# TODO: why not prior  = 1/100 ??? 
prior = function(theta) {
  ifelse(theta >= 0 & theta <= 100, 1/100, 0)
    }

# Compute the unnormalized posterior density function
likelihood =function(y_i, theta) {
  1/(1 + (y_i - theta)^2)
}

# posterior
unnormalized_posterior  = function(theta) {
  prod(sapply(y, likelihood, theta)) * prior(theta)
}

# TODO: why not the following but long integral
# posterior_values  =  sapply(theta_grid, unnormalized_posterior)

# normalized_posterior_values =  posterior_values/sum(posterior_values)

total_integral = integrate(unnormalized_posterior, lower = 0, upper = 100)$value

normalized_posterior = function(theta) {
  unnormalized_posterior(theta) / total_integral
}
#TODO: The error cannot coerce type 'closure' to vector of type 'double
normalized_posterior_values = sapply(theta_grid, normalized_posterior)


```


```R
#YOUR CODE HERE
plot(theta_grid, normalized_posterior_values, type='l', 
     col='red', xlab='Theta', ylab='Density')
```


    
![png](STAT4630-5630_HW3_files/STAT4630-5630_HW3_27_0.png)
    


**B.2 (b) Sample 1000 draws of $$\theta$$ from the posterior density. Plot a relative frequency histogram of the draws with the posterior density imposed on top.**

Use the first cell for computations and the second cell for plotting.


```R
#YOUR CODE HERE
theta_grid = seq(0, 100, length.out = 1000)

# normalized_posterior = unnormalized_posterior / sum(unnormalized_posterior)

total_integral = integrate(unnormalized_posterior, lower = 0, upper = 100)$value

normalized_posterior = function(theta) {
  unnormalized_posterior(theta) / total_integral
}

normalized_posterior_probs = sapply(theta_grid, normalized_posterior)

theta_samples = sample(theta_grid, size=1000, replace=TRUE, prob=normalized_posterior_probs)

```


```R
# plot(theta_grid, normalized_posterior_probs, col='red', lwd=2)
```


```R
#YOUR CODE HERE
hist(theta_samples, breaks=50, probability=TRUE, col='blue',
     xlab='Theta', ylab='Relative Frequency')

# TODO: Density aes(x = theta_samples, y = ..density..)
theta_density = density(theta_samples) 

lines(theta_grid, 
      normalized_posterior_probs*max(theta_density$y)/max(normalized_posterior_probs)
      , col='red')
legend("topright", legend = c("relative frequency", "posterior density"), col = c("blue", "red"), lty = 1:1)
```


    
![png](STAT4630-5630_HW3_files/STAT4630-5630_HW3_31_0.png)
    


**B.2 (c) Obtain $$1000$$ samples from the predictive distribution of a future observation, $$y_6$$, and plot a histogram of the predictive draws.**


```R
#YOUR CODE HERE
# random seeds?
set.seed(42)

predictive_samples = rcauchy(n = 1000, location = theta_samples, scale = 1)


hist(predictive_samples, breaks = 50, probability=TRUE, col='blue',
     xlab='Theta', ylab='Relative Frequency')
```


    
![png](STAT4630-5630_HW3_files/STAT4630-5630_HW3_33_0.png)
    



```R

```
