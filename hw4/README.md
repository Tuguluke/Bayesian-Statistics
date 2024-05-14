# Homework #4

**See Canvas for this assignment and due date**. Complete all of the following problems. Ideally, the theoretical problems should be answered in a Markdown cell directly underneath the question. If you don't know LaTex/Markdown, you may submit separate handwritten solutions to the theoretical problems, but please see the [class scanning policy](https://docs.google.com/document/d/17y5ksolrn2rEuXYBv_3HeZhkPbYwt48UojNT1OvcB_w/edit?usp=sharing). Please do not turn in messy work. Computational problems should be completed in this notebook (using the R kernel). Computational questions may require code, plots, analysis, interpretation, etc. Working in small groups is allowed, but it is important that you make an effort to master the material and hand in your own work. 

## A.1 Mixtures of conjugate priors

Let $$\theta_1,...,\theta_n$$ be continuous random variables with pdfs $$\pi_{i}(\theta)$$, $$i = 1,...,n$$. Define a *mixture distribution* to be a random variable $$\theta$$ with pdf

\begin{align*}
\pi(\theta) = \sum^n_{i=1}\alpha_i \pi_{i}(\theta) 
\end{align*}

where $$\alpha_i$$ are nonnegative real numbers such that $$\sum^n_{i=1}\alpha_i = 1$$.

**A.1 (a) [10 points] Show that $$\pi(\theta)$$ is a pdf.**

$$\int_{-\infty}^{+\infty} \pi(\theta) d\theta  = \int_{-\infty}^{+\infty} \sum^n_{i=1}\alpha_i \pi_{i}(\theta)  d\theta = \int_{-\infty}^{+\infty} \alpha_1 \pi_{1}(\theta)  d\theta + \int_{-\infty}^{+\infty} \alpha_2 \pi_{2}(\theta)  d\theta + \cdots + \int_{-\infty}^{+\infty} \alpha_n \pi_{n}(\theta)  d\theta$$

$$ = \alpha_1\underbrace{\int_{-\infty}^{+\infty}  \pi_{1}(\theta)  d\theta}_{ =1} + \alpha_2\underbrace{\int_{-\infty}^{+\infty}  \pi_{2}(\theta)  d\theta}_{ =1}  + \cdots + \alpha_n \underbrace{\int_{-\infty}^{+\infty} \pi_{n}(\theta)  d\theta}_{ =1}, \,$$  where $$\pi_{i}(\theta)$$ are PDFs (hence $$ = 1$$)

therefore:

$$\int_{-\infty}^{+\infty} \pi(\theta) d\theta  = \sum^n_{i=1}\alpha_i = 1 \,$$ shown that $$\pi(\theta)$$ is a pdf as well.

**A.1 (b) [10 points] Consider $$\pi(\theta) = \sum^n_{i=1}\alpha_i \pi_{i}(\theta)$$ as a prior on $$\theta$$ and let $$f(\mathbf{x} \, | \, \theta)$$ denote the relevant likelihood function. Show that the posterior on $$\theta \, | \, \mathbf{x}$$  is**

\begin{align*}
\pi(\theta \, | \, \mathbf{x}) \propto \sum^n_{i=1}A_i\pi_i(\theta\, | \, \mathbf{x}),
\end{align*}

**where $$\pi_i(\theta\, | \, \mathbf{x}) \propto f(\mathbf{x} \, | \, \theta)\pi_i(\theta)$$ and $$\sum^n_{i=1}A_i = 1$$.**


With Bayes' theorem, we have

$$\pi(\theta | \underline{x})  =  \dfrac{f(\underline{x}|\theta)\pi(\theta)}{\int f(\underline{x}|\theta)\pi(\theta)d\theta} = \dfrac{f(\underline{x}|\theta)\sum^n_{i=1}\alpha_i \pi_{i}(\theta)}{\int f(\underline{x}|\theta)\sum^n_{i=1}\alpha_i \pi_{i}(\theta)d\theta}= \dfrac{\sum^n_{i=1}\alpha_i f(\underline{x}|\theta)\pi_{i}(\theta)}{\int f(\underline{x}|\theta)\sum^n_{i=1}\alpha_i \pi_{i}(\theta)d\theta} $$

change index $$i$$ to $$j$$ in the denominator, we have 

$$ \pi(\theta | \underline{x})= \dfrac{\sum^n_{i=1}\alpha_i f(\underline{x}|\theta)\pi_{i}(\theta)}{\int f(\underline{x}|\theta)\sum^n_{j=1}\alpha_j \pi_{j}(\theta)d\theta} = \dfrac{\sum^n_{i=1}\alpha_i \pi_i(\theta |  \underline{x})\int f(\underline{x}|\theta)\pi_{i}(\theta)d\theta}{\int f(\underline{x}|\theta)\sum^n_{j=1}\alpha_j \pi_{j}(\theta)d\theta} $$

since $$  \boxed{\dfrac{f(\underline{x}|\theta)\pi_{i}(\theta)}{\int f(\underline{x}|\theta)\pi_{i}(\theta)d\theta}=  \pi_i(\theta |  \underline{x})}$$
 
$$= \dfrac{\sum^n_{i=1}\alpha_i \int f(\underline{x}|\theta)\pi_{i}(\theta)d\theta}{\int f(\underline{x}|\theta)\sum^n_{j=1}\alpha_j \pi_{j}(\theta)d\theta} \pi_i(\theta |  \underline{x})$$  


Once we defined 
$$A_i = \dfrac{\alpha_i \int f(\underline{x}|\theta) \pi_{i}(\theta)d\theta}{\sum^n_{j=1}\alpha_j\int f(\underline{x}|\theta) \pi_{j}(\theta)d\theta} $$ so that 
$$\sum^n_{i=1}A_i = \dfrac{\sum^n_{i=1}\alpha_i \int f(\underline{x}|\theta) \pi_{i}(\theta)d\theta}{\sum^n_{j=1}\alpha_j\int f(\underline{x}|\theta) \pi_{j}(\theta)d\theta}=  1$$

 we can get:
 
 \begin{align*}
\pi(\theta \, | \, \mathbf{x}) \propto \sum^n_{i=1}A_i\pi_i(\theta\, | \, \mathbf{x}),
\end{align*}

## A.2 The Wishart and the $$\chi^2$$

Let $$D = (\mathbf{X}_1,...,\mathbf{X}_n)$$ denote a $$p\times n$$ data matrix with each column $$\mathbf{X}_j = (X_1,...,X_p)^T \sim N_p(\boldsymbol 0, \Sigma_p)$$ for $$j = 1,...,n$$. $$\Sigma_p$$ is an $$p \times p$$ covariance matrix. Then $$M = DD^T$$ is a $$p\times p$$ positive definite matrix and $$M \sim W_p(n, \Sigma_p)$$, where $$\Sigma_p$$ is $$p \times p$$ and positive definite. 

**[10 points] Show that if $$p = 1$$ and $$\Sigma_p = 1$$ then $$M \sim \chi^2(n)$$.**

If $$p = 1$$ and $$\Sigma_p = 1$$ , then $$D = (\mathbf{X}_1,...,\mathbf{X}_n)$$ denote a $$1\times n$$ data matrix with each column $$\mathbf{X}_j\sim N_1(\boldsymbol 0, 1)$$ for $$j = 1,...,n$$. 

$$M = DD^T = (\mathbf{X}_1,...,\mathbf{X}_n)(\mathbf{X}_1,...,\mathbf{X}_n)^T = \mathbf{X}_1^2 + \cdots + \mathbf{X}_n^2 = \sum_{j =1}^n\mathbf{X}_j^2$$

This is the exact definition of [Chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution). Hence

$$M \sim \chi^2(n)$$.

## A.3 Conjugate normal model

Suppose $$\mathbf{X} = (X_1,...,X_n)^T$$ is an independent and identically distributed sample of size $$n$$ from the distribution $$N(\mu,\sigma^2)$$, where the prior distribution for $$(\mu,\sigma^2)$$ is 

$$$$\pi(\mu, \sigma^2) = \pi(\mu \, | \, \sigma^2) \, \times \, \pi(\sigma^2)$$$$

where 

- $$\pi(\mu \, | \, \sigma^2) = N\left(\mu_0, \,  \sigma^2\big/k_0\right)$$ 

- $$\pi(\sigma^2) = \text{inv-}\Gamma\left(v_0\big/2, \, v_0\sigma_0^2\big/2\right)$$. 

We might say that $$(\mu,\sigma^2) \sim \text{normal-inverse-}\Gamma\left(\mu, \sigma^2 \, | \, \mu_0, \sigma_0^2\big/k_0; v_0, \sigma_0^2\right)$$. The posterior distribution, $$\pi(\mu,\sigma^2\, | \, \mathbf{x})$$, is also normal-inverse-$$\Gamma$$.

**[15 points] Explicitly derive the posterior parameters in terms of the prior parameters and the sufficient statistics of the data.**

### Office hour Note: $$\bar{\mathbf{X}}$$ is sufficient for  $$\mu$$, $$\, s^2  = \dfrac{1}{n-1}\sum (X_i - \bar{\mathbf{X}})^2$$  is sufficient for $$\sigma^2$$

Priors: $$\pi(\mu | \sigma^2) \propto exp\left\{-\dfrac{k_0}{2\sigma^2}(\mu -\mu_0)^2\right\}$$, (skipping the therm $$\sigma^{-1}$$)

$$\pi(\sigma^2) \propto (\sigma^2)^{-(\frac{v_0}{2} +1)}exp\left\{-\dfrac{v_0\sigma_0^2}{2\sigma^2}\right\}$$

so that $$\pi(\mu,\sigma) \propto (\sigma^2)^{- \frac{v_0}{2} - 1} exp\left\{-\dfrac{k_0}{2\sigma^2}(\mu -\mu_0)^2  -\dfrac{v_0\sigma_0^2}{2\sigma^2}\right\}$$

Likelihood: $$f(\mathbf{x} | \, \mu,\sigma^2) = (\sigma^2)^{- \frac{n}{2}} exp\left\{-\dfrac{1}{2\sigma^2}(n -1)s^2  -\dfrac{1}{2\sigma^2}n(\bar{\mathbf{x}} - \mu)^2\right\}$$

now mulitiply  therm from above:

$
\pi(\mu, \sigma^2 | X) \propto (\sigma^2)^{-\frac{n}{2} - \frac{v_0}{2} - 1} \exp\left(-\frac{1}{2\sigma^2}\left[(n -1)s^2 + n(\bar{\mathbf{x}} - \mu)^2 + k_0(\mu -\mu_0)^2 + v_0\sigma_0^2\right]\right)
$

for $$\boxed{(n -1)s^2 + n(\bar{\mathbf{x}} - \mu)^2 + k_0(\mu -\mu_0)^2 + v_0\sigma_0^2}$$

By collection coefficient for $$\mu$$, we get 

$
(n + k_0)\mu^2 - 2(n\bar{x} + k_0\mu_0)\mu + n\bar{x}^2 + k_0\mu_0^2
$

This just shows that if we set it as a quadratic formula

$$(n + k_0)\left(\mu - \dfrac{n\bar{x} + k_0\mu_0}{n + k_0}\right)^2 - \dfrac{(n\bar{x} + k_0\mu_0)^2}{n + k_0} + n\bar{x}^2 + k_0\mu_0^2$$

where $$k_n = k_0 + n,\,  \mu_n = \frac{k_0}{k_n}\mu_0 + \frac{n}{k_n} \bar{x} $$

Expand the second part, 

$$\dfrac{(n\bar{x} + k_0\mu_0)^2}{n + k_0} = \dfrac{n^2\bar{x}^2 + 2nk_0\bar{x}\mu_0 + k_0^2\mu_0^2}{n + k_0}$$

and combine/simplify
$\dfrac{2n^2\bar{x}^2 + nk_0\bar{x}^2 + 2nk_0\bar{x}\mu_0 + 2k_0^2\mu_0^2 + nk_0\mu_0^2}{n + k_0}
$



Next, we plug  back $$v_0\sigma_0^2$$

$$(n + k_0)\left(\mu - \dfrac{n\bar{x} + k_0\mu_0}{n + k_0}\right)^2 \, - \dfrac{2n^2\bar{x}^2 + nk_0\bar{x}^2 + 2nk_0\bar{x}\mu_0 + 2k_0^2\mu_0^2 + nk_0\mu_0^2}{n + k_0}  + v_0\sigma_0^2$$

With the original: 

$$(\sigma^2)^{-\frac{v_0 + n}{2}  - 1} \exp\left(-\frac{1}{2\sigma^2}\left[(n -1)s^2 + n(\bar{\mathbf{x}} - \mu)^2 + k_0(\mu -\mu_0)^2 + v_0\sigma_0^2\right]\right)$$

Comparing with:
$$\boxed{(\sigma^2)^{-\frac{v_0 + n}{2}  - 1} \exp\left(-\frac{1}{2\sigma^2}\left[ v_0\sigma_0^2 + (n -1)s^2 + \frac{nk_0(\bar{\mathbf{x}} - \mu)^2}{n + k_0} + (n + k_0)(\mu -\frac{n\bar{x} + k_0\mu_0}{n + k_0})^2 \right]\right)}$$


we can get:

$$v_n = v_0 + n, \, \sigma_n^2 = \dfrac{v_0\sigma_0^2 + (n-1)s^2 +\frac{nk_0}{k_n}(\bar{x} - \mu_o)^2 }{v_n}$$

## B. Computational Problems

## B.1 Mixture simulations

Consider `x` in the cell below to be data. Let's suppose that a group of four researchers, none of which know the true mean `mu_true`, wish to estimate it with a Bayesian poterior. However, there is disagreement among the researchers' priors:

- Researcher #1: $$\mu \sim N(3,1)$$

- Researcher #2: $$\mu \sim N(7,1)$$

- Researcher #3: $$\mu \sim N(10,1)$$

- Researcher #4: $$\mu \sim N(12,1)$$

Further, Researcher #2 and Researcher #3 have more expertise about the data generating process, and so their prior beliefs should be given the highest weight. Researcher #1 is somewhat trusted, and Researcher #4 is inexperienced, and so we trust them relatively little. More precisely, we weight each researchers' prior according to `alpha = c(0.25, 0.4, 0.3, 0.05)`.

**B.1 (a) [10 points] Construct a prior distribution as a mixture of each individual researcher's prior. Plot the prior.**


```R
set.seed(5630)
n = 50; mu_true = 5; sig_true = 2; x = rnorm(n,mu_true,sig_true)
```


```R
#YOUR CODE HERE
library(ggplot2)
mu_prior  = c(3, 7, 10, 12)
sigma_prior = c(1, 1, 1, 1)
alpha = c(0.25, 0.4, 0.3, 0.05)

mixture_prior_density =  function(mu) {
  density = 0
  for (i in 1:length(alpha)) {
    density = density + alpha[i] * dnorm(mu, mean = mu_prior[i], sd = sigma_prior[i])
  }
  return(density)
}

# mu as seq
mu_values = seq(from = -10, to = 20, length.out = 1000)

prior_densities = sapply(mu_values, mixture_prior_density)

# Plot the prior
df = data.frame(mu = mu_values, density = prior_densities) # list

ggplot(df, aes(x = mu, y = density)) +
  geom_line() +
  xlab("mu") +
  ylab("Mixture prior density") +
  theme_bw()
```


    
![png](STAT5630_Sp23_HW4_files/STAT5630_Sp23_HW4_11_0.png)
    


**B.1 (b) [5 points] Construct and plot the likelihood function for the data.**

we utilize the log-likelihood: $$-\frac{1}{2\sigma^2}\sum(x_i - \mu)^2$$


```R
#YOUR CODE HERE
likelihood = function(mu) {
  -sum((x - mu)^2) / (2 * sig_true^2)
}

# list
likelihood_values = sapply(mu_values, likelihood)
# mu as seq
mu_values = seq(from = -5, to = 20, length.out = 1000)

# Plot the likelihood
df_likelihood = data.frame(mu = mu_values, likelihood =likelihood_values)

ggplot(df_likelihood, aes(x = mu, y = likelihood)) +
  geom_line() +
  xlab("mu") +
  ylab("Likelihood") +
  theme_bw()
```


    
![png](STAT5630_Sp23_HW4_files/STAT5630_Sp23_HW4_14_0.png)
    


**B.1 (c) [6 points] Construct and plot the posterior. Describe it's form.**


```R
#YOUR CODE HERE
# mixture_prior_density =  function(mu) {
#   density = 0
#   for (i in 1:length(alpha)) {
#     density = density + alpha[i] * dnorm(mu, mean = mu_prior[i], sd = sigma_prior[i])
#   }
#   return(density)
# }
prior = function(mu) {
  log(mixture_prior_density(mu))
}

# likelihood = function(mu) {
#   -sum((x - mu)^2) / (2 * sig_true^2)
# }

# TODO: Log!
posterior = function(mu) {
    likelihood(mu) + prior(mu)
}

# Compute the log posterior for each mu value TODO: - max(log_posteriors)
log_posteriors = sapply(mu_values, posterior)
#e^log
posteriors = exp(log_posteriors - max(log_posteriors))  # TODO:Subtract max  


df_posterior = data.frame(mu = mu_values, posterior = posteriors)
ggplot(df_posterior, aes(x = mu, y = posterior)) +
  geom_line() +
  xlab("mu") +
  ylab("Posterior density") +
  theme_bw()
```


    
![png](STAT5630_Sp23_HW4_files/STAT5630_Sp23_HW4_16_0.png)
    


spiked normal with the max same as likelihood, where $$\mu = 5$$

## B.2 Inference on the variance-covariance matrix

Let $$\mathbf{X}_1, \mathbf{X}_2 ,...,\mathbf{X}_{10}$$ each be vectors of length $$p = 4$$ that are distributed as multivariate normal. For $$i=1,...,10$$, $$E(\mathbf{X}_i) = \boldsymbol{\mu}_i = (0,0,0,0,0)^T$$ and $$Var(\mathbf{X}_i) = \boldsymbol\Sigma_p$$ is the $$p \times p$$ variance-covariance matrix. We'd like to estimate $$\boldsymbol\Sigma_p$$.

**B.2 (a) [3 points] Write an R function that computes an inverse Wishart pdf, to serve as the prior distribution for a Bayesian analysis. Let the prior parameters be $$R \approx 0$$ (e.g., a diagonal matrix with diagonal entries random numbers just above zero), and $$v = p = 4$$.**


```R
## From Unit 3 code

# rmvn = function(m,mu,sigma){
#     n = dim(sigma)[1]
#     L = t(chol(sigma))
#     theta = matrix(NA, nrow  = n, ncol = m)
#     for (i in 1:m){
#         z = rnorm(n,0,1)
#         theta[,i] = mu + L%*%z
#     }
#     return(theta)
# }
# rwish = function(v,n,S){
#     theta = rmvn(v,0,S);
#     W = matrix(0, ncol = n, nrow  = n)
    
#     for (i in 1:n){
#         W = W + (theta[,i]%*%t(theta[,i]))
        
#     }
#     return(W)
# }

```


```R
#YOUR CODE HERE
rmvn = function(m, mu, sigma) {
  n = dim(sigma)[1]
  L = t(chol(sigma))
  theta = matrix(NA, nrow = n, ncol = m)
  for (i in 1:m) {
    z = rnorm(n, 0, 1)
    theta[, i] = mu + L%*%z
  }
  return(theta)
}

rwish = function(v, p, S) {
  theta = rmvn(v, rep(0, p), S)
  W = matrix(0, ncol = p, nrow = p)
  
  for (i in 1:v) {
    W = W + theta[, i]%*%t(theta[, i])
  }
  return(W)
}

rinvwish = function(v, p) {
  # R \sim 0 zero
  R = diag(runif(p, 0.0001, 0.001))
  
  W = rwish(v, p, R)
  return(solve(W))
}

# b2a
v = 4  
p = 4  

# ?
inv_wish_pdf = rinvwish(v, p)
inv_wish_pdf
```


<table class="dataframe">
<caption>A matrix: 4 × 4 of type dbl</caption>
<tbody>
	<tr><td>16648.8935</td><td>9744.0708</td><td>1154.9951</td><td>-126.2384</td></tr>
	<tr><td> 9744.0708</td><td>9311.2933</td><td>1400.7535</td><td>-559.3138</td></tr>
	<tr><td> 1154.9951</td><td>1400.7535</td><td>1247.3788</td><td>-350.5753</td></tr>
	<tr><td> -126.2384</td><td>-559.3138</td><td>-350.5753</td><td> 375.8722</td></tr>
</tbody>
</table>



**B.2.(b) [3 points] Use the data below, in the matrix `X`, to write an R function that computes the likelihood of the data $$X$$, given $$\Sigma_p$$.** 


```R
# t(mvrnorm(n=n, mu=mu, Sigma=sigma))
```


```R
set.seed(11)
library(MASS)
n = 10; p = 4

# create the variance covariance matrix
sigma = rbind(c(1,-0.8,-0.7, -0.1), c(-0.8,1, 0.3, 0.3), 
              c(-0.7,0.3,1,0.4),c(-0.1,0.3,0.4, 1)); sigma

# create the mean vector
mu = c(0,0,0,0) 
# generate the multivariate normal distribution
X = t(mvrnorm(n=n, mu=mu, Sigma=sigma))

D = X
```


<table class="dataframe">
<caption>A matrix: 4 × 4 of type dbl</caption>
<tbody>
	<tr><td> 1.0</td><td>-0.8</td><td>-0.7</td><td>-0.1</td></tr>
	<tr><td>-0.8</td><td> 1.0</td><td> 0.3</td><td> 0.3</td></tr>
	<tr><td>-0.7</td><td> 0.3</td><td> 1.0</td><td> 0.4</td></tr>
	<tr><td>-0.1</td><td> 0.3</td><td> 0.4</td><td> 1.0</td></tr>
</tbody>
</table>



 [Log-likehood function](https://statlect.com/fundamentals-of-statistics/multivariate-normal-distribution-maximum-likelihoo)
 where 
 
 $\log L(\mathbf{x}_i | \mu, \Sigma_p) = -\frac{1}{2} \left[ p \log(2\pi) + \log|\Sigma_p| + (\mathbf{x}_i - \mu)^T \Sigma_p^{-1} (\mathbf{x}_i - \mu) \right]
$


```R
#YOUR CODE HERE
compute_log_likelihood = function(X, mu, Sigma_p) {
  n = ncol(X)  
  p = nrow(X)  
  
  # Compute the |sigma| and (sigma)^-1
  det_Sigma_p = det(Sigma_p)
  inv_Sigma_p = solve(Sigma_p)
  
  # Init
  log_likelihood = 0
  
  # TODO: log-likelihood 
  for (i in 1:n) {
    x_i = X[, i]
    log_likelihood_i = -0.5*(p*log(2*pi)+log(det_Sigma_p)+t(x_i-mu)%*%inv_Sigma_p%*%(x_i-mu))
    log_likelihood = log_likelihood + log_likelihood_i
  }
  
  return(log_likelihood)
}

# TODO: why D? 
log_likelihood = compute_log_likelihood(X, mu, sigma)
log_likelihood

```


<table class="dataframe">
<caption>A matrix: 1 × 1 of type dbl</caption>
<tbody>
	<tr><td>-28.86446</td></tr>
</tbody>
</table>



**B.2 (c) [3 points] Write an R function that computes the posterior distribution of $$\Sigma_p$$ given the data $$X$$.**


```R
# dim(X)    4 10
```


```R
# n = 10; p = 4

# # create the variance covariance matrix
# sigma = rbind(c(1,-0.8,-0.7, -0.1), c(-0.8,1, 0.3, 0.3), 
#               c(-0.7,0.3,1,0.4),c(-0.1,0.3,0.4, 1)); sigma

# # create the mean vector
# mu = c(0,0,0,0) 
# # generate the multivariate normal distribution
# X = t(mvrnorm(n=n, mu=mu, Sigma=sigma))
```


```R
#YOUR CODE HERE
compute_posterior_dist <- function(X, nu_0, S_0, mu_0) {
  n = dim(X)[2]  # 10
  p = dim(X)[1]  # 4
  
  X_mean = apply(X, 1, mean)  
  
  S = matrix(0, nrow = p, ncol = p)
  for (i in 1:n) {
    S = S + (X[, i] - X_mean)%*%t(X[, i] - X_mean)
  }
  
  
  # Updated parameters
  nu_n = nu_0 + n
  S_n = S_0 + S 
  
  #TODO: list    error: multi-argument returns are not permitted
  return(list(nu_n = nu_n, S_n = S_n))
}

# from last cell 
nu_0 = 3  
S_0 =diag(4)    #[1 1 1 1 ]
mu_0 = c(0, 0, 0, 0)  

# X 
posterior_params = compute_posterior_dist(X, nu_0, S_0, mu_0)
posterior_params


```


    Error in S_0 + S: non-conformable arrays
    Traceback:


    1. compute_posterior_dist(X, nu_0, S_0, mu_0)


**B.2 (d) [3 points] Compute the posterior mean and compare it to the sample covariance matrix** 

$$$$C = \left(\sum^n_{i=1}\mathbf{X}_i\mathbf{X}_i^T\right)\bigg/n.$$$$

**Why is the posterior mean close to $$C$$?**


```R
# Sigma_beta = solve(t(X)%*%solve(Sigma_n)%*%X + solve(Sigma_p));
# cat("The Bayesian estimate of the standard errors are given by", sqrt(diag(Sigma_beta)))

# mu_1 = (Sigma_beta)%*%t(X)%*%solve(Sigma_n)%*%y + solve(Sigma_p)%*%c(-100, -100, -100); # try another b_mlc(-100, -100, -100)
# cat("\n The mean of the posterior is given by", mu_1)
# summary(lmod)
```


```R
# n = 10; p = 4

# # create the variance covariance matrix
# sigma = rbind(c(1,-0.8,-0.7, -0.1), c(-0.8,1, 0.3, 0.3), 
#               c(-0.7,0.3,1,0.4),c(-0.1,0.3,0.4, 1)); sigma

# # create the mean vector
# mu = c(0,0,0,0) 
# # generate the multivariate normal distribution
# X = t(mvrnorm(n=n, mu=mu, Sigma=sigma))
# X
# A = matrix(runif((p)^2,0,10), ncol = p);A

# Sigma_p = t(A)%*%A; Sigma_p
# Sigma_n = sig2hat*diag(1,ncol = n, nrow = n);
```


```R
n = 10 
p = 3     # only way it works
mu = rep(0, p)  
mu_0 = rep(0, p)
sigma = diag(1, p)  
X = mvrnorm(n, mu, sigma) 
y = X%*%c(1, 2, -1) + rnorm(n, 0, 1) 
Sigma_p = diag(10, p)
Sigma_n = diag(1, n)
```


```R
# my portesior is not close to C, if 1 is consideed closed.
```


```R
#YOUR CODE HERE
# Posterior covariance
Sigma_beta = solve(t(X)%*%solve(Sigma_n)%*% X+solve(Sigma_p))

# Posterior mean of 
mu_1 = Sigma_beta%*%(t(X)%*%solve(Sigma_n)%*%y + solve(Sigma_p)%*%mu_0)

# =sample covariance matrix C
C = (t(X)%*%X)/nrow(X)

print("Posterior mean of regression coefficients (mu_1):")
mu_1

#TODO:  difference size!! 
cat("Difference with C:",mu_1 - diag(C))

```

    [1] "Posterior mean of regression coefficients (mu_1):"



<table class="dataframe">
<caption>A matrix: 3 × 1 of type dbl</caption>
<tbody>
	<tr><td> 1.0389349</td></tr>
	<tr><td> 2.0114440</td></tr>
	<tr><td>-0.9549358</td></tr>
</tbody>
</table>



    Difference with C: -0.5502289 1.576373 -2.082411

**B.2 (e) [8 points] Conduct the Monte Carlo analysis described on pages 52-53 and 55 in *Bayesian Statistical Methods*. Does the estimated mean match the answers from B.2 (d)?**


```R
#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer
```


    Error in fail(): could not find function "fail"
    Traceback:




```R
#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer
```

YOUR ANSWER HERE

## B.3 Samples from a multivariate normal with unknown mean and variance

Consider the following Bayesian modeling scenario:

\begin{align*}
X = (\mathbf{X}_1,...,\mathbf{X}_n) &\sim N(\underbrace{\boldsymbol\mu}_{p\times 1}, \Sigma_p), \,\,\,\, \mathbf{X}_i = (X_1,...,X_p)^T, \,\,\, p = 3 \\
\Sigma_p &\sim \text{Inv-Wishart}(v_0,S_0), \,\,\, v_0 = 3, \,\,\, S_0 = I_p \\
\boldsymbol\mu \, | \, \Sigma_p &\sim N(\boldsymbol\mu_0, \Sigma_p\big/k_0), \,\,\, \boldsymbol\mu_0 = (0,0,0)^T, \,\,\, k_0 = 5
\end{align*}


The parameters $$v_0$$ and $$S_0$$ describe the degrees of freedom and the scale matrix for the inverse-Wishart distribution on $$\Sigma_p$$. The remaining parameters are the prior mean, $$\boldsymbol\mu_0$$, and the number of "prior measurements", $$k_0$$, on the $$\Sigma_p$$ scale. Assume that $$\Sigma_p$$ and $$\boldsymbol\mu $$ are indepedent.

It can be shown that $$(\mu, \, \Sigma_p \, | \, X)$$ has a joint posterior distribution of the form:

\begin{align*}
\pi_1(\boldsymbol\mu, \, \Sigma_p \, | \, X) \propto |\Sigma_p|^{-((v_1 + p)/2 + 1)}\exp{\left\{-\frac{1}{2}tr\left(S_1\Sigma_p^{-1}  \right) - \frac{k_1}{2}(\mu - \mu_1)^T\Sigma_p^{-1}(\mu - \mu_1)\right\}},
\end{align*}

where 

- $$v_1 = v_0 + n$$

- $$k_1 = k_0 + n$$

- $$\boldsymbol\mu_1 = \frac{k_0}{k_1}\mu_0 + \frac{n}{k_1}\bar{\mathbf{x}}$$

- $$S_1 = S_0 + W + \frac{k_0n}{k_1}(\bar{\mathbf{x}} - \mathbf{\mu_0})(\bar{\mathbf{x}} - \mathbf{\mu_0})^T$$

- $$W = \displaystyle \sum^n_{i = 1}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T $$

**[15 points] We can use these results to generate samples from the joint posterior distribution on $$(\mu, \Sigma_p)\, | \, X$$, according to the following procedure:**

- Draw $$\Sigma_p \, | \, X \sim \text{Inv-Wishart}(v_1,S_1)$$.
- Then draw $$\boldsymbol\mu \, | \, \Sigma_p,X \sim N(\mu_1, \Sigma_p/k_1)$$

**Let the output of this problem be the mean of your realizations of $$\Sigma_p \, | \, X$$ and  $$\boldsymbol\mu \, | \, \Sigma_p,X$$.**

The "read only" cells  contain functions from the Unit #3 code and synthetic data generated using those functions.


```R
#random draw from a MV Normal
rmvn = function(m,mu,sigma){
    p = dim(sigma)[1]
    theta = matrix(NA, nrow = p, ncol = m)
    for (i in 1:m){
        z = rnorm(p,0,1)
        L = chol(sigma)
        theta[,i] = mu + L%*%z
    }
    return(theta)
}

#random draw from a Wishart (invert output for inv Wishart)
rwish = function(v,p,S){
    alpha = matrix(NA, ncol = v, nrow = p)
    for (i in 1:v){
        alpha[,i] = rmvn(1,0,S); 
    }
    W = matrix(0, ncol = p, nrow = p)

    for (i in 1:p){
        W = W + (alpha[,i]%*%t(alpha[,i]))
    }
    return(W)
    }


rwish = function(v,n,S){
    alpha = matrix(NA, ncol = v, nrow = n)
    for (i in 1:v){
        alpha[,i] = rmvn(1,0,S); 
    }

    #dim((alpha[,1]%*%t(alpha[,1])))
    W = matrix(0, ncol = n, nrow = n)

    for (i in 1:n){
        W = W + (alpha[,i]%*%t(alpha[,i]))
    }
    return(W)
    }


#rwish(v,n,sigma)

```


```R
set.seed(101)
#data 
n = 50000
p = 3

#cov matrix
sigma = rbind(c(1,0.6,0.1), c(0.6,1, 0.3), 
              c(0.1,0.3,1)); 

#mvn data
x = rmvn(n,0,sigma); 
mean(x[,1])
```


-0.195620648266004


- $$v_1 = v_0 + n$$

- $$k_1 = k_0 + n$$

- $$\boldsymbol\mu_1 = \frac{k_0}{k_1}\mu_0 + \frac{n}{k_1}\bar{\mathbf{x}}$$

- $$S_1 = S_0 + W + \frac{k_0n}{k_1}(\bar{\mathbf{x}} - \mathbf{\mu_0})(\bar{\mathbf{x}} - \mathbf{\mu_0})^T$$

- $$W = \displaystyle \sum^n_{i = 1}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T $$
\begin{align*}
X = (\mathbf{X}_1,...,\mathbf{X}_n) &\sim N(\underbrace{\boldsymbol\mu}_{p\times 1}, \Sigma_p), \,\,\,\, \mathbf{X}_i = (X_1,...,X_p)^T, \,\,\, p = 3 \\
\Sigma_p &\sim \text{Inv-Wishart}(v_0,S_0), \,\,\, v_0 = 3, \,\,\, S_0 = I_p \\
\boldsymbol\mu \, | \, \Sigma_p &\sim N(\boldsymbol\mu_0, \Sigma_p\big/k_0), \,\,\, \boldsymbol\mu_0 = (0,0,0)^T, \,\,\, k_0 = 5
\end{align*}
 I was not able to recreate the S_1, I am just printing out the  result when I was able to


```R
n = 500
p = 3
v_0 = 3
S_0 = diag(p)  
k_0 = 5
mu_0 = rep(0, p)  
x_bar = mean(x[,1])

v_1 = v_0 + n
k_1 = k_0 + n
mu_1 = (k_0 * mu_0 + n * x_bar) / k_1
mu_1
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>-0.19368381016436</li><li>-0.19368381016436</li><li>-0.19368381016436</li></ol>




```R
#YOUR CODE HERE
invwishart_normal <- function() {
  Sigma_p_sample = rwish(v_1, p, solve(S_1))  #   Inv-Wishart
  mu_sample = rmvn(1, mu_1, Sigma_p_sample / k_1)  #  Normal
  return(list(Sigma_p = Sigma_p_sample, mu = mu_sample))
}

# Draw 
sample = invwishart_normal()

# P and mu
sample$Sigma_p
sample$mu

```


<table class="dataframe">
<caption>A matrix: 3 × 3 of type dbl</caption>
<tbody>
	<tr><td> 0.014934567</td><td>-0.010290483</td><td> 0.002286311</td></tr>
	<tr><td>-0.010290483</td><td> 0.019552750</td><td>-0.003971491</td></tr>
	<tr><td> 0.002286311</td><td>-0.003971491</td><td> 0.010347126</td></tr>
</tbody>
</table>




<table class="dataframe">
<caption>A matrix: 3 × 1 of type dbl</caption>
<tbody>
	<tr><td>-0.2046692</td></tr>
	<tr><td>-0.1940573</td></tr>
	<tr><td>-0.2024512</td></tr>
</tbody>
</table>




```R
#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer
```

## B.4 Bayesian linear regression

Consider data on the number of hours of exercise per week versus the risk of developing a disease. We might look at two sets of patients, those below the age of 50 and those over the age of 50. 

**B.4 (a) [4 points] In the cell below, we read the dataset from https://www.colorado.edu/amath/sites/default/files/attached-files/simp.txt. Explore the data graphically and numerically. Are there relationships between variables?**


```R
library(ggplot2)
df = read.table("https://www.colorado.edu/amath/sites/default/files/attached-files/simp.txt", header = TRUE, sep = "\t")

df$$p = df$$p/100 #turns risk into a probability

#the next few lines creates a factor that is 1 if the individual is older than 50 and 0 otherwise.
x = ifelse(df$age > 50, 1,0)
#df = cbind(df,x)
dfu = df[x == 0,]
dfo = df[x == 1,]
df$$fifty = as.factor(ifelse(df$$age > 50, 1,0))

```


```R
head(df)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>ages</th><th scope=col>hours</th><th scope=col>p</th><th scope=col>fifty</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;fct&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>50</td><td>3.6239201</td><td>0.2952386</td><td>0</td></tr>
	<tr><th scope=row>2</th><td>26</td><td>2.4600381</td><td>0.2210206</td><td>0</td></tr>
	<tr><th scope=row>3</th><td>35</td><td>3.8070524</td><td>0.2095405</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>28</td><td>3.0390379</td><td>0.1829365</td><td>0</td></tr>
	<tr><th scope=row>5</th><td>43</td><td>0.7089589</td><td>0.3309662</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>45</td><td>4.4194311</td><td>0.2976093</td><td>0</td></tr>
</tbody>
</table>




```R
#YOUR CODE HERE
summary(df)
ggplot(df, aes(x = hours, y = p, color = fifty)) +
  geom_point() +
  labs(title = "Age vs Hours vs Probability",
       x = "Hours of exercise per week ",
       y = "Risk of developing a disease") +
  theme_bw() 


```


          ages           hours               p           fifty  
     Min.   :20.00   Min.   :-0.4344   Min.   :0.07764   0:103  
     1st Qu.:36.00   1st Qu.: 2.6493   1st Qu.:0.23956   1: 97  
     Median :50.00   Median : 4.1273   Median :0.32868          
     Mean   :51.48   Mean   : 4.2498   Mean   :0.33694          
     3rd Qu.:67.00   3rd Qu.: 5.7218   3rd Qu.:0.43225          
     Max.   :85.00   Max.   : 9.1519   Max.   :0.58640          



    
![png](STAT5630_Sp23_HW4_files/STAT5630_Sp23_HW4_50_1.png)
    


The more you excercise the less Risk of developing a disease.
Ffity and above has more  Risk of developing a disease

**B.4 (b) [10 points] Fit a Bayesian regression model with `p` as the repsonse and `hours`, as a predictor/feature.**

Notes:

- For your prior, assume that you are pretty unsure about the value of the intercept and slope parameters, before observing the data. You also think that these parameters are uncorrelated.

- For $$\Sigma_n$$, assume that each response measurement ($$p$$) is uncorrelated with each other. Further, assume that each response has constant variance equal to $$\sigma^2 = 0.1$$.


```R
# #posterior
# # better inverse?
# Sigma_beta = solve(t(X)%*%solve(Sigma_n)%*%X + solve(Sigma_p));
# cat("The Bayesian estimate of the standard errors are given by", sqrt(diag(Sigma_beta)))

# mu_1 = (Sigma_beta)%*%t(X)%*%solve(Sigma_n)%*%y + solve(Sigma_p)%*%c(-100, -100, -100); # try another b_mlc(-100, -100, -100)
# cat("\n The mean of the posterior is given by", mu_1)
```


```R
#YOUR CODE HERE
n = nrow(df)
#X = cbind(as.matrix(XX[,1:3])); 
X = cbind(1, df$hours)  
y = df$p

# Prior 
b_ml = c(0, 0)  
Sigma_p <- diag(c(1000, 1000))  

# Sigma_p = t(A)%*%A; Sigma_p
# Sigma_n = sig2hat*diag(1,ncol = n, nrow = n);
# Likelihood 
sigma2 = 0.1  
Sigma_n = sigma2*diag(n)

# Posterior 
Sigma_beta = solve(solve(Sigma_p) + t(X)%*%solve(Sigma_n)%*%X)
mu_1 = Sigma_beta%*%(solve(Sigma_p)%*%b_ml + t(X)%*%solve(Sigma_n)%*%y)

#
cat("The Bayesian estimate of the standard errors are given by", sqrt(diag(Sigma_beta)))

cat("\n The mean of the posterior is given by", mu_1)

lmod1 = lm(p ~ hours, data = df)
summary(lmod1)

```

    The Bayesian estimate of the standard errors are given by 0.05000504 0.01052449
     The mean of the posterior is given by 0.2775577 0.01397199


    
    Call:
    lm(formula = p ~ hours, data = df)
    
    Residuals:
          Min        1Q    Median        3Q       Max 
    -0.279701 -0.085412 -0.008813  0.076613  0.266713 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept) 0.277558   0.018034  15.391  < 2e-16 ***
    hours       0.013972   0.003796   3.681 0.000299 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 0.114 on 198 degrees of freedom
    Multiple R-squared:  0.06405,	Adjusted R-squared:  0.05933 
    F-statistic: 13.55 on 1 and 198 DF,  p-value: 0.0002994



**B.4 (c) [3 points] Construct a $$95\%$$ credible interval for the parameter associated with `hours`.**


```R
#YOUR CODE HERE

# z score qnorm(0.975)
lower_bound = 0.01397199 - qnorm(0.975)*sqrt(0.01052449)
upper_bound = 0.01397199 + qnorm(0.975)*sqrt(0.01052449)
cat("95% credible interval is ",lower_bound, "to", upper_bound)
```

    95% credible interval is  -0.1870986 to 0.2150426

**B.4 (d) [8 points] Construct a posterior predictive distribution for `p` at the `x_star` given below. Print the mean and $$95\%$$ credible interval from the posterior predictive distribution.**


```R
lmod_total = lm(p ~ hours, data = df)
X = model.matrix(lmod_total)
x_star = data.frame(t(colMeans(X)));  x_star
names(x_star) = c("intercept", "hours"); 
```


<table class="dataframe">
<caption>A data.frame: 1 × 2</caption>
<thead>
	<tr><th scope=col>X.Intercept.</th><th scope=col>hours</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>4.249805</td></tr>
</tbody>
</table>




```R
# Sigma_beta
```


```R
#YOUR CODE HERE
fail()
```


    Error in fail(): could not find function "fail"
    Traceback:



**B.4 (e) [4 points] Now fit a Bayesian regression model with `p` as the repsonse and `hours` and `fifty`, as predictors/features.**

Notes:

- For your prior, assume that you are pretty unsure about the value of the intercept and slope parameters, before observing the data. You also think that these parameters are uncorrelated.

- For $$\Sigma_n$$, assume that each response measurement ($$p$$) is uncorrelated with each other. Further, assume that each response has constant variance equal to $$\sigma^2 = 0.002$$.


```R
# n = nrow(df)
# #X = cbind(as.matrix(XX[,1:3])); 
# X = cbind(1, df$hours)  
# y = df$p

# # Prior 
# b_ml = c(0, 0)  
# Sigma_p <- diag(c(1000, 1000))  

# # Sigma_p = t(A)%*%A; Sigma_p
# # Sigma_n = sig2hat*diag(1,ncol = n, nrow = n);
# # Likelihood 
# sigma2 = 0.1  
# Sigma_n = sigma2*diag(n)

# # Posterior 
# Sigma_beta = solve(solve(Sigma_p) + t(X) %*% solve(Sigma_n)%*%X)
# mu_1 = Sigma_beta%*%(solve(Sigma_p)%*%b_ml + t(X) %*% solve(Sigma_n)%*%y)

# #
# cat("The Bayesian estimate of the standard errors are given by", sqrt(diag(Sigma_beta)))

# cat("\n The mean of the posterior is given by", mu_1)
```


```R
#YOUR CODE HERE
n = nrow(df)
X = model.matrix(~ hours + fifty, data = df)  
y = df$p

# Prior    TODO: add 1 more zero
b_ml = c(0, 0, 0) 
Sigma_p = diag(c(1000, 1000, 1000))  

# Likelihood 
sigma2 = 0.002  
Sigma_n = sigma2 * diag(n)

# Posterior calculations
Sigma_beta = solve(solve(Sigma_p) + t(X) %*% solve(Sigma_n) %*% X)
mu_post = Sigma_beta %*% (solve(Sigma_p) %*% b_ml + t(X) %*% solve(Sigma_n) %*% y)

# 
cat("The Bayesian estimate of the standard errors are given by", sqrt(diag(Sigma_beta)))

cat("\n The mean of the posterior is given by", mu_post)


lmod2 = lm(p ~ hours + fifty, data = df)
summary(lmod2)
```

    The Bayesian estimate of the standard errors are given by 0.007185967 0.001937324 0.008235905
     The mean of the posterior is given by 0.3184424 -0.02576613 0.2639065


    
    Call:
    lm(formula = p ~ hours + fifty, data = df)
    
    Residuals:
          Min        1Q    Median        3Q       Max 
    -0.100261 -0.034454 -0.000067  0.032765  0.242985 
    
    Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  0.318442   0.008269   38.51   <2e-16 ***
    hours       -0.025766   0.002229  -11.56   <2e-16 ***
    fifty1       0.263907   0.009477   27.85   <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 0.05146 on 197 degrees of freedom
    Multiple R-squared:  0.8104,	Adjusted R-squared:  0.8085 
    F-statistic:   421 on 2 and 197 DF,  p-value: < 2.2e-16



**B.4 (f) [5 points] Compare the parameter estimate associated with `hours` in B.4 (b) with the estimate associated with `hours` in B.4 (e). Do these values seem inconsistent or in tension with each other? Explain your answer. Which value do you think would be the better estimate of the effect of exercise hours on risk of disease?**

Of course in tension with each other. second one is better since this one allows the model to account the `fifty` while estimate, give us me inferences.


```R
#YOUR CODE HERE
lmod1_coef = coef(summary(lmod1))["hours", "Estimate"]; lmod1_coef
lmod2_coef = coef(summary(lmod2))["hours", "Estimate"]; lmod2_coef


```


0.0139718627340631



-0.0257661371737063

