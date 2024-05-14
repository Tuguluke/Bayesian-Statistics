# Homework #6

**See Canvas for this assignment and due date**. Complete all of the following problems. Ideally, the theoretical problems should be answered in a Markdown cell directly underneath the question. If you don't know LaTex/Markdown, you may submit separate handwritten solutions to the theoretical problems, but please see the [class scanning policy](https://docs.google.com/document/d/17y5ksolrn2rEuXYBv_3HeZhkPbYwt48UojNT1OvcB_w/edit?usp=sharing). Please do not turn in messy work. Computational problems should be completed in this notebook (using the R kernel). Computational questions may require code, plots, analysis, interpretation, etc. Working in small groups is allowed, but it is important that you make an effort to master the material and hand in your own work. 

# A. Theoretical Problems

### A.1 Justifications for Monte Carlo

Let $$X_1,...,X_n \overset{iid}{\sim}\text{binomial}(1, \theta)$$. Define $$\widehat\theta = \frac{1}{n}\sum_{i=1}^n X_i$$ and $$\widehat\sigma^2 = \widehat\theta(1-\widehat\theta)$$.

**A.1 (a) [6 points] Show that $$\widehat\theta$$ is an unbiased estimator of $$\theta$$ and that the variance of $$\widehat\theta$$ converges to zero as $$n\to\infty$$.**

First:

Expectation of binomial is $$\theta$$,


$$E[\sum_{i=1}^n X_i] =\sum_{i=1}^n E[X_i]  = n \theta$$

therefore

$$E[\widehat\theta] = E[\frac{1}{n}\sum_{i=1}^n X_i] = \frac{1}{n}E[\sum_{i=1}^n X_i]  = \theta $$

We showed that the $$\widehat\theta$$ is an unbiased estimator of $$\theta$$.

Second:

Variace of binomial is $$n\theta(1-\theta)$$,

$$Var(\sum_{i=1}^n X_i) =\sum_{i=1}^n Var(X_i)  = n\theta(1-\theta) $$

therefore

$$Var(\widehat\theta) = Var(\frac{1}{n}\sum_{i=1}^n X_i) = \frac{1}{n^2}Var(\sum_{i=1}^n X_i) = \frac{1}{n}\theta(1-\theta)$$

We get $$\lim_{n \to \infty}Var(\widehat\theta) \to 0$$.

Recall from MathStat that a sequence of random variables, $$Y_1, Y_2,...$$ *converges in probability* to $$Y$$ if, for every $$\epsilon > 0$$

\begin{align*}
\lim_{n\to \infty} P\left(|Y_n - Y| \le \epsilon \right) = 1.
\end{align*}

If $$Y_n$$ converges in probability to $$Y$$ we write $$Y_n \overset{P}{\to} Y$$. 

**A.1 (b) [4 points] Show that $$\widehat\theta \overset{P}{\to} \theta$$.**

HINT: You can use a theorem from MathStat about convergence in probability.

[Chebyshev's inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality) stated that $${\displaystyle P(|X-\mu |\geq k\sigma )\leq {\frac {1}{k^{2}}}.}$$

Let $$\widehat\theta = X$$ and $$\theta = \mu$$, $$\sigma^2 = Var(\widehat\theta) = \frac{1}{n}\theta(1-\theta)$$ and $
k\sigma = \epsilon$,


we have 

$$P(|\widehat\theta- \theta |\geq \epsilon )  \leq  \frac{\sigma^2}{\epsilon^2} = \frac{1}{n\epsilon^2}\theta(1-\theta)$$

therefore

$$P(|\widehat\theta- \theta |\leq \epsilon ) = 1 - P(|\widehat\theta- \theta |\geq \epsilon ) =  1  - \frac{1}{n\epsilon^2}\theta(1-\theta)$$

hence 

$$\lim_{n \to \infty}P(|\widehat\theta- \theta |\leq \epsilon )  = \lim_{n \to \infty} 1  - \frac{1}{n\epsilon^2}\theta(1-\theta) \to 1$$

we showed that $$\widehat\theta \overset{P}{\to} \theta$$

**A.1 (c) [6 points] Show that $$\widehat\sigma^2 \overset{P}{\to} \theta(1-\theta)$$.**

HINT: You can use a theorem from MathStat about convergence in probability.

Office hour hint: [Continuous mapping theorem](https://en.wikipedia.org/wiki/Continuous_mapping_theorem) stated that 
$${\displaystyle {\begin{aligned}X_{n}\ {\xrightarrow {\text{d}}}\ X\quad &\Rightarrow \quad g(X_{n})\ {\xrightarrow {\text{d}}}\ g(X);\\[6pt]X_{n}\ {\xrightarrow {\text{p}}}\ X\quad &\Rightarrow \quad g(X_{n})\ {\xrightarrow {\text{p}}}\ g(X);\\[6pt]X_{n}\ {\xrightarrow {\!\!{\text{a.s.}}\!\!}}\ X\quad &\Rightarrow \quad g(X_{n})\ {\xrightarrow {\!\!{\text{a.s.}}\!\!}}\ g(X).\end{aligned}}}$$


we know that $$\widehat\theta \overset{P}{\to} \theta$$.

consider the function $$f(x) = x(1-x)$$, 

if apply this to $$\widehat\theta$$, we get $$f(\widehat\theta) = \widehat\theta(1-\widehat\theta) = \widehat\sigma^2$$.

if apply  to $$\theta$$, we get $$f(\theta) = \theta(1-\theta)$$


therefore

$$\underbrace{f(\widehat\theta)}_{\widehat\sigma^2} \overset{P}{\to} \underbrace{f(\theta)}_{ \theta(1-\theta)}$$

we showed that $$\widehat\sigma^2 \overset{P}{\to} \theta(1-\theta)$$.

Suppose that $$\theta_1,...,\theta_m$$ are draws from the posterior distribution $$\pi(\theta\, | \, \mathbf{y})$$, for some data $$\mathbf{Y}$$. Futher, suppose that the posterior distribution is binomial. 

**A.1 (d) [2 points]  Use the above results to come up with a Monte Carlo estimator of the posterior variance, $$\sigma^2_1$$.**

$$\sigma^2_1 = \dfrac{1}{m}\sum_{i - 1}^{m}\theta_i(1-\theta_i)$$

# B. Computational Problems


## B.1 Rejection Sampling

The following admissions dataset was collected to learn relationships between variables:

- `gre` Graduate Record Exam scores 

- `gpa` grade point average  

- `rank` rank of the undergraduate institution

- `admit` admission into graduate school (`0 = not admitted`, `1 = admitted`)

Assume that the gre variable is approximately normally distributed. We will consider regressing `x = gpa` on `y = gre`, i.e., 

$$$$y_i = \beta_0 + \beta_1x_i + \varepsilon_i, \,\,\,\, \varepsilon_i\overset{iid}{\sim}N(0,107^2)$$$$

Consider the R code and output below. Then answer the questions below.


```R
admission = read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
head(admission)

#data 
df = admission[,c(2,3)]; n = length(df$gre); 

#prior N(0,100*I_n)
mu0 = c(0,0); sigma_p = matrix(c(100,0,0,100), ncol = 2); 

#standard deviation assumption
sig = 107


#frequentist
lm_gre = lm(gre ~ gpa, df)
summary(lm_gre)

```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>admit</th><th scope=col>gre</th><th scope=col>gpa</th><th scope=col>rank</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>380</td><td>3.61</td><td>3</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>660</td><td>3.67</td><td>3</td></tr>
	<tr><th scope=row>3</th><td>1</td><td>800</td><td>4.00</td><td>1</td></tr>
	<tr><th scope=row>4</th><td>1</td><td>640</td><td>3.19</td><td>4</td></tr>
	<tr><th scope=row>5</th><td>0</td><td>520</td><td>2.93</td><td>4</td></tr>
	<tr><th scope=row>6</th><td>1</td><td>760</td><td>3.00</td><td>2</td></tr>
</tbody>
</table>




    
    Call:
    lm(formula = gre ~ gpa, data = df)
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -302.394  -62.789   -2.206   68.506  283.438 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)   192.30      47.92   4.013 7.15e-05 ***
    gpa           116.64      14.05   8.304 1.60e-15 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 106.8 on 398 degrees of freedom
    Multiple R-squared:  0.1477,	Adjusted R-squared:  0.1455 
    F-statistic: 68.95 on 1 and 398 DF,  p-value: 1.596e-15



**B.1 (a) [6 points] Write a function that computes $$g(\boldsymbol\beta \, | \, \mathbf{x}, \mathbf{y})$$, the log of the posterior distribution of $$\boldsymbol\beta = (\beta_0, \beta_1)^T$$ conditioned on the gpa and gre data, up to a normalizing constant. Provide a visualization of $$g(\boldsymbol\beta \, | \, \mathbf{x}, \mathbf{y})$$.**

''Assume that the gre variable is approximately normally distributed. We will consider regressing `x = gpa` on `y = gre`, i.e., 

$$$$y_i = \beta_0 + \beta_1x_i + \varepsilon_i, \,\,\,\, \varepsilon_i\overset{iid}{\sim}N(0,107^2)$$$$''


```R
#YOUR CODE HERE
log_posterior = function(beta, x, y, sigma, mu0, sigma_p){
  # Log likelihood
  residuals = y - (beta[1] + beta[2]*x)
  log_like = -0.5 * sum((residuals/sigma)^2)
  
  # Log prior
  log_prior = -0.5 * t(beta - mu0)%*%solve(sigma_p)%*%(beta - mu0)
  
  # Log posterior TODO: Constant?
  log_posterior = log_like + log_prior
  return(log_posterior)
}
#TODO: Visiual
```

**B.1 (b) [3 points] Find the MAP of the posterior distribution of $$\boldsymbol\beta$$.**

Note: Use the `laplace()` or `optim()` function.


```R
df =  admission[, c("gpa", "gre")]
admission$gpa
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>3.61</li><li>3.67</li><li>4</li><li>3.19</li><li>2.93</li><li>3</li><li>2.98</li><li>3.08</li><li>3.39</li><li>3.92</li><li>4</li><li>3.22</li><li>4</li><li>3.08</li><li>4</li><li>3.44</li><li>3.87</li><li>2.56</li><li>3.75</li><li>3.81</li><li>3.17</li><li>3.63</li><li>2.82</li><li>3.19</li><li>3.35</li><li>3.66</li><li>3.61</li><li>3.74</li><li>3.22</li><li>3.29</li><li>3.78</li><li>3.35</li><li>3.4</li><li>4</li><li>3.14</li><li>3.05</li><li>3.25</li><li>2.9</li><li>3.13</li><li>2.68</li><li>2.42</li><li>3.32</li><li>3.15</li><li>3.31</li><li>2.94</li><li>3.45</li><li>3.46</li><li>2.97</li><li>2.48</li><li>3.35</li><li>3.86</li><li>3.13</li><li>3.37</li><li>3.27</li><li>3.34</li><li>4</li><li>3.19</li><li>2.94</li><li>3.65</li><li>2.82</li><li>3.18</li><li>3.32</li><li>3.67</li><li>3.85</li><li>4</li><li>3.59</li><li>3.62</li><li>3.3</li><li>3.69</li><li>3.73</li><li>4</li><li>2.92</li><li>3.39</li><li>4</li><li>3.45</li><li>4</li><li>3.36</li><li>4</li><li>3.12</li><li>4</li><li>2.9</li><li>3.07</li><li>2.71</li><li>2.91</li><li>3.6</li><li>2.98</li><li>3.32</li><li>3.48</li><li>3.28</li><li>4</li><li>3.83</li><li>3.64</li><li>3.9</li><li>2.93</li><li>3.44</li><li>3.33</li><li>3.52</li><li>3.57</li><li>2.88</li><li>3.31</li><li>3.15</li><li>3.57</li><li>3.33</li><li>3.94</li><li>3.95</li><li>2.97</li><li>3.56</li><li>3.13</li><li>2.93</li><li>3.45</li><li>3.08</li><li>3.41</li><li>3</li><li>3.22</li><li>3.84</li><li>3.99</li><li>3.45</li><li>3.72</li><li>3.7</li><li>2.92</li><li>3.74</li><li>2.67</li><li>2.85</li><li>2.98</li><li>3.88</li><li>3.38</li><li>3.54</li><li>3.74</li><li>3.19</li><li>3.15</li><li>3.17</li><li>2.79</li><li>3.4</li><li>3.08</li><li>2.95</li><li>3.57</li><li>3.33</li><li>4</li><li>3.4</li><li>3.58</li><li>3.93</li><li>3.52</li><li>3.94</li><li>3.4</li><li>3.4</li><li>3.43</li><li>3.4</li><li>2.71</li><li>2.91</li><li>3.31</li><li>3.74</li><li>3.38</li><li>3.94</li><li>3.46</li><li>3.69</li><li>2.86</li><li>2.52</li><li>3.58</li><li>3.49</li><li>3.82</li><li>3.13</li><li>3.5</li><li>3.56</li><li>2.73</li><li>3.3</li><li>4</li><li>3.24</li><li>3.77</li><li>4</li><li>3.62</li><li>3.51</li><li>2.81</li><li>3.48</li><li>3.43</li><li>3.53</li><li>3.37</li><li>2.62</li><li>3.23</li><li>3.33</li><li>3.01</li><li>3.78</li><li>3.88</li><li>4</li><li>3.84</li><li>2.79</li><li>3.6</li><li>3.61</li><li>2.88</li><li>3.07</li><li>3.35</li><li>2.94</li><li>3.54</li><li>3.76</li><li>3.59</li><li>3.47</li><li>3.59</li><li>3.07</li><li>3.23</li><li>3.63</li><li>3.77</li><li>3.31</li><li>3.2</li><li>4</li><li>3.92</li><li>3.89</li><li>3.8</li><li>3.54</li><li>3.63</li><li>3.16</li><li>3.5</li><li>3.34</li><li>3.02</li><li>2.87</li><li>3.38</li><li>3.56</li><li>2.91</li><li>2.9</li><li>3.64</li><li>2.98</li><li>3.59</li><li>3.28</li><li>3.99</li><li>3.02</li><li>3.47</li><li>2.9</li><li>3.5</li><li>3.58</li><li>3.02</li><li>3.43</li><li>3.42</li><li>3.29</li><li>3.28</li><li>3.38</li><li>2.67</li><li>3.53</li><li>3.05</li><li>3.49</li><li>4</li><li>2.86</li><li>3.45</li><li>2.76</li><li>3.81</li><li>2.96</li><li>3.22</li><li>3.04</li><li>3.91</li><li>3.34</li><li>3.17</li><li>3.64</li><li>3.73</li><li>3.31</li><li>3.21</li><li>4</li><li>3.55</li><li>3.52</li><li>3.35</li><li>3.3</li><li>3.95</li><li>3.51</li><li>3.81</li><li>3.11</li><li>3.15</li><li>3.19</li><li>3.95</li><li>3.9</li><li>3.34</li><li>3.24</li><li>3.64</li><li>3.46</li><li>2.81</li><li>3.95</li><li>3.33</li><li>3.67</li><li>3.32</li><li>3.12</li><li>2.98</li><li>3.77</li><li>3.58</li><li>3</li><li>3.14</li><li>3.94</li><li>3.27</li><li>3.45</li><li>3.1</li><li>3.39</li><li>3.31</li><li>3.22</li><li>3.7</li><li>3.15</li><li>2.26</li><li>3.45</li><li>2.78</li><li>3.7</li><li>3.97</li><li>2.55</li><li>3.25</li><li>3.16</li><li>3.07</li><li>3.5</li><li>3.4</li><li>3.3</li><li>3.6</li><li>3.15</li><li>3.98</li><li>2.83</li><li>3.46</li><li>3.17</li><li>3.51</li><li>3.13</li><li>2.98</li><li>4</li><li>3.67</li><li>3.77</li><li>3.65</li><li>3.46</li><li>2.84</li><li>3</li><li>3.63</li><li>3.71</li><li>3.28</li><li>3.14</li><li>3.58</li><li>3.01</li><li>2.69</li><li>2.7</li><li>3.9</li><li>3.31</li><li>3.48</li><li>3.34</li><li>2.93</li><li>4</li><li>3.59</li><li>2.96</li><li>3.43</li><li>3.64</li><li>3.71</li><li>3.15</li><li>3.09</li><li>3.2</li><li>3.47</li><li>3.23</li><li>2.65</li><li>3.95</li><li>3.06</li><li>3.35</li><li>3.03</li><li>3.35</li><li>3.8</li><li>3.36</li><li>2.85</li><li>4</li><li>3.43</li><li>3.12</li><li>3.52</li><li>3.78</li><li>2.81</li><li>3.27</li><li>3.31</li><li>3.69</li><li>3.94</li><li>4</li><li>3.49</li><li>3.14</li><li>3.44</li><li>3.36</li><li>2.78</li><li>2.93</li><li>3.63</li><li>4</li><li>3.89</li><li>3.77</li><li>3.76</li><li>2.42</li><li>3.37</li><li>3.78</li><li>3.49</li><li>3.63</li><li>4</li><li>3.12</li><li>2.7</li><li>3.65</li><li>3.49</li><li>3.51</li><li>4</li><li>2.62</li><li>3.02</li><li>3.86</li><li>3.36</li><li>3.17</li><li>3.51</li><li>3.05</li><li>3.88</li><li>3.38</li><li>3.75</li><li>3.99</li><li>4</li><li>3.04</li><li>2.63</li><li>3.65</li><li>3.89</li></ol>




```R
library(LearnBayes)
```


```R
# TODO
x = admission$gpa
y = admission$gre

sigma = 107
mu0 = c(0, 0)
sigma_p = matrix(c(100, 0, 0, 100), ncol = 2)



# Use optim() function 
initial_beta = c(mean(y),1)   # try 0
optim_results = optim(initial_beta, log_posterior, x=x, y=y, sigma=sigma, mu0=mu0, sigma_p=sigma_p)

# $par
MAP_beta = optim_results$par
MAP_beta

# #lpalace
# laplace_beta = laplace(initial_beta, log_posterior, x=x, y=y, sigma=sigma, mu0=mu0, sigma_p=sigma_p)

# MAP_beta = laplace_beta$par
# MAP_beta
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>2.93634592483097e+57</li><li>1.57748763706265e+58</li></ol>



The rejection sampling algorithm requires two user-defined "hyper-parameters": 

1. The proposal distribution for $$\boldsymbol\beta$$: $$\boldsymbol\beta \sim p(\boldsymbol\beta)$$. In this problem, we'll set $$p(\boldsymbol\beta) \equiv N(\boldsymbol\mu, \Sigma)$$, where $$\boldsymbol\mu = (51,154)$$ and $$\Sigma$$ is set to `fit$var`from the call to `fit = laplace()` in the MAP calculation.


2. A constant $$d$$ such that $$\log(g(\boldsymbol\theta | \mathbf{x}, \mathbf{y})) - \log(p(\boldsymbol\theta)) \le d$$ ($$d$$ is the log-transformed version of $$c$$ that shows up in the description of rejection sampling in our notes).


**B.1 (c) [5 points] Find a constant $$d$$ such that $$\log(g(\boldsymbol\theta | \mathbf{x}, \mathbf{y})) - \log(p(\boldsymbol\theta)) \le d$$.**


```R
#YOUR CODE HERE
# I can not get fit$var to work
```

**B.1 (d) [10 points] Write a loop that performs rejection sampling algorithm.**

NOTE: No credit will be awarded for using a built-in `R` rejection sampling algorithm! Also, to complete the rest of this question, it might be helpful to write your rejection sampling algorithm as a function, say, `rs = function(data, mu, Sigma, dmax, m)...`. 


```R
#YOUR CODE HERE

```

**B.1 (e) [6 points] Provide some numerical and graphical evidence that rejection sampling has worked!**


```R
#YOUR CODE HERE
```

YOUR ANSWER HERE

Rejection sampling is sensitive to the choice of proposal distribution. To see this, let's use the same proposal distribution above, *except* that we'll change $$\Sigma$$. Construct $$\Sigma$$ according to:

`A = matrix(runif(4,0,100), ncol = 2);  Sigma = t(A)%*%A;`

**B.1 (f) [5 points] Now run rejection sampling again, with the new proposal variance-covariance matrix. Provide evidence that the draws resulting from rejection sampling are not from the correct distribution.**


```R
#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer
```

YOUR ANSWER HERE

### B.2 MCMC Regression 

In this problem, we will use the Metropolis-Hastings algorithm on a linear regression. Our goal will be to predict the miles per gallon of a given vehicle from the vehicle's weight and horsepower. First, we read in the full dataset; select, center, and scale the relevant predictors; and fit a frequentist regression (with the `lm()` function) for reference.


```R
auto_full = read.csv(paste0("https://raw.githubusercontent.com/bzaharatos/",
                            "-Statistical-Modeling-for-Data-Science-Applications/",
                            "master/Modern%20Regression%20Analysis%20/",
                            "Datasets/auto-mpg.csv"), sep = ",")
auto_full = na.omit(auto_full) #removing the rows that have an NA for horsepower
auto_full$$cylinders = as.factor(auto_full$$cylinders)
#we'll work with standardized continuous predictors
auto_predictors = scale(auto_full[,c(4,5)])

#df contains the unstandardized response and standardized continuous predictors
df = data.frame(mpg = auto_full$$mpg, cylinders = auto_full$$cylinders, auto_predictors)
summary(df)

lmod = lm(mpg ~ horsepower + weight, data = df)
summary(lmod)
```


          mpg        cylinders   horsepower          weight       
     Min.   : 9.00   3:  4     Min.   :-1.5190   Min.   :-1.6065  
     1st Qu.:17.00   4:199     1st Qu.:-0.7656   1st Qu.:-0.8857  
     Median :22.75   5:  3     Median :-0.2850   Median :-0.2049  
     Mean   :23.45   6: 83     Mean   : 0.0000   Mean   : 0.0000  
     3rd Qu.:29.00   8:103     3rd Qu.: 0.5594   3rd Qu.: 0.7501  
     Max.   :46.60             Max.   : 3.2613   Max.   : 2.5458  



    
    Call:
    lm(formula = mpg ~ horsepower + weight, data = df)
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -11.0762  -2.7340  -0.3312   2.1752  16.2601 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  23.4459     0.2142 109.478  < 2e-16 ***
    horsepower   -1.8207     0.4267  -4.267 2.49e-05 ***
    weight       -4.9216     0.4267 -11.535  < 2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 4.24 on 389 degrees of freedom
    Multiple R-squared:  0.7064,	Adjusted R-squared:  0.7049 
    F-statistic: 467.9 on 2 and 389 DF,  p-value: < 2.2e-16



Next, we'll get the data into a nice format for the Metropolis-Hastings algorithm. Below, `X` is the design matrix for our regression. `y` is our response. `data` combines the two, and will be passed in to the log-posterior function.


```R
X = as.matrix(cbind(1, df[,c(3,4)])) #design matrix
y = df[,1];  #response

#data in the format that we'll need in MH
data = cbind(y,X)
head(data)


```


<table class="dataframe">
<caption>A matrix: 6 × 4 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>y</th><th scope=col>1</th><th scope=col>horsepower</th><th scope=col>weight</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>18</td><td>1</td><td>0.6632851</td><td>0.6197483</td></tr>
	<tr><th scope=row>2</th><td>15</td><td>1</td><td>1.5725848</td><td>0.8422577</td></tr>
	<tr><th scope=row>3</th><td>18</td><td>1</td><td>1.1828849</td><td>0.5396921</td></tr>
	<tr><th scope=row>4</th><td>16</td><td>1</td><td>1.1828849</td><td>0.5361602</td></tr>
	<tr><th scope=row>5</th><td>17</td><td>1</td><td>0.9230850</td><td>0.5549969</td></tr>
	<tr><th scope=row>6</th><td>15</td><td>1</td><td>2.4299245</td><td>1.6051468</td></tr>
</tbody>
</table>



Assume that $$\mathbf{Y} \, | \, \boldsymbol\beta, X, \sigma^2 \sim N\left(X\boldsymbol\beta, \sigma^2I_n\right)$$, with $$\sigma = 4.24$$. Also, assume a normal prior distribution on each $$\beta_j$$ parameter: $$\beta_j \overset{iid}{\sim} N(0,10^2)$$ for $$j = 0,...,2$$.

**B.2 (a) [10 points] Write a function that takes in the parameters of the regression model, $$\boldsymbol\beta = (\beta_0, \beta_1, \beta_2)^T$$, and the data from directly above (`data`), and returns the log of the posterior distribution, up to a normalizing constant.**


```R
# data[, 1]
```


```R
#YOUR CODE HERE
log_posterior = function(beta, data) {
  # correct?
  Y = data[, 1]
  X = data[, -1]
  sigma = 4.24
  
  #residuals
  residuals = Y - X%*%beta  ## TODO
  
  # Log likelihood
  log_likelihood = -0.5/sigma^2*sum(residuals^2)
  
  # Log prior 
  log_prior = -0.5/100*sum(beta^2)
  
  # Log posterior 
  log_posterior = log_likelihood + log_prior
  
  return(log_posterior)
}

```

**B.2 (b) [25 points] Write a Metropolis-Hastings algorithm function with the hyperparameters given below. The M-H algorithm will, after a "burn-in period", provide (near) random draws from the posterior distribution on $$\boldsymbol\beta \, | \, \mathbf{y}, X, \sigma^2$$. For efficiency, the algorithm should be written on the log-scale. Use the random draws produced by your function to estimate $$E\left[\boldsymbol\beta \, | \, \mathbf{y}, X, \sigma^2\right]$$ (HINT: use `colMeans()`, but only include rows after a burn-in period of `T/4` rows.). Compare these values to the frequentist estimates in the summary regression table above.**

The function should take in the following hyperparameters:

1. The initial value for the parameter vector: `beta0 = rep(1,3)`

2. The standard deviations for the multivariate normal proposal distribution: `proposal_sd = rep(0.25,3)`

3. The number of iterations: `T = 20000`.

4. The data: `data` from above.

The output from this function should be a $$T \times 3$$ matrix, where, after some "burn-in period", is a random draw from the multivariate posterior for $$\boldsymbol\beta \, | \, \mathbf{y}, X, \sigma^2$$. 


```R
# mhs = function(theta0,proposal_sd){
#     samples = matrix(NA,T,2)
#     theta = theta0
#     for (t in 1:T){  
#         proposal = rmvnorm(1,theta,diag(proposal_sd)); #sample a proposal
#         logR = betabinexch(proposal, cancermortality) - 
#             betabinexch(theta,cancermortality); #compute the log of the ratio
#         if(log(runif(1))<logR){
#             theta = proposal #accept/reject the proposal
#             }
#             samples[t,] = theta #store the accept/reject in the chain
#             }
#     return(samples)
# }
# #YOUR CODE HERE
# set.seed(89119)
# #initial values
# library(mvtnorm)

# T = 10000

# theta = c(0,0)
# theta2 = c(10,10)
# theta3 = c(-100,-100)
# proposal_sd = c(1,1)

# #three chains
# chain1 = mhs(theta,proposal_sd)
# chain2 = mhs(theta2,proposal_sd)
# chain3 = mhs(theta3,proposal_sd)


# burnin = 500
# dimnames(chain1)[[2]]=c("logit eta","log K")
# #xyplot(mcmc(chain1[-c(1:burnin),]),col="black") #in lattice
# plot(chain1[,1], type = "l")
# lines(chain2[,1], type = "l", col = "red")
# lines(chain3[,1], type = "l", col = "blue")

# colMeans(chain1) - post.means
```


```R
#YOUR CODE HERE
mhs = function(beta0, proposal_sd, T, data) {
  # Sample = 
  beta_matrix = matrix(NA, nrow = T, ncol = length(beta0))
    # Error in X %*% beta: non-conformable arguments!!!
  beta_matrix[1, ] = beta0  # for (t in 2:T) 
  
  #TODO: why not 1?
  for (t in 2:T) {
    current_beta = beta_matrix[t - 1, ]
    
    proposal = current_beta + rnorm(length(beta0), mean = 0, sd = proposal_sd)
    
    # log of acceptance ratio
    logR = log_posterior(proposal, data) - log_posterior(current_beta, data)
    
    # <= 
      #         if(log(runif(1))<logR){
#             theta = proposal #accept/reject the proposal
#             }
#             samples[t,] = theta #store the accept/reject in the chain
#             }
    if (log(runif(1)) < logR) {
      beta_matrix[t, ] = proposal  # Better
    } else {
      beta_matrix[t, ] = current_beta  # Reject 
    }
  }
  
  return(beta_matrix)
}

```


```R
#YOUR CODE HERE
T = 20000
beta_matrix = mhs(beta0 = rep(1, 3), proposal_sd = rep(0.25, 3), T = T, data = data)

# HINT: use colMeans(), but only include rows after a burn-in period of T/4 rows.
burnin =  T/4
beta_expected = colMeans(beta_matrix[(burnin + 1):T, ])

beta_expected

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>23.4287573921022</li><li>-1.77069537173608</li><li>-4.96624056546842</li></ol>




```R
#YOUR CODE HERE

colnames(beta_matrix) # = c("beta_0", "beta_1", "beta_2")
head(beta_matrix)
```


    NULL



<table class="dataframe">
<caption>A matrix: 6 × 3 of type dbl</caption>
<tbody>
	<tr><td>1.0000000</td><td>1.0000000</td><td>1.0000000</td></tr>
	<tr><td>0.9467836</td><td>0.7951415</td><td>0.6127080</td></tr>
	<tr><td>1.3583185</td><td>0.5896286</td><td>0.5389478</td></tr>
	<tr><td>1.2805934</td><td>0.3636962</td><td>0.3592996</td></tr>
	<tr><td>1.2805934</td><td>0.3636962</td><td>0.3592996</td></tr>
	<tr><td>1.8350979</td><td>0.1590043</td><td>0.1343445</td></tr>
</tbody>
</table>



**B.2 (c) [6 points] Plot the Markov chains for each of the parameters as a function of index. Do the chains appear to be mixing well after the burn-in period?**  


```R
# burnin = 500
# dimnames(chain1)[[2]]=c("logit eta","log K")
# #xyplot(mcmc(chain1[-c(1:burnin),]),col="black") #in lattice
# plot(chain1[,1], type = "l")
# lines(chain2[,1], type = "l", col = "red")
# lines(chain3[,1], type = "l", col = "blue")
```


```R
#YOUR CODE HERE
# Plot the chains after the burn-in period
burnin  =  T / 4 # 0

plot(beta_matrix[(burnin + 1):nrow(beta_matrix),1], type = "l", 
    ylim = range(beta_matrix[(burnin + 1):nrow(beta_matrix), ]),  # Set the y-axis range to fit all data
     col = "black")
lines(beta_matrix[(burnin + 1):nrow(beta_matrix),2], col = "red")
lines(beta_matrix[(burnin + 1):nrow(beta_matrix),3], col = "blue")

```


    
![png](STAT4630-5630_Sp24_HW6_files/STAT4630-5630_Sp24_HW6_43_0.png)
    


Yes, it mixed well.

Above, we *assumed* that the error variance was $$\sigma^2 = 4.24^2$$. But let's estimate it instead, based on the data!

**B.2 (d) [5 points] Imagine that the prior distribution on $$\sigma^2 \, | \, \boldsymbol\beta, \mathbf{y}, X$$ is $$\text{inv-}\Gamma(a_0 = 1,b_0 = 1)$$. Estimate $$\sigma^2 \, | \, \boldsymbol\beta, \mathbf{y}, X$$ using the mean of the inverse gamma:**

$$$$E\left(\sigma^2 \, | \, \boldsymbol\beta, \mathbf{y}, X\right) = \frac{b}{a-1},$$$$

where 

- $$a = a_0 + n/2$$

- $$\displaystyle b = b_0 + \frac{\sum_{i=1}^n\left(y_i - \beta_0 - \beta_1x_{i,1} - \beta_2x_{i,2}\right)^2}{2}$$

- $$x_{i,1}$$ is the $$i^{th}$$ value of horsepower (from the design matrix, `X`, above)

- $$x_{i,2}$$ is the $$i^{th}$$ value of weight (again, from the design matrix, `X`, above)

- $$\beta_0, \beta_1, \beta_2$$, are the column means from Metropolis-Hastings, computed in B.1 (b). 

$$b$$ can be computed easily with `sum((y - X%*%beta_means)^2)`.

Store $$E\left(\sigma^2 \, | \, \boldsymbol\beta, \mathbf{y}, X\right)$$ in `sig2`.


```R
#YOUR CODE HERE
# beta_expected = colMeans(beta_matrix[(burn_in + 1):T, ])
beta_means = colMeans(beta_matrix[(burnin + 1):T, ])
beta_means
# knowns
a0 = 1  
b0 = 1 
a = a0 + n / 2
b = b0 + sum((y - X%*%beta_means)^2) / 2

# sigma^2
sig2 = b / (a - 1)

sig2
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>23.4287573921022</li><li>-1.77069537173608</li><li>-4.96624056546842</li></ol>




17.4905225043721


Finally, we're going to calculate the posterior predictive distribution (ppd) at the mean value of horsepower and weight, `xstar`(convince yourself that zeros are the right values to predict at the means!). 

The first three lines of code produce a frequentist prediction point estimate (`fit`) and interval estimate (`lwr`, `upr`) at `xstar`. The last line stores `xstar` as numeric to be used in the ppd. 


```R
xstar = data.frame(0,0) #data frame with prediction values (1 at the beginning for the intercept in the predict function)
colnames(xstar) = colnames(data)[3:4]
predict(lmod, xstar, interval = "prediction") #frequentist prediction, for reference

xstar = as.numeric(xstar) #relavant values, for ppd algorithm
```


<table class="dataframe">
<caption>A matrix: 1 × 3 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>fit</th><th scope=col>lwr</th><th scope=col>upr</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>23.44592</td><td>15.09877</td><td>31.79306</td></tr>
</tbody>
</table>



**B.2 (e) [10 points] ] Use `xstar`, your random draws from $$\boldsymbol\beta \, | \, \mathbf{y}, X, \sigma^2$$ (`samples`) and $$\sigma^2 \, | \, \mathbf{y}, X, \boldsymbol\beta$$ (`sig2`) from above to estimate the posterior predictive distribution. Calculate the Monte Carlo mean and a $$95\%$$ credible interval from this distribution.**

Note that our Monte Carlo ppd algorithm is:

1. draw a posterior sample: $$\boldsymbol\beta^* \sim \pi(\boldsymbol\beta \, | \, \mathbf{y}, X, \sigma^2)$$. That is, generate a random number from the posterior. **But we've already done this with Metropolis-Hastings!**

2. then generate a random number from the likelihood function, with parameter $$\boldsymbol\beta^*$$: $$y^* \, | \, \boldsymbol\beta^*, x^*, \sigma^2 \sim f(\mathbf{y^*} \, | \, \boldsymbol\beta^*, x^*, \sigma^2)$$ 

3. repeat 1-2 $$m$$ (many) times.

The values of $$y^*$$ will follow the posterior predictive distribution. Don't forget to exclude the burn-in! 🔥 


```R
#YOUR CODE HERE

# Random draw (step 1)
samples = beta_matrix[(burnin + 1):T, ]

# y^star initialization
ystar = numeric(nrow(samples))

# step 2
for (i in 1:nrow(samples)) {
  beta_star = samples[i, ]  
  ystar[i] = rnorm(1, mean = as.matrix(xstar)%*%beta_star, sd = sqrt(sig2))  # sig2 = 17.8472654085071
}

# TODO: around this 
# y_star
mc_mean = mean(ystar)

# Calculate the Monte Carlo mean and a  95%  credible interval from this distributiomn
lwr = quantile(ystar, probs = 0.025)
upr = quantile(ystar, probs = 0.975)

# Results
# mc_mean
c('Monte_Carlo_Mean = ', mc_mean)
c('95% credible interval = ',c(lwr, upr))

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'Monte_Carlo_Mean = '</li><li>'-0.0512056949140401'</li></ol>




<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>1</dt><dd>'95% credible interval = '</dd><dt>2.5%</dt><dd>'-8.2284243160326'</dd><dt>97.5%</dt><dd>'8.18392342604224'</dd></dl>



### B.3 Multimodal posterior with MCMC?

Consider `x` in the cell below to be data. Let's suppose that a group of two researchers, neither of which knows the true mean `mu_true`, wishs to estimate it with a Bayesian posterior. However, there is disagreement among the researchers' priors:

- Researcher #1: $$\mu \sim N(0,1)$$

- Researcher #2: $$\mu \sim N(50,1)$$

Further, Researcher #1 and Researcher #2 have roughly equal expertise about the data generating process, with researcher #1 being slightly more knowledeable. More precisely, we weight each researchers' prior according to `alpha = c(0.6,0.4)`.


**B.2 (a) [5 points] Construct a prior distribution as a mixture of each individual researcher's prior. Plot the prior.**


```R
set.seed(5630)
n = 7; mu_true = 45; sig_true = 35; x = rnorm(n,mu_true,sig_true)
```


```R
#YOUR CODE HERE
alpha = c(0.6, 0.4)  
mu1 = 0
sigma1 = 1  
mu2 = 50
sigma2 = 1 


# func 
mixture_density = function(mu){
  alpha[1] * dnorm(mu, mean = mu1, sd = sigma1) +
    alpha[2] * dnorm(mu, mean = mu2, sd = sigma2)
}

# [0 50]
mu_values = seq(-10, 60, by = 0.1)


# Plot +- 10
plot(mu_values, sapply(mu_values, mixture_density),
     type = 'l',
     xlab = "Mu", 
     ylab = "Prior",
     col = "blue")

```


    
![png](STAT4630-5630_Sp24_HW6_files/STAT4630-5630_Sp24_HW6_53_0.png)
    


**B.2 (b) [4 points] Construct and plot the likelihood function for the data.**


```R
#YOUR CODE HERE
set.seed(5630)
n = 7; mu_true = 45; sig_true = 35; x = rnorm(n,mu_true,sig_true)

sigma_2 = var(x)

# mixture_density = function(mu){
#   alpha[1] * dnorm(mu, mean = mu1, sd = sigma1) +
#     alpha[2] * dnorm(mu, mean = mu2, sd = sigma2)
# }

log_likelihood = function(mu, x, sigma_2) {
  n = 7
  -n/2 * log(2 * pi * sigma_2) - sum((x - mu)^2)/(2 * sigma_2)
}

mu_values = seq(-10, 60, by = 0.1)

# why not n = len(x)
likelihood_values = exp(sapply(mu_values, log_likelihood, x=x, sigma_2=sigma_2))


plot(mu_values, likelihood_values, type='l',
     xlab="Mu", ylab="Likelihood",
    col = 'red')
```


    
![png](STAT4630-5630_Sp24_HW6_files/STAT4630-5630_Sp24_HW6_55_0.png)
    


**B.2 (c) [10 points] Construct and plot the posterior up to a normalizing constant. Describe it's form.**


```R
#YOUR CODE HERE
# done
mixture_density = function(mu){
  result = alpha[1] * dnorm(mu, mean = mu1, sd = sigma1) +
    alpha[2] * dnorm(mu, mean = mu2, sd = sigma2)
    return(result) 
}

# done
log_likelihood = function(mu, x, sigma_2) {
  n = 7
  -n/2 * log(2 * pi * sigma_2) - sum((x - mu)^2)/(2 * sigma_2)
}


# log_post
log_posterior = function(mu, x, sigma_2) {
  log_prior = log(mixture_density(mu))
  log_likelihood = log_likelihood(mu, x, sigma_2)
  log_post = log_prior + log_likelihood
  return(log_post)
}

# 
mu_values = seq(-10, 60, by = 0.1)
log_posterior_values = sapply(mu_values, log_posterior, x=x, sigma_2=sigma_2)

# non log
posterior_values = exp(log_posterior_values - max(log_posterior_values))  # TODO:

# 
plot(mu_values, posterior_values, type='l',
     xlab="Mu", ylab="Posterior",
    col = 'green')

```


    
![png](STAT4630-5630_Sp24_HW6_files/STAT4630-5630_Sp24_HW6_57_0.png)
    


similar to prior, but more weight on $$mu = 0$$, just like HW1

**B.2 (d) [12 points] Use Metropolis-Hastings to attempt to draw samples from the posterior distribution. In particular, set:**

- $$\mu^{(0)} = 0$$


- The proposal to be normal with variance = 1.


- $$T = 50,000$$

**Produce a trace plot and comment on the convergence of the chain.**


```R
# iterations = 50000

# # smaller burn_in
# burn_in = iterations * 0.1

# # 
# adapt = Metro_Hastings(log_post, beta, prop_sigma = NULL,
#                         par_names = NULL, iterations = iterations, burn_in = burn_in,
#                         adapt_par = c(100, 20, 0.5, 0.75), quiet = FALSE,
#                         Y = Y, N = N, t = t, tau = tau)

# plot(adapt$trace[, 1], type="l", xlab="Iteration", ylab=expression(beta[0]))

# # Plot the trace of the second parameter, beta[1]
# plot(adapt$trace[, 2], type="l", xlab="Iteration", ylab=expression(beta[1]))


# mhs = function(beta0, proposal_sd, T, data) {
#   # Sample = 
#   beta_matrix = matrix(NA, nrow = T, ncol = length(beta0))
#     # Error in X %*% beta: non-conformable arguments!!!
#   beta_matrix[1, ] = beta0  # for (t in 2:T) 
  
#   #TODO: why not 1?
#   for (t in 2:T) {
#     current_beta = beta_matrix[t - 1, ]
    
#     proposal = current_beta + rnorm(length(beta0), mean = 0, sd = proposal_sd)
    
#     # log of acceptance ratio
#     logR = log_posterior(proposal, data) - log_posterior(current_beta, data)
    
#     # <= 
#       #         if(log(runif(1))<logR){
# #             theta = proposal #accept/reject the proposal
# #             }
# #             samples[t,] = theta #store the accept/reject in the chain
# #             }
#     if (log(runif(1)) < logR) {
#       beta_matrix[t, ] = proposal  # Better
#     } else {
#       beta_matrix[t, ] = current_beta  # Reject 
#     }
#   }
  
#   return(beta_matrix)
# }

```


```R
#YOUR CODE HERE
set.seed(123)  

# Given
mu0 = 0
T = 50000
proposal_sd = sqrt(1)  

# Init
mu_chain = numeric(T)
mu_chain[1] = mu0  

# nhs
for (t in 2:T) {
  # sample
  current_mu = mu_chain[t - 1]
  
  #     proposal = current_beta + rnorm(length(beta0), mean = 0, sd = proposal_sd)
    
  proposal_mu = rnorm(1, mean = current_mu, sd = proposal_sd)
  
  # Calculate log of acceptance ratio
  logR = log_posterior(proposal_mu, x, sigma_2) - log_posterior(current_mu, x, sigma_2)
  
  # #     if (log(runif(1)) < logR) {
#       beta_matrix[t, ] = proposal  # Better
#     } else {
#       beta_matrix[t, ] = current_beta  # Reject 
#     }
#   }
    
  if (log(runif(1)) < logR) {
    mu_chain[t] = proposal_mu  # Accept
  } else {
    mu_chain[t] = current_mu  # Reject
  }
}

```


```R
#YOUR CODE HERE
plot(mu_chain, type = 'l',  
     xlab = "T", ylab = "Mu")
```


    
![png](STAT4630-5630_Sp24_HW6_files/STAT4630-5630_Sp24_HW6_62_0.png)
    


Convergence Looking good, but as good as adaptive from Unit 4 note

**B.2 (e) [6 points] Now, rerun the Metropolis algorithm with the proposal variance of $$100^2$$. Comment on the resulting trace plot.**


```R
#YOUR CODE HERE
set.seed(123)  

# only change here!!
proposal_sd = sqrt(100^2)  

# Init
mu_chain = numeric(T)
mu_chain[1] = mu0  

# nhs
for (t in 2:T) {
  # sample
  current_mu = mu_chain[t - 1]
  
  #     proposal = current_beta + rnorm(length(beta0), mean = 0, sd = proposal_sd)
    
  proposal_mu = rnorm(1, mean = current_mu, sd = proposal_sd)
  
  # Calculate log of acceptance ratio
  logR = log_posterior(proposal_mu, x, sigma_2) - log_posterior(current_mu, x, sigma_2)
  
  # #     if (log(runif(1)) < logR) {
#       beta_matrix[t, ] = proposal  # Better
#     } else {
#       beta_matrix[t, ] = current_beta  # Reject 
#     }
#   }
    
  if (log(runif(1)) < logR) {
    mu_chain[t] = proposal_mu  # Accept
  } else {
    mu_chain[t] = current_mu  # Reject
  }
}
plot(mu_chain, type = 'l',  
     xlab = "T", ylab = "Mu")
```


    
![png](STAT4630-5630_Sp24_HW6_files/STAT4630-5630_Sp24_HW6_65_0.png)
    


Now it looks worse


```R

```
