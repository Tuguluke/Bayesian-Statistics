# Homework #5

**See Canvas for this assignment and due date**. Complete all of the following problems. Ideally, the theoretical problems should be answered in a Markdown cell directly underneath the question. If you don't know LaTex/Markdown, you may submit separate handwritten solutions to the theoretical problems, but please see the [class scanning policy](https://docs.google.com/document/d/17y5ksolrn2rEuXYBv_3HeZhkPbYwt48UojNT1OvcB_w/edit?usp=sharing). Please do not turn in messy work. Computational problems should be completed in this notebook (using the R kernel). Computational questions may require code, plots, analysis, interpretation, etc. Working in small groups is allowed, but it is important that you make an effort to master the material and hand in your own work. 

## A.1 Non-identically distributed Poisson data

Assume $$Y_i|\lambda \sim \text{Poisson}(N_i\lambda)$$ for $$i = 1, \ldots, n$$.

**A.1 (a) [10 points] Identify a conjugate prior for $$\lambda$$ and derive the posterior that follows from this prior.**

From the textbook, we know the CP of Poisson is Gamma, since likelihood of Poisson is $$\Pi\dfrac{e^{-N_i\lambda}(N_i\lambda)^{y_i}}{y_i!}$$, which share the similar kernel as Gamma.

Therefore, the posterior $$\propto \Pi\dfrac{e^{-N_i\lambda}(N_i\lambda)^{y_i}}{y_i!} *\dfrac{\beta^\alpha\lambda^{\alpha - 1}e^{-\beta\lambda}}{\Gamma(\alpha)}$$

Collection the similar terms of $$\lambda$$ and $$e$$, we have

$$\propto e^{-\beta\lambda  - \sum N_i\lambda}\lambda^{\alpha - 1 + \sum y_i} \propto \lambda^{(\alpha  + \sum y_i) - 1}e^{-(\beta  + \sum N_i)\lambda}$$ where $$\sum  = \sum_{i =1}^n$$

**A.1 (b) [8 points] Using the prior $$\lambda \sim \text{Uniform}(a, b)$$, derive the MAP estimator of $$\lambda$$.**

we know the prior of uniform is $$\dfrac{1}{b-a}, \, \forall \lambda \in (a,b)$$  
Likelihood of  poisson is :$$\Pi\dfrac{e^{-N_i\lambda}(N_i\lambda)^{y_i}}{y_i!}$$

therefore the posterior $$\propto \Pi\dfrac{e^{-  N_i\lambda}(N_i\lambda)^{y_i}}{y_i!}\dfrac{1}{b-a} \propto e^{-\lambda\sum N_i}\lambda^{\sum {y_i}}(\Pi N_i^{y_i})$$, where $$\sum  = \sum_{i =1}^n$$

By taking the log:

$$\log e^{-\lambda\sum N_i}(\sum N_i^{y_i})\lambda^{\sum {y_i}} = -\lambda\sum N_i + (\sum {y_i})\log \lambda$$

and set the derivative to zero:
$$\dfrac{d}{d\lambda}\left(-\lambda\sum N_i + (\sum {y_i})\log \lambda\right) = 0 \Longleftrightarrow -\sum N_i + (\sum {y_i})\dfrac{1}{\lambda} = 0 $$

$$\lambda =\dfrac{\sum_{i = 1}^n {y_i}}{\sum_{i = 1}^n N_i}$$

**A.1 (c) [5 points] Using the prior $$\lambda \sim \text{Uniform}(0, 20)$$, plot the posterior on a grid of $$\lambda$$ assuming $$n=2$$, $$N_1 =50$$, $$N_2 =100$$, $$Y_1 =12$$, and $$Y_2 =25$$ and show that the MAP estimate is indeed the maximizer.**


```R
#YOUR CODE HERE
fail()
```


    Error in fail(): could not find function "fail"
    Traceback:



## A.2 Regression log posterior

Assume that $$\mathbf{Y}\, | \, X, \boldsymbol\beta \sim N\left(X\boldsymbol\beta, \Sigma_n\right)$$, where:

- $$X$$ is a $$n \times p$$ design matrix with a column of ones, and two columns of predictor measurements ($$p = 3$$). 

- $$\boldsymbol{\beta} = (\beta_0, \beta_2, \beta_3)^T$$ is a vector of regression parameters.

- $$\Sigma_n = \sigma^2I_n$$, where $$\sigma = 0.11$$ is known.

- $$\beta_j \overset{iid}{\sim} N(0,1)$$, $$j = 1,2,3$$. 

**A.2 (a) [15 points] Analytically derive the log posterior distribution.**

Likelihood: 

$$(2\pi\sigma^2)^{-n/2}exp\left(-\dfrac{1}{2\sigma^2}(\mathbf{Y} - X\beta)^2(\mathbf{Y} - X\beta)\right)$$

From Unit 3 note page 10, we know prior:

$$\pi(\beta) \propto exp\left(-\dfrac{1}{2}\beta^T\beta\right)$$, when $$\mu = 1$$

therefore, we take the log of the posterior

$$\log \pi(\beta |\mathbf{Y}, X)  \propto -\frac{1}{2\sigma^2} (\mathbf{Y} - X\beta)^T (\mathbf{Y} - X\beta) - \frac{1}{2} \beta^T \beta $$

**A.2 (b) [STAT 5630 Only, 5 points] Briefly describe the relationship between the maximizer of the log posterior distribution in (a) and the solution to ridge regression.**

According to the text book page 127. equation (4.18), the solution to ridge regression is 

$\beta_R = \arg\min_{\beta} \sum_{i=1}^{n} \left( Y_i - \sum_{j=1}^{p} X_{ij}\beta_j \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 
$

Comparing maximizer of the log posterior distribution in (a)  
$
 -\frac{1}{2\sigma^2} (Y - X\beta)^T (Y - X\beta) - \frac{1}{2} \beta^T \beta 
$


we can see they look similar. 

## B. Computational Problems

## B.1 Bayesian Regression modeling in R continued

The following dataset containts measurements related to the impact of three advertising medias on sales of a product, $$P$$. The variables are:

- `youtube`($$\mathbf{x}_1$$): the advertising budget allocated to YouTube. Measured in thousands of dollars;

- `facebook` ($$\mathbf{x}_2$$): the advertising budget allocated to Facebook. Measured in thousands of dollars; and 

- `newspaper`($$\mathbf{x}_3$$): the advertising budget allocated to a local newspaper. Measured in thousands of dollars.

- `sales`($$\mathbf{Y}$$): the value in the $$i^{th}$$ row of the sales column is a measurement of the sales (in thousands of units) for product $$P$$ for company $$i$$.

The advertising data treat "a company selling product $$P$$" as the statistical unit, and "all companies selling product $$P$$" as the population. We assume that the $$n = 200$$ companies in the dataset were chosen at random from the population (a strong assumption!).

In the Unit #4 code, we found the MAP for a simple linear regression on these (standardized) data. Here, let's find the MAP including all of the predictors. 

Assume that $$\mathbf{Y}\, | \, X, \boldsymbol\beta \sim N\left(X\boldsymbol\beta, \Sigma_n\right)$$, where:

- $$X$$ is a design matrix with a column of 1s and subsequent columns of predictor measurements.

- $$\boldsymbol{\beta} = (\beta_0, \beta_1, \beta_2, \beta_3)^T$$ is a vector of regression parameters.

- $$\Sigma_n = \sigma^2I_n$$.

- We assume $$\sigma^2 = 0.32^2$$ 

- $$\beta_j \sim N(0,1)$$, $$j = 1,2,3$$. 

Another way of stating the above is that, for $$i = 1,...,n$$, 

$$$$Y_i = \beta_0 + \beta_1x_{i,1} + \beta_2x_{i,2} + \beta_3x_{i,3} + \varepsilon_i, \, \, \,  \varepsilon_i \overset{iid}{\sim} N(0,\sigma^2),$$$$

where $$x_{i,1}$$ are the facebook measurements, $$x_{i,2}$$ are the youtube measurements, and $$x_{i,3}$$ are the newspaper measurements. Note that the data are centered and scaled, so we aren't directly estimating $$\boldsymbol\beta = (\beta_0, \beta_1, \beta_2, \beta_3)^T$$ but a transformed version, $$\boldsymbol\alpha = (\alpha_1, \alpha_2, \alpha_3)^T$$, where the intercept $$\alpha_0$$ is fixed at zero.

**[15 points] Compute the MAP for $$\boldsymbol\alpha \, | \, \mathbf{Y}$$ and compare your answer to the least squares solution.** 

Use the `laplace()` function; the rest of the problem will be easier. In the first empty cell, write out any R functions that you will need. In the second empty cell, find the MAP.


```R
library(ggplot2)

marketing = read.table(url("https://raw.githubusercontent.com/bzaharatos/-Statistical-Modeling-for-Data-Science-Applications/master/Modern%20Regression%20Analysis%20/Datasets/marketing.txt"), sep = "")
n = dim(marketing)[1]; p = dim(marketing)[2] - 1

#scaling the data 
databar = colMeans(marketing)
XX = scale(marketing);
df = data.frame(XX); head(df)
X = cbind(as.matrix(XX[,1:3])); p = dim(X)[2]; n = dim(X)[1];dim(X)
y = df$sales; 
#frequentist comparison
lmod_full = lm(sales ~ .-1, data = df)
summary(lmod_full)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>youtube</th><th scope=col>facebook</th><th scope=col>newspaper</th><th scope=col>sales</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td> 0.96742460</td><td> 0.9790656</td><td>1.7744925</td><td> 1.5481681</td></tr>
	<tr><th scope=row>2</th><td>-1.19437904</td><td> 1.0800974</td><td>0.6679027</td><td>-0.6943038</td></tr>
	<tr><th scope=row>3</th><td>-1.51235985</td><td> 1.5246374</td><td>1.7790842</td><td>-0.9051345</td></tr>
	<tr><th scope=row>4</th><td> 0.05191939</td><td> 1.2148065</td><td>1.2831850</td><td> 0.8581768</td></tr>
	<tr><th scope=row>5</th><td> 0.39319551</td><td>-0.8395070</td><td>1.2785934</td><td>-0.2151431</td></tr>
	<tr><th scope=row>6</th><td>-1.61136487</td><td> 1.7267010</td><td>2.0408088</td><td>-1.3076295</td></tr>
</tbody>
</table>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>200</li><li>3</li></ol>




    
    Call:
    lm(formula = sales ~ . - 1, data = df)
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -1.69195 -0.17074  0.04634  0.22795  0.54226 
    
    Coefficients:
               Estimate Std. Error t value Pr(>|t|)    
    youtube    0.753066   0.022895  32.892   <2e-16 ***
    facebook   0.536482   0.024442  21.949   <2e-16 ***
    newspaper -0.004331   0.024444  -0.177     0.86    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 0.3222 on 197 degrees of freedom
    Multiple R-squared:  0.8972,	Adjusted R-squared:  0.8956 
    F-statistic: 573.2 on 3 and 197 DF,  p-value: < 2.2e-16




```R
# b_ml = solve(t(X)%*%X)%*%t(X)%*%y; 
# rss = sum((y - X%*%b_ml)^2)
# sig2hat = rss/(n-(p+1)); sqrt(sig2hat)

# lmod = lm(sales ~ youtube, data = df)   # youtube - 1 in class note, here we are not trying to estimate the intercept
# summary(lmod)
```


```R
#YOUR CODE HERE
log_posterior = function(alpha, X, y) {
  sigma2 = 0.32^2  
  n = nrow(X)

  rss = (y - X %*% alpha)^2
  log_likelihood = -n/2*log(2*pi*sigma2)-sum(rss)/(2*sigma2)
  
  #  beta ~ N(0,1) 
  log_prior = -1/2 * sum(alpha^2)
  log_post = log_likelihood + log_prior
  return(log_post)
}
```


```R
#YOUR CODE HERE
library(LearnBayes)
log_posterior(c(100,100,100),X,y)

fit = laplace(log_posterior, c(100,100,100), X, y)
fit$mode
```


-37905465.9747622



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0.753964805411558</li><li>0.521938916039789</li><li>-0.00101810290030276</li></ol>



```Coefficients:
           Estimate Std. Error t value Pr(>|t|)    
youtube    0.753066   0.022895  32.892   <2e-16 ***
facebook   0.536482   0.024442  21.949   <2e-16 ***
newspaper -0.004331   0.024444  -0.177     0.86   
```
Very similar to least square's result.

## B.2 Running MAP

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

#data storage
y = admission$$gre; x = admission$$gpa; n = length(x); n1 = 1:n;

#prior
mu0 = c(0,0); sigma_p = matrix(c(100,0,0,100), ncol = 2); 

#standard deviation assumption
sig = 107


#frequentist
lm_gre = lm(y ~ x)
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
    lm(formula = y ~ x)
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -302.394  -62.789   -2.206   68.506  283.438 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)   192.30      47.92   4.013 7.15e-05 ***
    x             116.64      14.05   8.304 1.60e-15 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 106.8 on 398 degrees of freedom
    Multiple R-squared:  0.1477,	Adjusted R-squared:  0.1455 
    F-statistic: 68.95 on 1 and 398 DF,  p-value: 1.596e-15



**B.2 (a) [6 points] Given the prior and standard deviation assumptions above, write a function that takes in a `beta` (vector of length = 2), `x = gpa` and `y = gre` and computes the log-posterior distribution, up to a normalizing constant.**


```R
#YOUR CODE HERE
log_posterior = function(beta,x,y){
    X = as.matrix(cbind(1,x));
    log_likelihood = -1/2*sum((y-X%*%beta)^2)
    log_prior = -1/(2*100)*sum(beta^2)
    log_post = log_likelihood + log_prior
    return(log_post)
}

log_posterior(c(200,100),x,y)
```


-2751832


**B.2 (b) [6 points] Given the assumptions and function above, estimate the MAP of $$\boldsymbol\beta = (\beta_0, \beta_1)^T$$.**


```R
#YOUR CODE HERE
fit = laplace(log_posterior,c(200,100),x,y)
fit$mode
# summary(lm_gre)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>191.91977458031</li><li>116.7546105363</li></ol>



**B.2 (c) [10 points] For $$i = 1,...,n$$, compute an $$n \times 2$$ "running MAP" matrix, where the $$i^{th}$$ row contains the MAP for $$\boldsymbol\beta$$ using up to (and including) the $$i^{th}$$ $$(x,y)$$ pair in the data.** 

On unit3 class note page 11, we have

$\beta | \mathbf{X}, \mathbf{y}, \Sigma_n \sim \mathcal{N}\left( \overbrace{\Sigma_{\beta} \left( \Sigma_{p}^{-1} \mu_{\beta} + \mathbf{X}^{\mathrm{T}} \Sigma_{n}^{-1} \mathbf{y} \right)}^{\mu_\beta}, \overbrace{\left( \Sigma_{p}^{-1} + \mathbf{X}^{\mathrm{T}} \Sigma_{n}^{-1} \mathbf{X} \right)^{-1}}^{\Sigma_\beta} \right)
$


```R
#YOUR CODE HERE
#empty n by 2
running_MAP = matrix(0, nrow = n, ncol = 2)
#prior
mu0 = c(0,0); sigma_p = matrix(c(100,0,0,100), ncol = 2); 

sig = 107

for (i in 1:n) {
  # i pair
  X_i = as.matrix(cbind(1, x[1:i]))
  y_i = y[1:i]
  
  # \Sigma_\beta
  Sigma_inv = solve(sigma_p) + (1/sig^2)*t(X_i)%*%X_i
  Sigma_beta = solve(Sigma_inv)
  
  # \mu_\beta
  mu_i = Sigma_beta%*%(solve(sigma_p)%*%mu0 + (1/sig^2)*t(X_i)%*%y_i)
  
  # TODO: 
  running_MAP[i,] = t(mu_i)
  # return update
  mu0 = mu_i
  sigma_p = Sigma_beta
}

running_MAP
```


<table class="dataframe">
<caption>A matrix: 400 × 2 of type dbl</caption>
<tbody>
	<tr><td> 2.956690</td><td> 10.67365</td></tr>
	<tr><td> 9.036657</td><td> 32.90020</td></tr>
	<tr><td>16.003576</td><td> 60.06897</td></tr>
	<tr><td>22.028400</td><td> 81.81453</td></tr>
	<tr><td>26.753973</td><td> 97.62450</td></tr>
	<tr><td>31.264210</td><td>111.01575</td></tr>
	<tr><td>34.912659</td><td>121.09277</td></tr>
	<tr><td>37.456923</td><td>127.96192</td></tr>
	<tr><td>39.394465</td><td>133.11475</td></tr>
	<tr><td>40.876802</td><td>137.47627</td></tr>
	<tr><td>41.981718</td><td>141.54615</td></tr>
	<tr><td>42.702608</td><td>144.37927</td></tr>
	<tr><td>43.181516</td><td>146.98101</td></tr>
	<tr><td>43.806770</td><td>149.35186</td></tr>
	<tr><td>44.261667</td><td>151.34190</td></tr>
	<tr><td>44.561400</td><td>152.74516</td></tr>
	<tr><td>44.743632</td><td>154.15722</td></tr>
	<tr><td>44.669266</td><td>155.25699</td></tr>
	<tr><td>44.516122</td><td>156.44769</td></tr>
	<tr><td>44.353700</td><td>157.28944</td></tr>
	<tr><td>44.119084</td><td>157.96668</td></tr>
	<tr><td>43.863277</td><td>158.59745</td></tr>
	<tr><td>43.776343</td><td>159.18245</td></tr>
	<tr><td>43.782804</td><td>159.76684</td></tr>
	<tr><td>43.861243</td><td>160.39667</td></tr>
	<tr><td>43.886572</td><td>161.09233</td></tr>
	<tr><td>43.885757</td><td>161.69532</td></tr>
	<tr><td>43.910180</td><td>162.11525</td></tr>
	<tr><td>44.082722</td><td>162.56772</td></tr>
	<tr><td>44.202169</td><td>162.93889</td></tr>
	<tr><td>⋮</td><td>⋮</td></tr>
	<tr><td>127.4600</td><td>136.2945</td></tr>
	<tr><td>127.6634</td><td>136.2297</td></tr>
	<tr><td>127.9315</td><td>136.1467</td></tr>
	<tr><td>128.1978</td><td>136.0644</td></tr>
	<tr><td>128.4702</td><td>135.9801</td></tr>
	<tr><td>128.7407</td><td>135.8962</td></tr>
	<tr><td>129.0090</td><td>135.8129</td></tr>
	<tr><td>129.2538</td><td>135.7372</td></tr>
	<tr><td>129.5041</td><td>135.6602</td></tr>
	<tr><td>129.7615</td><td>135.5813</td></tr>
	<tr><td>130.0119</td><td>135.5049</td></tr>
	<tr><td>130.2610</td><td>135.4286</td></tr>
	<tr><td>130.5093</td><td>135.3523</td></tr>
	<tr><td>130.7573</td><td>135.2761</td></tr>
	<tr><td>131.0019</td><td>135.2009</td></tr>
	<tr><td>131.2306</td><td>135.1299</td></tr>
	<tr><td>131.4469</td><td>135.0630</td></tr>
	<tr><td>131.6611</td><td>134.9968</td></tr>
	<tr><td>131.8792</td><td>134.9297</td></tr>
	<tr><td>132.0945</td><td>134.8636</td></tr>
	<tr><td>132.3346</td><td>134.7912</td></tr>
	<tr><td>132.5720</td><td>134.7196</td></tr>
	<tr><td>132.8074</td><td>134.6487</td></tr>
	<tr><td>133.0422</td><td>134.5779</td></tr>
	<tr><td>133.3064</td><td>134.4976</td></tr>
	<tr><td>133.5760</td><td>134.4156</td></tr>
	<tr><td>133.8451</td><td>134.3338</td></tr>
	<tr><td>134.1060</td><td>134.2543</td></tr>
	<tr><td>134.3602</td><td>134.1772</td></tr>
	<tr><td>134.6189</td><td>134.0985</td></tr>
</tbody>
</table>



**B.2 (d) [5 points] Plot each column of this matrix against the vector `1:n`. Interpret the answer.**


```R
#YOUR CODE HERE
plot(1:n, running_MAP[, 1], type = 'l', col = 'blue', 
     xlab = 'Index', ylab = 'First column')

# Plot for the second column of running_MAP (beta_1 estimates)
plot(1:n, running_MAP[, 2], type = 'l', col = 'red', 
     xlab = 'Index', ylab = 'Second column')
```


    
![png](STAT4630-5630_Sp24_HW5_files/STAT4630-5630_Sp24_HW5_29_0.png)
    



    
![png](STAT4630-5630_Sp24_HW5_files/STAT4630-5630_Sp24_HW5_29_1.png)
    


They are converging, it seems like.


```R
dim(running_MAP);
dim(admission)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>400</li><li>2</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>400</li><li>4</li></ol>



**B.2 (e) [10 points] Randomly sample $$200$$ rows from the data frame and use those to compute the MAP. Why might you use a subsample of the data to compute the MAP?**


```R
#YOUR CODE HERE
set.seed(123)
sub_indices = sample(nrow(admission), 200)
sub_data = admission[sub_indices, ]
dim(sub_data)

y = sub_data$$gre; x = sub_data$$gpa; # n = length(x); n1 = 1:n;

fit = laplace(log_posterior,c(200,100),x,y)
fit$mode
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>200</li><li>4</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>166.280854872919</li><li>125.930562366499</li></ol>



Compare with full data

``
191.91977458031    116.7546105363
``

Faster with less memory, maybe no overfitting, hence probably converges to local/global minumum.


```R

```
