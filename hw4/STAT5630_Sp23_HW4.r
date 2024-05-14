set.seed(5630)
n = 50; mu_true = 5; sig_true = 2; x = rnorm(n,mu_true,sig_true)

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

# t(mvrnorm(n=n, mu=mu, Sigma=sigma))

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


# dim(X)    4 10

# n = 10; p = 4

# # create the variance covariance matrix
# sigma = rbind(c(1,-0.8,-0.7, -0.1), c(-0.8,1, 0.3, 0.3), 
#               c(-0.7,0.3,1,0.4),c(-0.1,0.3,0.4, 1)); sigma

# # create the mean vector
# mu = c(0,0,0,0) 
# # generate the multivariate normal distribution
# X = t(mvrnorm(n=n, mu=mu, Sigma=sigma))

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



# Sigma_beta = solve(t(X)%*%solve(Sigma_n)%*%X + solve(Sigma_p));
# cat("The Bayesian estimate of the standard errors are given by", sqrt(diag(Sigma_beta)))

# mu_1 = (Sigma_beta)%*%t(X)%*%solve(Sigma_n)%*%y + solve(Sigma_p)%*%c(-100, -100, -100); # try another b_mlc(-100, -100, -100)
# cat("\n The mean of the posterior is given by", mu_1)
# summary(lmod)

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

n = 10 
p = 3     # only way it works
mu = rep(0, p)  
mu_0 = rep(0, p)
sigma = diag(1, p)  
X = mvrnorm(n, mu, sigma) 
y = X%*%c(1, 2, -1) + rnorm(n, 0, 1) 
Sigma_p = diag(10, p)
Sigma_n = diag(1, n)

# my portesior is not close to C, if 1 is consideed closed.

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


#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer

#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer

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


#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer

library(ggplot2)
df = read.table("https://www.colorado.edu/amath/sites/default/files/attached-files/simp.txt", header = TRUE, sep = "\t")

df$p = df$p/100 #turns risk into a probability

#the next few lines creates a factor that is 1 if the individual is older than 50 and 0 otherwise.
x = ifelse(df$age > 50, 1,0)
#df = cbind(df,x)
dfu = df[x == 0,]
dfo = df[x == 1,]
df$fifty = as.factor(ifelse(df$age > 50, 1,0))


head(df)

#YOUR CODE HERE
summary(df)
ggplot(df, aes(x = hours, y = p, color = fifty)) +
  geom_point() +
  labs(title = "Age vs Hours vs Probability",
       x = "Hours of exercise per week ",
       y = "Risk of developing a disease") +
  theme_bw() 



# #posterior
# # better inverse?
# Sigma_beta = solve(t(X)%*%solve(Sigma_n)%*%X + solve(Sigma_p));
# cat("The Bayesian estimate of the standard errors are given by", sqrt(diag(Sigma_beta)))

# mu_1 = (Sigma_beta)%*%t(X)%*%solve(Sigma_n)%*%y + solve(Sigma_p)%*%c(-100, -100, -100); # try another b_mlc(-100, -100, -100)
# cat("\n The mean of the posterior is given by", mu_1)

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


#YOUR CODE HERE

# z score qnorm(0.975)
lower_bound = 0.01397199 - qnorm(0.975)*sqrt(0.01052449)
upper_bound = 0.01397199 + qnorm(0.975)*sqrt(0.01052449)
cat("95% credible interval is ",lower_bound, "to", upper_bound)

lmod_total = lm(p ~ hours, data = df)
X = model.matrix(lmod_total)
x_star = data.frame(t(colMeans(X)));  x_star
names(x_star) = c("intercept", "hours"); 

# Sigma_beta

#YOUR CODE HERE
fail()

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

#YOUR CODE HERE
lmod1_coef = coef(summary(lmod1))["hours", "Estimate"]; lmod1_coef
lmod2_coef = coef(summary(lmod2))["hours", "Estimate"]; lmod2_coef


