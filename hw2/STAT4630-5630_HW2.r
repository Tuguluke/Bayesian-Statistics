#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
prior_variance  = 100


x = c(145, 142, 158, 143, 185, 164, 141, 148, 151, 150)

#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer

#This cell simulates data with known parameters. We'll imagine that we don't know the true values 
# of the parameters that generate the data. We'll try to infer them from Bayesian inference.

set.seed(1181)
n = 10; mu_true = 12000; sig_true = 200; #sample size and true parameters for the data distribution
x = round(rnorm(n,mu_true,sig_true),1) #simulated data, centered at 12,000


mean(x)
var(x)

x

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

#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
lower_bound = posterior_mean - 1.96 * sqrt(posterior_variance)
upper_bound = posterior_mean + 1.96 * sqrt(posterior_variance)
lower_bound
upper_bound

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

#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
lower_bound = posterior_mean - 1.96 * sqrt(posterior_variance)
upper_bound = posterior_mean + 1.96 * sqrt(posterior_variance)
lower_bound
upper_bound

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

#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
# posterior no longer normal 
# lower_bound
# upper_bound

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

#YOUR CODE HERE
# unit 1 ppd_sim = table(pred_sim) / m
relative_freq = table(prediction_value) / mc_sim
relative_freq

#YOUR CODE HERE
barplot(relative_freq, names.arg = names(relative_freq),
        xlab = "number of failures", ylab = "relative frequency")

#YOUR CODE HERE
prob_0_le_X_le_35 <- mean(prediction_value <= 35)
prob_0_le_X_le_35

#YOUR CODE HERE
lambda = 10
n = 0:30

prior = dpois(n, lambda)

plot(n, prior, col = "blue", type = 'l', # line
     xlab = 'x', ylab = "prior")

#YOUR CODE HERE
theta = .2
y = 5
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$Y \, | \, n \sim Binomial(n, \theta)$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
posterior

#YOUR CODE HERE
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(prior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)

#YOUR CODE HERE
theta = .2
y = 10
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$Y \, | \, n \sim Binomial(n, \theta)$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)

#YOUR CODE HERE
theta = .2
y = 15
lambda = 10
n = 0:100 #TODO: why likelihood is far off? 

prior = dpois(n, lambda)

#$Y \, | \, n \sim Binomial(n, \theta)$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)

#YOUR CODE HERE
theta = .8
y = 5
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$Y \, | \, n \sim Binomial(n, \theta)$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)

#YOUR CODE HERE
theta = .8
y = 10
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$Y \, | \, n \sim Binomial(n, \theta)$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)

#YOUR CODE HERE
theta = .8
y = 15
lambda = 10
n = 0:30 #TODO n

prior = dpois(n, lambda)

#$Y \, | \, n \sim Binomial(n, \theta)$,
likelihood = dbinom(y, size = n, prob = theta)

posterior  = likelihood*prior / sum(likelihood*prior) # normalizing, otherwise no plot
plot(n, posterior, col = "red", type = 'l', # line
     xlab = 'x', ylab = "posterior",
 ylim = c(0, max(c(posterior, likelihood)))
    )
lines(n, prior, col = "green")
lines(n, likelihood, col = "blue")

legend('topright', legend = c('Posterior', 'Prior', 'Likelihood'), col = c('red', 'green', 'blue'), lty = 1)


