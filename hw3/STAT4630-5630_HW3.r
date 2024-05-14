 # meaning of lambda: x are gamma.   lambda is rate parameter. 

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


#YOUR CODE HERE
#  demoniators.  dont forget about that. 
denom  =  integrate(function(lambda) lambda^(n-1) * exp(-lambda * sum_x_i), lower = 0, upper = Inf)

normal_posterior =  function(lambda) {
  lambda^(n-1) * exp(-lambda * sum_x_i)  / denom$value # TODO: 
} 
                    
integrate(normal_posterior, lower = 0, upper = Inf)
# it is proper since the integration is 1. 

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



#YOUR CODE HERE
plot(theta_grid, normalized_posterior_values, type='l', 
     col='red', xlab='Theta', ylab='Density')

#YOUR CODE HERE
theta_grid = seq(0, 100, length.out = 1000)

# normalized_posterior = unnormalized_posterior / sum(unnormalized_posterior)

total_integral = integrate(unnormalized_posterior, lower = 0, upper = 100)$value

normalized_posterior = function(theta) {
  unnormalized_posterior(theta) / total_integral
}

normalized_posterior_probs = sapply(theta_grid, normalized_posterior)

theta_samples = sample(theta_grid, size=1000, replace=TRUE, prob=normalized_posterior_probs)


# plot(theta_grid, normalized_posterior_probs, col='red', lwd=2)

#YOUR CODE HERE
hist(theta_samples, breaks=50, probability=TRUE, col='blue',
     xlab='Theta', ylab='Relative Frequency')

# TODO: Density aes(x = theta_samples, y = ..density..)
theta_density = density(theta_samples) 

lines(theta_grid, 
      normalized_posterior_probs*max(theta_density$y)/max(normalized_posterior_probs)
      , col='red')
legend("topright", legend = c("relative frequency", "posterior density"), col = c("blue", "red"), lty = 1:1)

#YOUR CODE HERE
# random seeds?
set.seed(42)

predictive_samples = rcauchy(n = 1000, location = theta_samples, scale = 1)


hist(predictive_samples, breaks = 50, probability=TRUE, col='blue',
     xlab='Theta', ylab='Relative Frequency')


