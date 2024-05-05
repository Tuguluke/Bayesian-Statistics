#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
f_Y = function(y) {
  0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((y - 0) / 2)^2) + 0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((y - 5) / 2)^2)
}
# https://www.educative.io/answers/how-to-use-curve-in-rd
curve(f_Y, -20, 20, xlab = "y", ylab = "f_Y(y)", col = "red")  

#YOUR CODE HERE
# fail() # No Answer - remove if you provide an answer
theta_0 =  0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((1 - 0) / 2)^2)
theta_5 =  0.5 * (1 / (2 * sqrt(2 * pi))) * exp(-0.5 * ((1 - 5) / 2)^2)

result = theta_0/ (theta_0 + theta_5)
print(result)

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

#note the updated/corrected simulation
library(MASS)
set.seed(7309)

mu = c(0,0)
rho = 0
Sig = matrix(c(1,rho,rho,1), ncol = 2)
    
n = 15
X = mvrnorm(n, mu, Sig)
x = X[,1]; y = X[,2]


#YOUR CODE HERE
plot(x, y, main = "B.1 (a)", xlab = "X", ylab = "Y", pch = 16, col = "blue")

# fail() # No Answer - remove if you provide an answer

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

# 1. Create a grid of $\rho$ values between $-0.99$ and $0.99$. Store these in `r`. 
r = seq(-0.99, 0.99, by = 0.01)

# 2. Compute the prior density (uniform between $-1$ and $1$, as above) at each value of $\rho$.
prior = rep(U_constant, length(r))  

# 3. Program the likelihood function for the observations given in part (a) of this problem. 
#TODO: no product? why
likelihood = function(rho, x, y) {
  prod(1 / (2 * pi * sqrt(1 - rho^2)) * exp(-((x^2 - 2 * rho * x * y + y^2) / (2 * (1 - rho^2)))))
}

# 4. Compute the posterior distribution at each value of $\rho$.
posterior =  prior * sapply(r, likelihood, x = x, y = y)

# 5. Plot the posterior as a function of $\rho$.
plot(r, posterior, col = "blue",
     xlab = 'rho', ylab = "posterior",
)
# adding the likelihood function
lines(r, sapply(r, likelihood, x = x, y = y), col = "red")
legend("topright", legend = c("Posterior", "Likelihood"), col = c("blue", "red"), lty = 1:1)


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

#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer


