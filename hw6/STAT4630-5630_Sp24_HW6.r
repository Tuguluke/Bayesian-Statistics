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

df =  admission[, c("gpa", "gre")]
admission$gpa

library(LearnBayes)

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

#YOUR CODE HERE
# I can not get fit$var to work

#YOUR CODE HERE


#YOUR CODE HERE

#YOUR CODE HERE
fail() # No Answer - remove if you provide an answer

auto_full = read.csv(paste0("https://raw.githubusercontent.com/bzaharatos/",
                            "-Statistical-Modeling-for-Data-Science-Applications/",
                            "master/Modern%20Regression%20Analysis%20/",
                            "Datasets/auto-mpg.csv"), sep = ",")
auto_full = na.omit(auto_full) #removing the rows that have an NA for horsepower
auto_full$cylinders = as.factor(auto_full$cylinders)
#we'll work with standardized continuous predictors
auto_predictors = scale(auto_full[,c(4,5)])

#df contains the unstandardized response and standardized continuous predictors
df = data.frame(mpg = auto_full$mpg, cylinders = auto_full$cylinders, auto_predictors)
summary(df)

lmod = lm(mpg ~ horsepower + weight, data = df)
summary(lmod)

X = as.matrix(cbind(1, df[,c(3,4)])) #design matrix
y = df[,1];  #response

#data in the format that we'll need in MH
data = cbind(y,X)
head(data)



# data[, 1]

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


#YOUR CODE HERE
T = 20000
beta_matrix = mhs(beta0 = rep(1, 3), proposal_sd = rep(0.25, 3), T = T, data = data)

# HINT: use colMeans(), but only include rows after a burn-in period of T/4 rows.
burnin =  T/4
beta_expected = colMeans(beta_matrix[(burnin + 1):T, ])

beta_expected


#YOUR CODE HERE

colnames(beta_matrix) # = c("beta_0", "beta_1", "beta_2")
head(beta_matrix)

# burnin = 500
# dimnames(chain1)[[2]]=c("logit eta","log K")
# #xyplot(mcmc(chain1[-c(1:burnin),]),col="black") #in lattice
# plot(chain1[,1], type = "l")
# lines(chain2[,1], type = "l", col = "red")
# lines(chain3[,1], type = "l", col = "blue")

#YOUR CODE HERE
# Plot the chains after the burn-in period
burnin  =  T / 4 # 0

plot(beta_matrix[(burnin + 1):nrow(beta_matrix),1], type = "l", 
    ylim = range(beta_matrix[(burnin + 1):nrow(beta_matrix), ]),  # Set the y-axis range to fit all data
     col = "black")
lines(beta_matrix[(burnin + 1):nrow(beta_matrix),2], col = "red")
lines(beta_matrix[(burnin + 1):nrow(beta_matrix),3], col = "blue")


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

xstar = data.frame(0,0) #data frame with prediction values (1 at the beginning for the intercept in the predict function)
colnames(xstar) = colnames(data)[3:4]
predict(lmod, xstar, interval = "prediction") #frequentist prediction, for reference

xstar = as.numeric(xstar) #relavant values, for ppd algorithm

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


set.seed(5630)
n = 7; mu_true = 45; sig_true = 35; x = rnorm(n,mu_true,sig_true)

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


#YOUR CODE HERE
plot(mu_chain, type = 'l',  
     xlab = "T", ylab = "Mu")

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


