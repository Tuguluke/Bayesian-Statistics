#YOUR CODE HERE
fail()

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

# b_ml = solve(t(X)%*%X)%*%t(X)%*%y; 
# rss = sum((y - X%*%b_ml)^2)
# sig2hat = rss/(n-(p+1)); sqrt(sig2hat)

# lmod = lm(sales ~ youtube, data = df)   # youtube - 1 in class note, here we are not trying to estimate the intercept
# summary(lmod)

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

#YOUR CODE HERE
library(LearnBayes)
log_posterior(c(100,100,100),X,y)

fit = laplace(log_posterior, c(100,100,100), X, y)
fit$mode

admission = read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
head(admission)

#data storage
y = admission$gre; x = admission$gpa; n = length(x); n1 = 1:n;

#prior
mu0 = c(0,0); sigma_p = matrix(c(100,0,0,100), ncol = 2); 

#standard deviation assumption
sig = 107


#frequentist
lm_gre = lm(y ~ x)
summary(lm_gre)


#YOUR CODE HERE
log_posterior = function(beta,x,y){
    X = as.matrix(cbind(1,x));
    log_likelihood = -1/2*sum((y-X%*%beta)^2)
    log_prior = -1/(2*100)*sum(beta^2)
    log_post = log_likelihood + log_prior
    return(log_post)
}

log_posterior(c(200,100),x,y)

#YOUR CODE HERE
fit = laplace(log_posterior,c(200,100),x,y)
fit$mode
# summary(lm_gre)

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

#YOUR CODE HERE
plot(1:n, running_MAP[, 1], type = 'l', col = 'blue', 
     xlab = 'Index', ylab = 'First column')

# Plot for the second column of running_MAP (beta_1 estimates)
plot(1:n, running_MAP[, 2], type = 'l', col = 'red', 
     xlab = 'Index', ylab = 'Second column')

dim(running_MAP);
dim(admission)

#YOUR CODE HERE
set.seed(123)
sub_indices = sample(nrow(admission), 200)
sub_data = admission[sub_indices, ]
dim(sub_data)

y = sub_data$gre; x = sub_data$gpa; # n = length(x); n1 = 1:n;

fit = laplace(log_posterior,c(200,100),x,y)
fit$mode


