#########################################################
## Stat 202A - Final project: 100 points
## Author: 
## Date : 
## Description: This script implements a 2 layer neural
## network, a support vector machine, an adaboost 
## classifier, logistic regression, and a Bayesian variable 
## selection scheme based on the Gibbs sampler
#########################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
##
## Make sure that the digits dataset is called "digits.csv"
## and is in your current working directory. If you change the 
## name or change the directory of where this file is, I will 
## get an error when grading your exam.
#############################################################

## We will use the function fread from the package "data.table" 
## to read in the digits data. Do not change these lines.
library(data.table)  

## Create a function to read in and prepare the data for
## subsequent tasks.
prepare_data <- function(valid_digits = c(3, 5)){
  
  ## valid_digits is a vector containing the digits
  ## we wish to classify.
  
  ## Do not change anything inside of this function
  
  if(length(valid_digits) != 2){
    stop("Error: you must specify exactly 2 digits for classification")
  }

  filename <- "digits.csv"
  digits   <- fread(filename) 
  digits   <- as.matrix(digits)
  
  if(filename != "digits.csv"){
    stop("I didn't bother reading the instructions!")
  }
  
  ## Keep only the digits "3" and "5" for binary classification
  valid <- (digits[,65] == valid_digits[1]) | (digits[,65] == valid_digits[2])
  X     <- digits[valid, 1:64]
  Y     <- digits[valid, 65]
  
  ## Rescale X values to lie between 0 and 1
  X <- t(sapply(1:nrow(X), function(x) X[x,] / max(X[x,])))
  
  ## Relabel Y values as 0 and 1
  Y[Y == valid_digits[1]] <- 0
  Y[Y == valid_digits[2]] <- 1
  
  ## Randomly split data 75% and 25%
  training_set <- sample(1:360, 270)
  testing_set  <- setdiff(1:360, training_set)
  
  X_train <- X[training_set, ]
  Y_train <- Y[training_set]
  X_test  <- X[testing_set, ]
  Y_test  <- Y[testing_set]
  
  output <- list(X_train = X_train, Y_train = Y_train, 
                 X_test = X_test, Y_test = Y_test)
  output
  
}

## Save prepared data
cleaned_data <- prepare_data()
X_train      <- cleaned_data$X_train
Y_train      <- cleaned_data$Y_train
X_test       <- cleaned_data$X_test
Y_test       <- cleaned_data$Y_test

## Define a measure of accuracy
accuracy <- function(p, y){
  return(mean((p > 0.5) == (y == 1)))
}

############################################
## Function 1: Neural network (12 points) ##
############################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Train a neural network to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


my_NN <- function(X_train, Y_train, X_test, Y_test, num_hidden = 20, 
                     num_iterations = 1000, learning_rate = 1e-1){
  
  ## X_train: Training set of images (8 by 8 images/matrices)
  ## Y_train: Training set of labels corresponding to X_train
  ## X_test: Testing set of images (8 by 8 images/matrices)
  ## Y_test: Testing set of labels correspdonding to Y_train
  ## num_hidden: Number of hidden units in the hidden layer
  ## num_iterations: Number of iterations.
  ## learning_rate: Learning rate.
  
  ## Function should learn a neural network with 1 hidden layer.
  ## Input data consist of 8x8 dimensional digit images (matrices)
  ## and their corresponding labels. 
  
  #######################
  ## FILL IN CODE HERE ##
  #######################
 
  n <- dim(X_train)[1]
  p <- dim(X_train)[2] + 1
  ntest <- dim(X_test)[1]
  X_train1 <- cbind(rep(1, n), X_train)
  X_test1 <- cbind(rep(1, ntest), X_test)
  
  
  alpha <- matrix(rnorm(p * num_hidden), nrow = p)
  beta <- matrix(rnorm((num_hidden + 1)), nrow = num_hidden + 1)
   
  acc_train <- rep(0, num_iterations)
  acc_test <- rep(0, num_iterations)
  
  for(it in 1:num_iterations)
  {
    Z <- 1 / (1 + exp(-X_train1 %*% alpha))
    Z1 <- cbind(rep(1, n), Z)
    pr <- 1 / (1 + exp(-Z1 %*% beta))
  
    dbeta <- t(Y_train-pr)%*%Z1
    beta <- beta + learning_rate * t(dbeta)
    
    for(k in 1:num_hidden)
    {
      da <- (Y_train - pr)*beta[k+1]*Z[, k]*(1-Z[, k])
      dalpha <- matrix(rep(1, n), nrow = 1)%*%((matrix(da, n, p)*X_train1))/n
      alpha[, k] <- alpha[, k] + learning_rate * t(dalpha)
      
    }
    acc_train[it] <- accuracy(pr, Y_train)
    Ztest <- 1/(1 + exp(-X_test1 %*% alpha))
    Ztest1 <- cbind(rep(1, ntest), Ztest)
    prtest <- 1/(1 + exp(-Ztest1 %*% beta))
    acc_test[it] <- accuracy(prtest, Y_test)
  }
  ## Function should output 4 things:
  ## 1. The learned weights of the input layer, alpha
  ## 2. The learned weights of the hidden layer, beta
  ## 3. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
  ## 4. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
  model <- list(alpha = alpha, beta = beta, 
                acc_train = acc_train, acc_test = acc_test)
  model
  
}


####################################################
## Function 2: Support vector machine (12 points) ##
####################################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Train an SVM to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

my_SVM <- function(X_train, Y_train, X_test, Y_test, lambda = 0.01,
                   num_iterations = 1000, learning_rate = 0.1){
  
  ## X_train: Training set of images (8 by 8 images/matrices)
  ## Y_train: Training set of labels corresponding to X_train
  ## X_test: Testing set of images (8 by 8 images/matrices)
  ## Y_test: Testing set of labels correspdonding to Y_train
  ## lambda: Regularization parameter
  ## num_iterations: Number of iterations.
  ## learning_rate: Learning rate.
  
  ## Function should learn the parameters of an SVM.
  ## Input data consist of 8x8 dimensional digit images (matrices)
  ## and their corresponding labels. 
  
  #######################
  ## FILL IN CODE HERE ##
  #######################
  
  n <- dim(X_train)[1]
  p <- dim(X_train)[2] + 1
  X_train1 <- cbind(rep(1, n), X_train)
  
  Y_train <- 2 * Y_train - 1
  beta <- matrix(rep(0, p), nrow = p)
  
  ntest <- nrow(X_test)
  X_test1 <- cbind(rep(1, ntest), X_test)
  Y_test <- 2 * Y_test - 1
  
  acc_train <- rep(0, num_iterations)
  acc_test <- rep(0, num_iterations)
  
  for(it in 1:num_iterations)
  {
    s <- X_train1 %*% beta
    db <- s * Y_train < 1
    
    dbeta     <- matrix(rep(1,n),nrow = 1)%*%(matrix(db*Y_train,n,p)*X_train1)/n
    
    beta <- beta + learning_rate * t(dbeta)
    beta[2:p] <- beta[2:p] - lambda * beta[2:p]
  
    acc_train[it] <- mean(sign(s * Y_train))
    acc_test[it] <- mean(sign(X_test1 %*% beta * Y_test))
  }
  
  ## Function should output 3 things:
  ## 1. The learned parameters of the SVM, beta
  ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
  ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
  model <- list(beta = beta, acc_train = acc_train, acc_test = acc_test)
  model

}



######################################
## Function 3: Adaboost (12 points) ##
######################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Use Adaboost to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

myAdaboost <- function(X_train, Y_train, X_test, Y_test,
                       num_iterations = 200) {
  
  ## X_train: Training set of images (8 by 8 images/matrices)
  ## Y_train: Training set of labels corresponding to X_train
  ## X_test: Testing set of images (8 by 8 images/matrices)
  ## Y_test: Testing set of labels correspdonding to Y_train
  ## num_iterations: Number of iterations.

  ## Function should learn the parameters of an adaboost classifier.
  ## Input data consist of 8x8 dimensional digit images (matrices)
  ## and their corresponding labels. 
  
  #######################
  ## FILL IN CODE HERE ##
  #######################
  
  n <- dim(X_train)[1]
  p <- dim(X_train)[2]
  threshold <- 0.8
  
  X_train1 <- 2 * (X_train > threshold) - 1
  Y_train <- 2 * Y_train - 1
  
  X_test1 <- 2 * (X_test > threshold) - 1
  Y_test <- 2 * Y_test - 1
  
  beta <- matrix(rep(0,p), nrow = p)
  w <- matrix(rep(1/n, n), nrow = n)
  
  weak_results <- Y_train * X_train1 > 0
  acc_train <- rep(0, num_iterations)
  acc_test <- rep(0, num_iterations)
  
  for(it in 1:num_iterations)
  {
    w <- w / sum(w)
    weighted_weak_results <- w[,1] * weak_results
    weighted_accuracy <- colSums(weighted_weak_results)
    
    e <- 1 - weighted_accuracy
    j <- which.min(e)
    
    dbeta <- 0.5*log((1-e[j])/e[j])
    
    beta[j] <- beta[j] + dbeta
    
    w <- w * exp(-weak_results[,j]*dbeta)
    
    acc_train[it] <- mean(sign(X_train1 %*% beta) == Y_train)
    acc_test[it] <- mean(sign(X_test1 %*% beta) == Y_test)
  }
  
  ## Function should output 3 things:
  ## 1. The learned parameters of the adaboost classifier, beta
  ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
  ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
  output <- list(beta = beta, acc_train = acc_train, acc_test = acc_test)
  output
}



#################################################
## Function 4: Logistic Regression (12 points) ##
#################################################

myLogistic <- function(X_train, Y_train, X_test, Y_test, 
                       num_iterations = 500, learning_rate = 1e-1){
  
  ## X_train: Training set of images (8 by 8 images/matrices)
  ## Y_train: Training set of labels corresponding to X_train
  ## X_test: Testing set of images (8 by 8 images/matrices)
  ## Y_test: Testing set of labels correspdonding to Y_train
  ## num_iterations: Number of iterations.
  
  ## Function should learn the parameters of the logistic
  ## regression model.
  ## Input data consist of 8x8 dimensional digit images (matrices)
  ## and their corresponding labels. 
  
  #######################
  ## FILL IN CODE HERE ##
  #######################
  
  n <- dim(X_train)[1]
  p <- dim(X_train)[2]+1
  ntest <- dim(X_test)[1]
  
  X_train1 <- cbind(rep(1, n), X_train)
  X_test1 <- cbind(rep(1, ntest), X_test)
  
  sigma <- .1
  beta <- matrix(rnorm(p)*sigma, nrow=p)
  
  acc_train <- rep(0, num_iterations)
  acc_test <- rep(0, num_iterations)
  
  for(it in 1:num_iterations)
  {
    pr <- 1/(1 + exp(-X_train1 %*% beta))
    dbeta <- matrix(rep(1, n), nrow = 1) %*%((matrix(Y_train - pr, n, p)*X_train1))/n
    beta <- beta + learning_rate * t(dbeta)
    prtest <- 1/(1 + exp(-X_test1 %*% beta))
    acc_train[it] <- accuracy(pr, Y_train)
    acc_test[it] <- accuracy(prtest, Y_test)
  }
  ## Function should output 3 things:
  ## 1. The learned parameters of the logistic regression model, beta
  ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
  ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
  output <- list(beta = beta, acc_train = acc_train, acc_test = acc_test)  
  output

}

##################################################################
## Function 5: Variable Selection via Gibbs Sampler (12 points) ##
##################################################################

## We're not going to be using the digits data anymore.
## You can use the following function to simulate data in 
## order to test your Gibbs sampler; this function is here 
## simply to help you test out your Gibbs sampler. It is 
## already complete and will not be graded. You can modify
## it however you want!

simulate_data <- function(n = 1000, p = 100, s = 10, tau = 3, sigma = 1){
  
  ## n: Sample size
  ## p: Number of variables
  ## s: Number of active variables
  ## tau:   SD of beta prior
  ## sigma: SD of noise component of Y
  
  X <- matrix(rnorm(n * p), nrow = n)
  beta_true      <- matrix(rep(0, p), nrow = p)
  beta_true[1:s] <- rnorm(s) * tau
  Y <- X %*% beta_true + rnorm(n) * sigma
  
  output <- list(X = X, Y = Y, beta_true = beta_true)
  output
  
}


myGibbs <- function(X, Y, numIter, tau = 3, sigma = 1, s = 10){
  
  ## X: nxp matrix of data (independent variables)
  ## Y: n dimensional vector of responses
  ## numIter: Number of iterations
  ## tau:   SD of beta prior
  ## sigma: SD of noise component of Y
  ## s: Number of active variables. Since we're "cheating", 
  ## make sure to set rho <- s / p inside of your function
  
  ## The goal of this function is to classify variables (e.g. 
  ## the columns of X) as "important" and "not important", 
  ## and return the estimates of beta.
  
  #######################
  ## FILL IN CODE HERE ##
  #######################
  
  n = nrow( X )
  p = ncol(X)
  rho = s/p
  T1 = numIter
  
  beta = matrix(rep(0, p), nrow = p) 
  beta_all = matrix(rep(0, p*T1), nrow = p)
  
  R=Y
  ss = rep(0, p)
  for (j in 1:p)
    ss[j] = sum(X[, j]^2)
  
  for (t in 1:T1)
  {
    for (j in 1:p)
    {
        db = sum(R*X[, j])/ss[j]
        b = beta[j]+db
        v0 = sigma^2/ss[j]
        v1 = tau^2 + v0
        a = rho/(1-rho)
        a = a*sqrt(v0)/sqrt(v1)
        a = a*exp(min(b^2/(2*v0)-b^2/(2*v1),100))
        pr = a/(a+1)
        if (runif(1)<pr)
        {
          mu = (b*tau^2)/(tau^2+v0)
          sig = 1/((1/tau^2)+(1/v0))
          bj = rnorm(1)*sqrt(sig)+mu
        }

        else
        {
          bj = 0
        }
        db = bj - beta[j]
        beta[j] = bj
        beta_all[j, t] = beta[j]
        R = R - X[, j]*db
    }
  }
  ## Output returns our estimate of beta, called "beta",
  ## which is a p-dimensional vector. Many of the components
  ## should be 0 if you did this problem correctly.
  output <- beta
  output
}
  

####################################################
## Optional examples (comment out your examples!) ##
####################################################
# 
# n1 <- dim(X_train)[1]
# n2 <- dim(X_test)[1]
# 
# # test NN
# model_nn = my_NN(X_train, Y_train, X_test, Y_test)
# 
# # test SVM
# model_svm = my_SVM(X_train, Y_train, X_test, Y_test)
# 
# # test Adaboost
# model_adaboost = myAdaboost(X_train, Y_train, X_test, Y_test,1000)
# 
# # test Logistic
# model_logistic = myLogistic(X_train, Y_train, X_test, Y_test,1000)
# 
# x = 1:1000
# plot(x,model_nn[[3]],col="red",type="l")
# lines(x,model_nn[[3]])
# par(new=TRUE)
# plot(x,model_svm[[3]],col="blue",type="l")
# lines(x,model_svm[[3]])
# par(new=TRUE)
# plot(x,model_adaboost[[3]],col="green",type="l")
# lines(x2,model_adaboost[[3]])
