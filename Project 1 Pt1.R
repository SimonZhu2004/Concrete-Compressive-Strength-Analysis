gradient_descent <- function(x, y,learning_rate, max_iterations, tolerance) {
  
  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)
  
  x <- cbind(1, x)
  parameters <- rep(0, ncol(x))
  
  mse_history <- numeric(max_iterations)
  previous_mse <- Inf
  
  for (i in 1:max_iterations) {
    
    predictions <- as.vector(x %*% parameters)
    errors <- predictions - y
    gradient <- (2 / n) * as.vector(t(x) %*% errors)
    parameters <- parameters - learning_rate * gradient
    
    predictions <- as.vector(x %*% parameters)
    current_mse <- mean((y - predictions)^2)
    mse_history[i] <- current_mse
    
    if (!is.finite(current_mse)) break
    if (abs(previous_mse - current_mse) < tolerance) break
    
    previous_mse <- current_mse
  }
  
  observed_variance <- mean((y - mean(y))^2)
  r_squared <- 1 - (current_mse / observed_variance)
  
  return(list(
    parameters = parameters,
    mse = current_mse,
    mse_history = mse_history[1:i],
    r_squared = r_squared,
    iterations = i
  ))
}
#============================================================================
#Data Prep
#============================================================================

setwd(".")
concrete <- read.csv("Concrete_Data.csv")

x <- as.matrix(concrete[, 1:8])
y <- as.numeric(concrete[, 9])


test_rows <- 501:630
x_train <- x[-test_rows, ]
y_train <- y[-test_rows]
x_test <- x[test_rows, ]
y_test <- y[test_rows]


# ============================================================
# Scaling & Convergence Speed
# ============================================================
means <- colMeans(x_train)
sds <- apply(x_train, 2, sd)

x_scaled_train <- scale(x_train,center = means,scale = sds)
x_scaled_test <- scale(x_test,center = means,scale = sds)

rgd <- gradient_descent(x_train,y_train,learning_rate = 0.0000001,max_iterations = 100000, tolerance = 1e-6)
sgd <- gradient_descent(x_scaled_train, y_train, learning_rate = 0.01,max_iterations = 100000,tolerance = 1e-6)

rgd_pred_test <- as.vector(cbind(1, x_test) %*% rgd$parameters)
rgd_mse_test <- mean((y_test - rgd_pred_test)^2)
rgd_r2_test <- (1 -rgd_mse_test /(mean((y_test - mean(y_test))^2)))


# Test performance: scaled model
sgd_pred_test <- as.vector(cbind(1, x_scaled_test) %*% sgd$parameters)
sgd_mse_test <- mean((y_test - sgd_pred_test)^2)
sgd_r2_test <- 1 - (sgd_mse_test /(mean((y_test - mean(y_test))^2)))


# Compare results
gd_results <- data.frame(
  model = c("Untransformed", "Standardized"),
  mse_train = c(rgd$mse, sgd$mse),
  mse_test = c(rgd_mse_test, sgd_mse_test),
  r2_train = c(rgd$r_squared, sgd$r_squared),
  r2_test = c(rgd_r2_test, sgd_r2_test),
  iterations = c(rgd$iterations, sgd$iterations)
)

print(gd_results)


# Loss over iterations
plot(rgd$mse_history,type = "l",xlab = "Iteration",ylab = "MSE",main = "Loss for each Iterations", 
     xlim = c(0, 4000), ylim = range(c(rgd$mse_history, sgd$mse_history)), lwd = 3, col = "black")
lines(sgd$mse_history,col = "blue", lwd = 3)

legend("topright",legend = c("original", "standardized"),col = c("black", "blue"),lty=1, lwd = 3)



# ============================================================
# Scaling & The Optimal Solution
# ============================================================
rlm <- lm(y_train ~ .,data = data.frame(x_train))
slm <- lm(y_train ~ .,data = data.frame(x_scaled_train))
summary(rlm)
summary(slm)

rlm_pred_train <- predict(rlm)
slm_pred_train <- predict(slm)

rlm_pred_test <- predict(rlm,newdata = data.frame(x_test))
slm_pred_test <- predict(slm,newdata = data.frame(x_scaled_test))

rlm_mse_train <- mean((y_train - rlm_pred_train)^2)
rlm_mse_test <- mean((y_test - rlm_pred_test)^2)
slm_mse_train <- mean((y_train - slm_pred_train)^2)
slm_mse_test <- mean((y_test - slm_pred_test)^2)


rlm_r2_train <- summary(rlm)$r.squared
rlm_r2_test <- 1 -(rlm_mse_test /(mean((y_test - mean(y_test))^2)))
slm_r2_train <- summary(slm)$r.squared
slm_r2_test <- 1 - (slm_mse_test / (mean((y_test - mean(y_test))^2)))


lm_results <- data.frame(
  model = c("Untransformed", "Standardized"),
  mse_train = c(rlm_mse_train, slm_mse_train),
  mse_test = c(rlm_mse_test, slm_mse_test),
  r2_train = c(rlm_r2_train, slm_r2_train),
  r2_test = c(rlm_r2_test, slm_r2_test)
)

print(lm_results)

coef(rlm)
coef(slm)

# ==========================================================
# Nonlinear Transformations
# ==========================================================
x_logged_train <- log(x_train+1)
x_logged_test <- log(x_test+1)

lgd <- gradient_descent(x_logged_train, y_train, learning_rate = 0.005, max_iterations = 100000, tolerance = 1e-6)

logged_pred_test <- as.vector(cbind(1, x_logged_test) %*% lgd$parameters)
logged_mse_test <- mean((y_test-logged_pred_test)^2)
logged_r2_test <- 1-(logged_mse_test / mean((y_test-mean(y_test))^2))

plot(lgd$mse_history, type = "l", xlab = "Iteration", ylab = "MSE", main = "Loss of Iterations", lwd = 2.5, 
    # xlim = c(1, max(lgd$iterations, rgd$iterations, sgd$iterations)),
    xlim = c(1, 2000),
     ylim = range(c(lgd$mse_history,rgd$mse_history,sgd$mse_history)))
lines(rgd$mse_history, col = "blue", lwd = 2.5)
lines(sgd$mse_history, col = "green", lwd = 2.5)
legend("topright", legend = c("Logged", "Original", "Standardized"), col = c("black", "blue", "green"), lty = 1, lwd = 2.5)

llm <- lm(y_train ~ ., data = data.frame(x_logged_train))
summary(llm)
logged_lm_pred_train <- predict(llm)
logged_lm_pred_test <- predict(llm, newdata = data.frame(x_logged_test))

logged_lm_mse_train <- mean((y_train-logged_lm_pred_train)^2)
logged_lm_mse_test <- mean((y_test-logged_lm_pred_test)^2)

logged_lm_r2_train <- summary(llm)$r.squared
logged_lm_r2_test <- 1- (logged_lm_mse_test / mean((y_test - mean(y_test))^2))

coef(llm)

summart_table <- data.frame(model = c("Unscaled", "Scaled", "Logged"), mse_train = c(rlm_mse_train, slm_mse_train, logged_lm_mse_train),
                            mse_test = c(rlm_mse_test, slm_mse_test, logged_lm_mse_test), r2_train = c(rlm_r2_train, slm_r2_train, logged_lm_r2_train),
                            r2_test = c(rlm_r2_test, slm_r2_test, logged_lm_r2_test))
print(summart_table)

#===================================================================
#Univariate and Multivariate
#===================================================================


cement_gd <- gradient_descent(x_train[, 1],y_train,learning_rate = 0.00001,max_iterations = 100000,tolerance = 1e-6)

slag_gd <- gradient_descent(x_train[, 2],y_train,learning_rate = 0.00005,max_iterations = 100000,tolerance = 1e-6)

flyash_gd <- gradient_descent(x_train[, 3],y_train,learning_rate = 0.0001,max_iterations = 100000,tolerance = 1e-6)

water_gd <- gradient_descent(x_train[, 4],y_train,learning_rate = 0.000009,max_iterations = 100000,tolerance = 1e-6)

superplasticizer_gd <- gradient_descent(x_train[, 5],y_train,learning_rate = 0.01,max_iterations = 100000,tolerance = 1e-6)

coarse_gd <- gradient_descent(x_train[, 6],y_train,learning_rate = 0.00000000001,max_iterations = 100000,tolerance = 1e-6)

fine_gd <- gradient_descent(x_train[, 7],y_train,learning_rate = 0.0000001,max_iterations = 100000,tolerance = 1e-6)

age_gd <- gradient_descent(x_train[, 8],y_train,learning_rate = 0.0001,max_iterations = 100000,tolerance = 1e-6)

#=======================================================================================================
cement_lm <- lm(y_train ~ x_train[, 1])

slag_lm <- lm(y_train ~ x_train[, 2])

flyash_lm <- lm(y_train ~ x_train[, 3])

water_lm <- lm(y_train ~ x_train[, 4])

superplasticizer_lm <- lm(y_train ~ x_train[, 5])

coarse_lm <- lm(y_train ~ x_train[, 6])

fine_lm <- lm(y_train ~ x_train[, 7])

age_lm <- lm(y_train ~ x_train[, 8])

cement_coef <- coef(cement_lm)[2]

slag_coef <- coef(slag_lm)[2]

flyash_coef <- coef(flyash_lm)[2]

water_coef <- coef(water_lm)[2]

superplasticizer_coef <- coef(superplasticizer_lm)[2]

coarse_coef <- coef(coarse_lm)[2]

fine_coef <- coef(fine_lm)[2]

age_coef <- coef(age_lm)[2]

coefficient_table <- data.frame(Feature = c("Cement","Blast Furnace Slag","Fly Ash","Water","Superplasticizer","Coarse Aggregate","Fine Aggregate","Age"),
Univariate = c(cement_coef,slag_coef,flyash_coef,water_coef,superplasticizer_coef,coarse_coef,fine_coef,age_coef),
Multivariate = coef(rlm)[-1])

print(coefficient_table)
#============================================================================
p_unscaled <- summary(rlm)$coefficients[-1, 4]
p_scaled <- summary(slm)$coefficients[-1, 4]
p_logged <- summary(llm)$coefficients[-1, 4]

# Create consolidated table
pvalue_table <- data.frame(Feature = c("Cement","Blast Furnace Slag","Fly Ash","Water","Superplasticizer","Coarse Aggregate","Fine Aggregate","Age"),
Unscaled = p_unscaled, Scaled = p_scaled, Logged = p_logged)

print(pvalue_table)
