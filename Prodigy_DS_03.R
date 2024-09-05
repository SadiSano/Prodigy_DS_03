install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")
install.packages("e1071") 
install.packages("readr")
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(readr)
library(readr)
data <- read_delim("bank.csv", delim = ";", 
                   escape_double = FALSE, trim_ws = TRUE)
View(data)

# Convert categorical variables to factors
data$y <- as.factor(data$y)
data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$month <- as.factor(data$month)
data$poutcome <- as.factor(data$poutcome)

# Split the dataset into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$y, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Build the decision tree model
model <- rpart(y ~ ., data = trainData, method = "class",
               control = rpart.control(cp = 0.01))  # Adjust cp for pruning

# Plot the decision tree
rpart.plot(model, type = 3, extra = 106, fallen.leaves = TRUE,
           main = "Decision Tree for Bank Marketing",cex = 0.5)

predictions <- predict(model, testData, type = "class") # Make predictions on the test data
confusionMatrix(predictions, testData$y) # Confusion Matrix and Accuracy

# Calculate and print additional metrics: Precision, Recall, F1-score
precision <- posPredValue(predictions, testData$y, positive = "yes")
recall <- sensitivity(predictions, testData$y, positive = "yes")
f1 <- (2 * precision * recall) / (precision + recall)

cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1-Score: ", f1, "\n")


# Cross-validation for hyperparameter tuning
control <- trainControl(method = "cv", number = 10)
tunedModel <- train(y ~ ., data = trainData, method = "rpart", 
                    trControl = control, tuneLength = 10)

# Print the best model
print(tunedModel)

# Predict and evaluate with the tuned model
tunedPredictions <- predict(tunedModel, testData)
confusionMatrix(tunedPredictions, testData$y)

# The decision tree model for predicting customer subscriptions in the Bank Marketing dataset achieved an accuracy of 89.09%. However, the model is biased towards predicting "no" due to class imbalance:
# Precision for "yes": 0.57 (57% of "yes" predictions were correct).
# Recall for "yes": 0.31 (31% of actual "yes" cases were correctly identified).
# F1-Score: 0.40 (balance between precision and recall).
# 
# Key Insights:
# The model struggles with predicting the "yes" class due to imbalance.
# Hyperparameter tuning slightly improved performance but didn't resolve the bias.
