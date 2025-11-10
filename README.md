!sudo apt-get install -y r-base
!pip install rpy2

%load_ext rpy2.ipython


%%R
# ---------- Hypertension Prediction in R ----------

# Install required R packages (only first time)
packages <- c("caret", "randomForest", "pROC", "ggplot2", "dplyr")
new.pkgs <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new.pkgs)) install.packages(new.pkgs, repos='https://cloud.r-project.org')

library(caret)
library(randomForest)
library(pROC)
library(ggplot2)
library(dplyr)

set.seed(123)

# Step 1: Create synthetic dataset (10,000 records)
n <- 10000
data <- data.frame(
  Age = round(rnorm(n, mean = 45, sd = 12)),
  Gender = factor(sample(c("Male", "Female"), n, replace = TRUE)),
  BMI = round(rnorm(n, mean = 26, sd = 5), 1),
  Smoking = factor(sample(c("Yes", "No"), n, replace = TRUE, prob = c(0.3, 0.7))),
  Alcohol = factor(sample(c("Yes", "No"), n, replace = TRUE, prob = c(0.4, 0.6))),
  PhysicalActivity = factor(sample(c("Low", "Medium", "High"), n, replace = TRUE, prob = c(0.3, 0.4, 0.3))),
  Cholesterol = round(rnorm(n, mean = 200, sd = 30)),
  Glucose = round(rnorm(n, mean = 100, sd = 20))
)

# Step 2: Target variable (Hypertension)
data$Hypertension <- with(data, ifelse(
  (Age > 50 & BMI > 28) | Smoking == "Yes" | Cholesterol > 220 |
    Glucose > 120 | PhysicalActivity == "Low", 1, 0
))
data$Hypertension <- factor(data$Hypertension, labels = c("No", "Yes"))

# Step 3: Train/test split (80/20)
trainIndex <- createDataPartition(data$Hypertension, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test  <- data[-trainIndex, ]

# Step 4: Logistic Regression
log_model <- glm(Hypertension ~ Age + Gender + BMI + Smoking + Alcohol +
                   PhysicalActivity + Cholesterol + Glucose,
                 data = train, family = "binomial")

log_pred <- predict(log_model, newdata = test, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, "Yes", "No")

# Step 5: Random Forest
rf_model <- randomForest(Hypertension ~ Age + Gender + BMI + Smoking + Alcohol +
                           PhysicalActivity + Cholesterol + Glucose,
                         data = train, ntree = 200, importance = TRUE)
rf_pred <- predict(rf_model, newdata = test)

# Step 6: Evaluate Models
cat("\n--- Logistic Regression ---\n")
print(confusionMatrix(as.factor(log_pred_class), test$Hypertension))

cat("\n--- Random Forest ---\n")
print(confusionMatrix(rf_pred, test$Hypertension))

# Step 7: ROC + AUC
rf_prob <- predict(rf_model, newdata = test, type = "prob")[,2]
roc_obj <- roc(test$Hypertension, rf_prob)
auc_val <- auc(roc_obj)
cat("\nAUC for Random Forest:", auc_val, "\n")

plot(roc_obj, col = "blue", main = paste("ROC Curve (AUC =", round(auc_val, 3), ")"))

# Step 8: Feature Importance
importance_df <- data.frame(Feature = rownames(importance(rf_model)),
                            Importance = importance(rf_model)[,1])
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)",
       x = "Features", y = "Importance") +
  theme_minimal()
