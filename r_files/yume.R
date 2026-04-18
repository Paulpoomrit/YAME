library(tidyverse)
data <- read_csv('../results/yame_ablation.csv')

data$true_y <- as.factor(data$true_y)
data$pred_y <- as.factor(data$pred_y)
data$model <- as.factor(data$model)
str(data)
head(data)

logistic <- glm(correct ~ model , data = data, family="binomial")
summary(logistic)

library(gtsummary)
tbl_regression(logistic)