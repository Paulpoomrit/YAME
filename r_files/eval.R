library(tidyverse)
data <- read_csv('../results/eval.csv')

data |>
  ggplot(aes(x=ablated_feature, y = f1, color=ablated_feature)) + geom_point() + ylim(0.9,1)