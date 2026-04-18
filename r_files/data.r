library(tidyverse)
library(tidyr)
data <- read_tsv('../data/now_data_subset/now_data_subset_w_features.tsv')

# data |>
#   group_by(label) |>
#   summarize(sum_emdash_ratio = sum(emdash)) |>
#   ggplot(aes(x=label, y = sum_emdash_ratio)) + geom_col() +ylim(0,1)
# 
# data |>
#   group_by(label) |>
#   summarize(sum_emoji_ratio = sum(emoji)) |>
#   ggplot(aes(x=label, y = sum_emoji_ratio)) + geom_col() +ylim(0,1)

sum_data <- data |>
  group_by(label) |>
  summarise(across(everything(), mean, na.rm=TRUE)) 

sum_data |>
  gather(key="variable", value="value", -label) |>
  ggplot(aes(x=var, y=value)) +
    geom_point()+
    facet_wrap(~var, scales="free")
  
  
  


    

  
  
  
  
  
  
  