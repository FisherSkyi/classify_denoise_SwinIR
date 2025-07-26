data <- read.csv("/Users/sky/Research/David/onsite/CNN_tensorflow/class_image_counts.csv")

# If not installed yet
# install.packages("ggplot2")

# Load library
library(ggplot2)

# Basic bar plot
ggplot(data, aes(x = Class, y = ImageCount)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Number of Images per Class",
       x = "Class",
       y = "Image Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
