#Homework: Diabetes
# contact: martyna.plomecka@uzh.ch

#x a matrix with 10 columns
#y a numeric vector (442 rows)
#x2 a matrix with 64 columns

install.packages("elasticnet")
library(elasticnet)
data(diabetes)
colnames(diabetes)
?diabetes


#TODO
#1. Fit LASSO and Elastic Net to the data with optimal tuning parameter chosen by cross validation.
#2. Compare solution paths for the two methods