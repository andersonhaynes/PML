# Practical Machine Learning Project #

## Background ##

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Goal ##

The goal of the project is to predict the manner in which they did the exercise. 

### Data Processing and Manipulation


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
training_set<- read.csv("pml-training.csv",sep=",",na.strings = c("NA",""),header=TRUE)
testing_set <- read.csv("pml-testing.csv",sep=",",na.strings = c("NA",""),header=TRUE)

inTrain <- createDataPartition(training_set$classe, p=0.70, list=FALSE)
training <- training_set[inTrain,]
validation <- training_set[-inTrain,]

training<-training[,colSums(is.na(training)) == 0]
classe<-training$classe
nums <- sapply(training, is.numeric)
training<-cbind(classe,training[,nums])
training$X<-training$num_window<-NULL

validation<-validation[,colSums(is.na(validation)) == 0]
vclasse<-validation$classe
vnums <- sapply(validation, is.numeric)
validation<-cbind(vclasse,validation[,vnums])
colnames(validation)[1]<-"classe"
validation$X<-validation$num_window<-NULL

testing<-testing_set[,colSums(is.na(testing_set)) == 0]
tnums <- sapply(testing, is.numeric)
testing<-testing[,tnums]
testing$X<-testing$num_window<-NULL
```

### Data Modelling


```r
library(randomForest)
fit <- train(training$classe~.,data=training, method="rf")
save(fit,file="fit.RData")
```

```r
load(file = "./fit.RData")
fit$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9925 0.9905   0.001760 0.002228
## 2   28   0.9959 0.9948   0.001203 0.001521
## 3   54   0.9899 0.9873   0.003732 0.004714
```

### Error estimation 

The model used on the data set produced an out of error rate < 1% with an accuracy level of > 99%

```r
traincontrol <- trainControl(method = "cv", number = 5)
```

```r
fit_crossvalidation <- train(validation$classe~.,data=validation, method="rf",trControl=traincontrol)
save(fit_crossvalidation,file="fit_crossvalidation.RData")
```

```r
load(file="./fit_crossvalidation.RData")
fit_crossvalidation$resample
```

```
##   Accuracy  Kappa Resample
## 1   0.9907 0.9882    Fold1
## 2   0.9915 0.9893    Fold3
## 3   0.9856 0.9817    Fold2
## 4   0.9890 0.9860    Fold5
## 5   0.9881 0.9850    Fold4
```

```r
fit_crossvalidation$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9818 0.9770   0.002133 0.002703
## 2   28   0.9890 0.9860   0.002327 0.002945
## 3   54   0.9864 0.9828   0.005699 0.007217
```

```r
confusionMatrix(predict(fit_crossvalidation, newdata=validation), validation$classe)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    2    0    0    0
##          B    3 1130    2    0    0
##          C    0    7 1022    6    0
##          D    0    0    2  957    2
##          E    0    0    0    1 1080
## 
## Overall Statistics
##                                         
##                Accuracy : 0.996         
##                  95% CI : (0.994, 0.997)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.995         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.992    0.996    0.993    0.998
## Specificity             1.000    0.999    0.997    0.999    1.000
## Pos Pred Value          0.999    0.996    0.987    0.996    0.999
## Neg Pred Value          0.999    0.998    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.174    0.163    0.184
## Detection Prevalence    0.284    0.193    0.176    0.163    0.184
## Balanced Accuracy       0.999    0.996    0.997    0.996    0.999
```


```r
fit_crossvalidation$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 0.66%
## Confusion matrix:
##      A    B    C   D    E class.error
## A 1670    4    0   0    0    0.002389
## B    7 1128    4   0    0    0.009658
## C    0    5 1016   5    0    0.009747
## D    0    0    5 955    4    0.009336
## E    0    0    1   4 1077    0.004621
```

### Test Cases

In order to establish the predicton capabilities of the model; the following test was carried out on 20 cases. 

```r
test_prediction<-predict(fit, newdata=testing)
test_prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(test_prediction)
```
