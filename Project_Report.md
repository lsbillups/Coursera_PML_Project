# Coursera_PML Project Report
lsbillups  
Friday, January 23, 2015  



## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Analysis strategy
The goal of our project is to predict the manner in which they did the exercise. The data available has been already separated into two parts: training data and test data. However, test data is solely for prediction purpose since there's no actually exercise type (i.e. "classe" variable) in the dataset for us to verify the result. Therefore, we have to use cross-validation to generate training and testing set among the original training set.

In this case, I just randomly separate the original training data into two parts (3:1), and use the larger set as the training set (named *training*) and test on the other part of the data (named *testing*). 

The details are as follows.

### 1. Load the data

```r
setwd("D:/R/Coursera PML/Project 1")
TotalTraining<-read.csv("pml-training.csv")
TotalTesting<-read.csv("pml-testing.csv")
```

### 2. Separate the training data into two parts

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(333)
TraingIndex<-createDataPartition(y=TotalTraining$classe,p=0.75,list=FALSE)
training<-TotalTraining[TraingIndex,]
testing<-TotalTraining[-TraingIndex,]
```

Let's take a look at the training set we generated.


```r
dim(training)
```

```
## [1] 14718   160
```

If we run the *summary* function on the training set, it will show us that there are many missing values in some columns. The result is too long hence omitted.

### 3. Select predictors

First, we exclude the columns with nearly zero variance.

```r
novarindex<-nearZeroVar(training)
training<-training[,-novarindex]
```

Next, we get rid of columns with lots of missing values. Here we take the threshold of 50%. That is, if 50% or more entries in this column are missing values, we will exclude this column.


```r
lotNA<-function(colvec){
    if (sum(is.na(colvec))/length(colvec)>=0.5){
        return(TRUE)
    }
    else{
        return(FALSE)
    }
}

NAindex<-sapply(training,lotNA)
training<-training[,!NAindex]
```

Finally, we exclude the columns for index,names and time.

```r
training<-training[,-seq(from=1,to=6)]
```

And we got the tidy training set we can build our models on.

```r
dim(training)
```

```
## [1] 14718    53
```

### 4.Build the model
In this case, we choose the random forests algorithm since it's one of the commonly used machine learning algorithms and also very accurate. Here, we load the **randomForest** Package and use the function *randomForest* to fit the model.


```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modFit<-randomForest(classe ~.,data=training)
print(modFit)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.39%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4181    4    0    0    0 0.0009557945
## B   11 2833    4    0    0 0.0052668539
## C    0   13 2552    2    0 0.0058433970
## D    0    0   18 2392    2 0.0082918740
## E    0    0    1    3 2702 0.0014781966
```
### 5. Check error rate

First, we test the in sample error rate. That is, apply the model on the training set and check the error rate.

```r
predictontrain<-predict(modFit,training)
print(confusionMatrix(predictontrain,training$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
As we can see, the model fits the training data extremely well (zero error rate). To make sure the result is not because of overfitting, let's test the out of sample error rate using the testing set.


```r
predictontest<-predict(modFit,testing)
print(confusionMatrix(predictontest,testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  941    2    0    0
##          C    0    3  852    6    2
##          D    0    0    1  798    2
##          E    0    0    0    0  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9957          
##                  95% CI : (0.9935, 0.9973)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9946          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9916   0.9965   0.9925   0.9956
## Specificity            0.9986   0.9995   0.9973   0.9993   1.0000
## Pos Pred Value         0.9964   0.9979   0.9873   0.9963   1.0000
## Neg Pred Value         1.0000   0.9980   0.9993   0.9985   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1919   0.1737   0.1627   0.1829
## Detection Prevalence   0.2855   0.1923   0.1760   0.1633   0.1829
## Balanced Accuracy      0.9993   0.9955   0.9969   0.9959   0.9978
```

The out of sample error rate is 0.4%. Hence we believe in great confidence that our model will make pretty accurate predictions.

### 6. Make prediction

Finally, we make the predictions using the original test dataset and generate text files for testing purposes.


```r
finalprediction<-predict(modFit,TotalTesting)
print(finalprediction)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
answers<-as.character(finalprediction)
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)
```

