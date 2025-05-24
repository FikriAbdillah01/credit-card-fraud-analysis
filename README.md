# Credit Card Fraud Analysis

## Problem
A credit card is an electronic payment tool that uses a card issued by a bank or financial institution to make transactions. With this practical thing, customers can make an easy and immediate transaction. However, the problem lies on the transaction that not fully secured. However, these transactions are not completely safe. Based on the The Federal Trade Commision 2024 data, the fraud problem significantly rose by six million customers from 2001 to 2023 with the total loss of 10 billion USD.

<p align = "center">
  <img width = "400" height = "300" src = "https://github.com/FikriAbdillah01/credit-card-fraud-analysis/blob/2478ebd7bb3bfa5716a91cb118646d8bd174473a/figures/FTC_graph.png">
</p>

## Objective
The objective of this project is:

- the amount of fraud transaction
- create fraud detection using machine learning model

## Data Description
The data published by Machine Learning Research Group University Libe de Bruxelles. Credit card dataset cannot be uploaded to Github due to large data size. This dataset can be downloaded in Kaggle platform. The dataset contains:

- Time: This feature represents time when the credit card transaction occured 
- the amount of transaction: represent the total transaction occured at that time
- Identity of the customer (V1 to V32): due to the protection regulation of the bank, the identity customer is turned into random numbers
- Class: A column that contains indications of fraud or not a transaction.

## Exploratory Data Analysis
This page explains about what is inside of the dataset.

### The Amount of the Transaction
This subsubpage contains the total of the occured transaction. The figure below shows the distribution data of the amount coloumn. 
- Based on the picture, the majority of the total transaction occured between 0 and 1000 USD, either fraud or not. The plot picture also means that the fraud transactions in that range are hard to be detected.
- Transactions made above this nominal amount are also often carried out with varying values.

<p align = "center">
  <img width = "600" height "300" src = "https://github.com/FikriAbdillah01/credit-card-fraud-analysis/blob/b597919dadca4910004d01c0182187fb7323c5fd/figures/The%20Amount%20of%20Fraud%20Transaction%20(1).png">
</p>

### Class
The class feature contains 0 or 1 that represent the transactions are categorized non-fraud or fraud, respectively.

<p align = "center">
  <img width = "400" height = "400" src = "https://github.com/FikriAbdillah01/credit-card-fraud-analysis/blob/2026523ff91659d09aec80e5f07dffab8c3253a1/figures/Countplot%20of%20CC%20Class.png" alt = "dist class figure">
</p>

- The figure shows that the fraud transaction rarely occured. It less than 0.5% (about 490) of more than 250 thousands transanction in total.
- The difference in number between fraud and non-fraud is significant. The plot shows severe imbalace between those two transactions.

### Customer Identity (V1-V32)

The V1 to V32 features represent the customer personal identity, such as name, address, sex, job, etc. which had been changed by PCA method due to the complexity of the dataset and bank protection regulation.

### Outlier and Skewness
Outliers are data points that have significant differences in value between the values ​​in the dataset. This anomaly can be seen by a boxplot.
- The plot illustrates the numerous outliers in data. 

<p align = "center">
  <img width = "400" height = "300" src = "https://github.com/FikriAbdillah01/credit-card-fraud-analysis/blob/5cba031f89890f256f55633141aec67de6f95f66/figures/Boxplot%20of%20the%20Amount%20Transaction.png">
</p>

Skewness is a measure of how asymmetrical the data distribution is and is one way to determine the symmetry of the data distribution. If the mean, median, mode of data are in single one value, then the feature is symmetric. There is one another way to measure the skewness or asymetrical data spread, by using Pearson first coefficient of skewness or Pearson second coefficient of skewness. The range of normal data distribution is between -1 and 1. 

<p align = "center">
  <img width = "800" height = "300" src = "https://github.com/FikriAbdillah01/credit-card-fraud-analysis/blob/e3a51f5ce11a1f3bc917b51e20bbb8154691b1ab/figures/Skewness%20Score%20for%20each%20Features.png">
</p>

- These features are beyond normal distribution score. Moreover, the Class, Amount, and V8 are the most skewed data. We will handle the most severe ones.

## Preprocessing Step
### Oversampling
The class feature has severe data imbalance that needs to be addressed by using oversampling method called Synthetic Minnority Oversampling Technique (SMOTE). The result of the technique can be seen in figure below. 

<p align = "center">
  <img width ="500" height = "400" src = "">
</p>
- The sythetic fraud class is oversampled by SMOTE in order to avoid bias of model accuracy.

### Handling Outlier
The outlier of data can be handled by using two metdos, imputation or Isolation Forest. The result of those methods can we see in figure below.

<p align = "center">
  <img width = "900" height = "200" src = "https://github.com/FikriAbdillah01/credit-card-fraud-analysis/blob/799f50d82591796debfb37a34cb9d719bca57643/figures/Amount%20Plot%20Dist.png">
</p>

- If Raw data distribution compared to the Isolation Forest, there is no significant alteration between them. However, Imputation method notably shifted the data. 

## Machine Learning Result
This page dedicated to explain the machine learning metrics result. 

## Conclusion
The conclusion of the research is.......

## Futher Research
