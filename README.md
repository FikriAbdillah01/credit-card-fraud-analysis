# Credit Card Fraud Analysis

## Problem
A credit card is an electronic payment tool that uses a card issued by a bank or financial institution to make transactions. With this practical thing, customers can make an easy and immediate purchases. However, the problem lies on the transaction that not fully secured. Based on the The Federal Trade Commision 2024 data, the fraud problem significantly rose by six million customers from 2001 to 2023 with the total loss of 10 billion USD.

<p align = "center">
  <img width = "600" height = "350" src = "figures/FTC_graph.png">
</p>

## Objective
The objective of this project is:

- the amount of fraud transaction at the time
- How many transaction occured at that time
- How much amount of USD the fraudster get at the time
- create fraud detection using machine learning model
- To see the difference of machine learning performance and feature importances between imputation and oversampling method through Area Under Curve (AUC).

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

<p align = "center">
  <img width = "600" height "300" src = "figures/The%20Amount%20of%20Fraud%20Transaction%20(1).png">
</p>

- Based on the picture, the majority of the total transaction occured between 0 and 1000 USD, either fraud or not. The plot picture also means that the fraud transactions in that range are hard to be detected.

- Transactions made above this nominal amount are also often carried out with varying values.

<p align = "center">
  <img width = 800 height = "300" src = "figures/plot of the amount transaction.png">
</p>

- From the 50-hour recorded transaction, the amount of fraud one is less than ten thousand USD. 

- The maximum total transaction value obtained by the fraudster from this credit card incident was around 5000 USD in the 10th hour. The criminal received it at a busy time when many customers were doing transaction.

### Class
The class feature contains 0 or 1 that represent the transactions are categorized non-fraud or fraud, respectively.

<p align = "center">
  <img width = "400" height = "300" src = "figures/Countplot of CC Class.png" alt = "dist class figure">
</p>

- The figure shows that the fraud transaction rarely occured. It less than 0.5% (about 490) of more than 250 thousands transanction in total.
- The difference in number between fraud and non-fraud is significant. The plot shows severe imbalace between those two categories of transaction.

### Customer Identity (V1-V32)

The V1 to V32 features represent the customer personal identity, such as name, address, sex, job, etc. which had been changed by PCA method due to the complexity of the dataset and bank protection regulation. The distribution of the sample, V1 - V4, can be seen in the image below.

<p align = 'center'>
  <img width = "" height = "" src = "figure/">
</p>

### Outlier and Skewness
Outliers are data points that have significant differences in value between the values ​​in the dataset. This anomaly can be seen by a boxplot.
- The plot illustrates the numerous outliers in the total purchases. There are various amount of transaction above the range of the most occured ones.

<p align = "center">
  <img width = "400" height = "300" src = "figures/Boxplot of the Amount Transaction.png">
</p>

- If we look to the features other than the Amount

Skewness is a measure of how asymmetrical the data distribution is and is one way to determine the symmetry of the data distribution. If the mean, median, mode of data are in one single value, then the distribution of the feature is symmetric. There is one another way to measure the skewness or asymetrical data spread, by using Pearson first coefficient of skewness or Pearson second coefficient of skewness. The range of normal data distribution is between -1 and 1. 

<p align = "center">
  <img width = "800" height = "300" src = "figures/Skewness Score for each Features.png">
</p>

- These features are beyond normal distribution score. Moreover, the Class, Amount, V8, and V28 are the most skewed data.

## Preprocessing Step
### Oversampling
The class feature has severe data imbalance that needs to be addressed by using oversampling method called Synthetic Minnority Oversampling Technique (SMOTE). The result of the technique can be seen in figure below. 

<p align = "center">
  <img width ="500" height = "400" src = "figures/oversampling process.png">
</p>
- The technique is used to overcome the severe imbalance issue by having the minority class compensate for the majority (oversampling).

<p align = "center">
  <img width = "500" height = "400" src = ".png">
</p>

- The sample of the customer identity, V1 to V4, data distribution shows some of them realized  that their credit credit card had been fraud by criminal.

### Handling Outlier
The outlier of data can be handled by using two methods, imputation or Isolation Forest. The result of those methods can we see in figure below.

<p align = "center">
  <img width = "800" height = "250" src = "figures/Amount Plot Dist.png">
</p>

- If Raw data distribution compared to the Isolation Forest, there is no significant alteration between them. However, the imputation method notably shifted the data. 

## Machine Learning Result
### Feature Selection

This page dedicated to explain feature selection result by machine leaning model.

### Model Performance
This project uses several open-sources such as XGBoost, Decision Tree, and Random Forest. Due to limited computational power and time, some models cannot be used to predict fraudulent of the credit card.


<p align = "center">
  <img width = "400" height = "350" src = "">
</p>

- From the plot, we see the performance of 


## Conclusion
The conclusion of the research is

- The transaction mostly occured in range between 1 to 1000 USD. 
- The best model, based on the AUC metric, XGBoost can accurately predict the fraud.

## Futher Research

The research suggestion of this project:

- Try another machine learning model to predict the credit card fraud.
- Some preprocessing models, instead of oversampling, may be potentially improve the machine learning performance.

## Reference

## Code

### Amount Feature Distribution Plot

```python
# Fraud-Not Fraud transaction
sns.set_style('whitegrid')
sns.histplot(data = df[df.Class == 1]['Amount'], bins = 30, kde = True, color = 'red', stat = 'density', alpha = 0.4, label = 'Fraud')
sns.histplot(data = df[df.Class == 0]['Amount'], bins = 30, kde = True, color = 'blue', stat = 'density', alpha = 0.2, label = 'Normal')
plt.title('The Amount of Fraud Transaction')
plt.legend()
#plt.savefig('The Amount of Fraud Transaction.png')
#files.download('The Amount of Fraud Transaction.png')
plt.show()
```


### Class Countplot

```python
# Countplot of Class
sns.countplot(data=df, x='Class')
plt.title('Countplot of Class')
plt.xticks([0,1], ['Not Fraud', 'Fraud'])
#plt.savefig('Countplot of CC Class.png')
#files.download('Countplot of CC Class.png')
plt.show()
```

### Customer Identity Plot Distribution

```python

```

### Skewness Score Barplot

```python

```

### Outlier Countplot 
```python

```

### Isolation Forest 

```python

```
