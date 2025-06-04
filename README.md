# Credit Card Fraud Analysis (On Going)

## Problem
A credit card is an electronic payment tool that uses a card issued by a bank or financial institution to make transactions. With this practical thing, customers can make an easy and immediate transaction. However, the problem lies on the transaction that not fully secured. However, these transactions are not completely safe. Based on the The Federal Trade Commision 2024 data, the fraud problem significantly rose by six million customers from 2001 to 2023 with the total loss of 10 billion USD.

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

## Limitation
Limitation that exist in this project:

- Inadequate computing hardware makes this project time-consuming to execute on large amount dataset like credit card fraud, so not all open-source models are used.
- The data used was published in 2016. Moreover, this data comes from several bank samples in Belgium, so it is necessary to have the latest data and calibrate the model to obtain good detection accuracy.

## Tools
Tools we used in this project:

- python ver 2.2.2
- matplotlib versi 3.10.0
- seaborn ver 0.13.2
- scikit learn ver 1.6.1
- numphy versi 2.0.2
- pandas ver 2.2.2
- imbalanced-learn ver 0.13.0

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
  <img width = 800 height = "300" src = "figures/transaction amount.png">
</p>

- From the 48-hour recorded transaction, the amount of fraud one is less than ten thousand USD - about 0.1% of normal credit card transaction. 

- The maximum total transaction obtained by the fraudster from this credit card incident was around 5000 USD in the 10th hour. The criminal received it at a busy time when many customers were doing transaction.

<p align = "center">
  <img width = "800" height = "300" src = "figures/transaction occured.png">
</p>

- The figure above illustrates of the credit card utilization that have been recorded over a 48-hour period. Normal transactions that occur take place from morning to evening and occur periodically. The majority of customer use credit card at day, then significantly dropped at night. Meanwile, the fraud transaction always happened, whether day or night. From the record, the peak of the credit card fraud occured at 11th and around 25th hour.


### Class
The class feature contains 0 or 1 that represent the transactions are categorized non-fraud or fraud, respectively.

<p align = "center">
  <img width = "400" height = "300" src = "figures/Countplot of CC Class.png" alt = "dist class figure">
</p>

- The figure shows that the fraud transaction rarely occured. It less than 0.5% (about 490) of more than 250 thousands transanction in total.
- The difference in number between fraud and non-fraud is significant. The plot shows severe imbalace between those two categories of transaction.

### Customer Identity (V1-V32)

The V1 to V32 features represent the customer personal identity, such as name, address, sex, job, etc. which had been changed by PCA method due to the complexity of the dataset and bank protection regulation. 

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

### Handling Outlier
The outlier of data can be handled by using two methods, imputation or Isolation Forest. The result of those methods can we see in figure below.

<p align = "center">
  <img width = "800" height = "250" src = "figures/Amount Plot Dist.png">
</p>

- If Raw data distribution compared to the Isolation Forest, there is no significant alteration between them. However, the imputation method notably shifted the data. 

## Machine Learning Result
### Feature Selection

Feature selection is a technique for filtering important features so that limited computing power can be utilized effectively. This dataset has more than 30 features with tens of thousands of transactions recorded in 48 hours. It would be a lot of time wasted if all features were included in the model. From this project, we will select features that get a feature importance score of more than 0.05.

#### Raw Data
We need to compare the raw, isolation, inputation version dataset. Raw data is the dataset without outlier treatment. 

<p align = 'center'>
  <img width = "700" height = "500" src = "figures/random forest feature importances raw data.png">
</p>

- The feature importances by no-outlier-treatment data shows that there are six features will be the input to the model, V17, V12, V14, V16, V10, and V11.

#### Imputation Version

The imputation method is used to eliminate outliers in the data by substitute them with random variables that are still included in the (Interquartile Range) IQR.

<p align = "center">
  <img width = "700" height = "500" src = "figures/random forest feature importances imputation.png">
</p>

- The figure above shows the rank based on feature importances score by random forest model with imputation method. Ther are seven features with the score above 0.05, that is V17, V11, V10, V12, V16, V7, and V3.

#### Isolation Forest Version

Isolation Forest is an unsupervised model used to detect data points that are considered anomalies (outliers). 

<p align = "center">
  <img width = "700" height = "500" src = "figures/random forest feature importances isolation forest.png" >

- Based on the Random Forest model, V4, V10, V14, and V17 are features with highest feature importance scores. 

- If we compared all of dataset with different preprocessing method, we can find that V17 is the most important feature in the dataset. 

### Model Performance
#### Confusion Matrix without Oversampling Dataset

Confusion matrix is ​​a performance evaluation table of a classification model.This project utilizes the Random Forest model with 3 datasets that are treated differently.

<p align = "center">
  <img width = "800" height = "300" src = "figures/rf dt conf mat no oversampling.png">
</p>

- From the plot, we see the performance of decision tree and random forest model. They can detect more than 50000 of non fraud transaction and about 70 for otherwise.

- Isolation Forest dataset in total is not as much as both raw and imputation data. It because some of the data has been considered anomaly and then removed.


| Preprocessing Method | Accuracy | Recall | F1 | Precision |
| :-----------------:| :------:| :------: | :-------: | :-----:|
| Raw                |   100 % | 67%  | 74% | 83% |
| Isolation Forest   |   100 % | 0%   | 0%  | 0%   |
| Imputation         |   100 % | 69%  | 76% | 83%   |

- The accuracy is how well the model detect fraud and normal transaction. The high accuracy means that the model can predict all fraud and non-fraud transaction correctly with small to none mistake. 

- The model that utilize non-oversampling data can have 100 percent accuracy, but the recall score only around 70 percent. We noticed that this is biased because the number of False Positive and False Negative, based on confusion matrix figure above, are still more than zero. It means that the machine learning still fail to detect a handfull of credit card fraud and non-fraud. Futhremore, this bias accuracy is the result of severly imbalanced data.

- Recall (sensitivity) is measure ability of model to accurately predict of the actual fraud samples among the all fraud samples in data. The high recall score means that the model can totally predict actual fraud with small to none failure.

- Model with Isolation Forest cannot detect fraud (0% recall). This indicates that most of the deleted data is considered anomaly. As a result, the model does not predict fraud. Meanwhile, raw data and imputation have sligthly less than 70% recall. It means that the model can correctly guess the fraud transaction around 70% of total fraud prediction.

- Precision is a model ability to accurately predict fraud  

#### Confusion Matrix with Oversampling Dataset

This subchapter discuss what happened to the model after using oversampling method to significant imbalanced data. 

<p align = "center">
  <img width = "800" height = "300" src = "figures/rf dt conf mat oversampling.png">
</p>

- Based on Confusion Matrix above, the model utilize raw and imputation data can detect around 50 thousand credit card that are suspected of being fraud.

| Preprocessing Method | Accuracy | Recall | F1 | Precision |
| :----------:| :------:| :------: | :-------: | :------:|
| Raw  |  92 % | 86% | 92% | 98% |
| Isolation Forest   |  64 %  | 41% | 53% | 75%|
| Imputation       |   93 % | 86% | 93% | 100 %|

- After using oversampling method, the isolation forest dataset drastically changed. The accuracy singnificantly dropped to 64%. However, this report frequently changed when the code executed. The Raw and imputation performed well and does not experienced a significant change after executed.

#### Confusion Matrix with Oversampled Data and Selected Parameter

<p align = "center">
  <img width = "800" height = "300" src = "figures/rf dt conf mat best parameter.png">
</p>

| Preprocessing Method | Accuracy | Recall | F1 | Precision |
| :--------------:| :------:| :------: | :-------: | :-------:|
| Raw Data       |   92% | 87% | 92% | 98% |
| Isolation Forest   | 65% | 38% | 52% | 81% |
| Imputation        |  93% | 88% | 93% | 98%|

- There is no significant change in all aspect after using Grid Search hyperparameter method.

- Reminder Note: Each dataset has different features as explained in the sub-chapter feature importances.

### ROC-AUC Curve

The Receiving Operating Characteristic Curve (ROC) is a curve that shows the classification performace of the model by varying treshold, while Area Under Curve (AUC) is a measure how well . The model is set to `random_state = 1`. 

#### The Perfomance with Oversampling Method

<p align = "center">
  <img width = "800" height = "300" src = "figures/ROC Curve different dataset.png">
</p>

- The performance each dataset treatment are good.

#### The Performance of Oversampling with Hyperparameter

<p align = "center">
  <img widht = "800" height = "300" src = "figures/ROC Curve best parameter.png">
</p>

- Isolation Forest model AUC score slightly dropped by 0.05 points after use Grid Search hyperparameter method. 

## Conclusion
The conclusion of the research is

- The transaction mostly occured in range between 1 to 1000 USD. 
- The best model, based on the AUC metric, XGBoost can accurately predict the fraud.

## Futher Research

The research suggestion of this project:

- Try another machine learning model to predict the credit card fraud.
- Some preprocessing models, instead of SMOTE, may be potentially improved the machine learning performance.



## Reference

- Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

- Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon.

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

### The 48-hour Period of Transaction (amount USD)

```python
# aggregate time
df['Hour'] = df['Time'].apply(lambda x: np.floor(x/3600))

tmp = df.groupby(['Hour','Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df_tmp = pd.DataFrame(tmp)
df_tmp.columns = ['Hour', 'Class', 'Min', 'Max', 'Transaction', 'Sum', 'Mean', 'Median', 'Var']
df_tmp.head()

# create funtion of line plotting
def line_plotting(dataset = df_tmp, x_axis=None, y_axis=None, title = None):
  fig, axs = plt.subplots(ncols = 2, nrows = 1, figsize = (13,5), sharey = False)

  sns.lineplot(data = dataset.loc[dataset.Class == 0][[x_axis, y_axis]], x = x_axis, y = y_axis, ax = axs[0])
  sns.lineplot(data = dataset.loc[dataset.Class == 1][[x_axis, y_axis]], x = x_axis, y = y_axis, ax = axs[1], color = 'red')

  for ax in axs:
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific (False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    ax.tick_params(axis = 'x', labelsize = 12)
    ax.tick_params(axis = 'y', labelsize = 12)

  plt.suptitle('The 48-hour Period of Transaction (in terms of {})'.format(title), fontsize = 14)
  fig.set_facecolor('#a0cdf8')
  axs[0].set_title('Normal Transaction', fontsize = 14)
  axs[1].set_title('Fraud Transaction', fontsize = 14)
  axs[0].set_xlabel('Hour', fontsize = 14)
  axs[1].set_xlabel('Hour', fontsize = 14)
  axs[0].set_ylabel(y_axis, fontsize = 14)
  axs[1].set_ylabel(y_axis, fontsize = 14)
  fig.tight_layout()

  #plt.savefig('Plot of {} Transaction.png'.format(y_axis), dpi = 300)
  #files.download('Plot of {} Transaction.png'.format(y_axis))
  return plt.show()
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

### Isolation Forest (Handle Anomaly)

```python

```