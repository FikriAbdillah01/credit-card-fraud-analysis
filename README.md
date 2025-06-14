# Credit Card Fraud Analysis

## Problem
A credit card is an electronic payment tool that uses a card issued by a bank or financial institution to make transactions. With this practical thing, customers can make an easy and immediate transaction. However, the problem lies on the transaction that not fully secured. However, these transactions are not completely safe. Based on the The Federal Trade Commision 2024 data, the fraud problem significantly rose by six million customers from 2001 to 2023 with the total loss of 10 billion USD.

<p align = "center">
  <img width = "600" height = "350" src = "figures/FTC_graph.png">
</p>

## Objective
The objective of this project is:

- Identifying the peak periods of fraud risk and quantifying the resulting financial losses
- Comparing machine learning models with 3 datasets that go through 3
 preprocessing techniques, raw, imputation, and isolation forest.
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
- V1 to V32: Credit card customer identity.
- Class: A column indicating whether a transaction is fraudulent or not.

## Exploratory Data Analysis
This page explains about what is inside of the dataset.

### The Amount of the Transaction
This subsubpage discuss about the total of the occured transaction. The figure below shows the distribution data of the amount coloumn. 

<p align = "center">
  <img width = "600" height "300" src = "figures/The%20Amount%20of%20Fraud%20Transaction%20(1).png">
</p>

- Based on the picture, the majority of the total transaction occured between 0 and 1000 USD, either fraud or not. The plot picture also tells us that the fraud transactions in that range are hard to be detected.

<p align = "center">
  <img width = 800 height = "300" src = "figures/transaction amount.png">
</p>

- **Legitimate Transactions Overview**: Over a 48-hour period, credit card transactions ranged from $200,000 to $900,000 for legitimate purchases. The total transaction volume in the morning (9th to 22nd hour) reached $11,000,000, while nighttime transactions (22nd to 34th hour) totaled slightly above $4,000,000. These transactions follow a periodic pattern, occurring frequently during the day and significantly declining at night.

- **Fraudulent Transaction Insight**: Fraudulent transaction amounts ranged from $50 to $6,000, with total fraud-related transactions summing to $60,000 over the 48-hour period. During the daytime (9th to 22nd hour), fraud accounted for $22,000 (0.2% of total daytime transactions), while at night (22nd to 34th hour), fraudulent transactions totaled just over $12,000 (0.28% of nighttime transactions). Additionally, while fraud can occur at any time, fraudsters tend to acquire more money during nighttime transactions.

<p align = "center">
  <img width = "800" height = "300" src = "figures/transaction occured.png">
</p>

- **Fraud Probability in Daytime vs. Nighttime**: Over a 48-hour period, 115 thousand daytime transactions were recorded, with just over 170 classified as fraudulent, resulting in a fraud probability of 0.15%. At night, around 54 thousand transactions occurred, with 106 categorized as fraud, yielding a slightly higher fraud probability of just under 0.20%.

- **Peak Fraud Hours and Nighttime Risk**: Fraudulent transactions peaked during the 10th hour, with 45 out of 8,000 transactions being fraudulent (0.56% probability). Nighttime transactions had a higher fraud risk, with 35 out of 1,800 transactions being fraudulent, translating to a fraud probability of 1.94%—1.38% higher than during the daytime.

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
- The plot illustrates the numerous outliers in the total purchases. There are **various amount of transaction above the range** of the most occured ones.

<p align = "center">
  <img width = "400" height = "300" src = "figures/Boxplot of the Amount Transaction.png">
</p>


**Skewness** is a measure of how asymmetrical the data distribution is and is one way to determine the symmetry of the data distribution. If the mean, median, mode of data are in one single value, then the distribution of the feature is symmetric. There is one another way to measure the skewness or asymetrical data spread, by using Pearson first coefficient of skewness or Pearson second coefficient of skewness. The range of normal data distribution is between -1 and 1. 

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

Feature selection is a technique for filtering important features so that limited computing power can be utilized effectively. This dataset has more than 30 features with tens of thousands of transactions recorded in 48 hours. It would be a time inefficiency if all features were included in the model. From this project, the feature is eligible to use to train model if the feature importance score more than 0.05.

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

Confusion matrix is ​​a performance evaluation table of a classification model. This project utilizes the Random Forest model with 3 datasets that are treated differently.

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

- Accuracy measures how well the model correctly classifies both fraudulent and legitimate transactions. A high accuracy score suggests that the model performs well overall.

- Recall (sensitivity) is measure ability of model to accurately predict of the actual fraud samples among the all fraud samples in data. The high recall score means that the model can totally predict actual fraud with small to none failure.

- The model that utilize non-oversampling data can have 100 percent accuracy, but the recall score only around 70 percent. We noticed that this is biased because the number of False Positive and False Negative, based on confusion matrix figure above, are still more than zero. It means that the machine learning still fail to detect a small handfull of credit card fraud and non-fraud. Futhremore, this bias accuracy is the result of severly imbalanced data.

- Model with Isolation Forest cannot detect fraud (0% recall). This indicates that most of the deleted data is considered anomaly. As a result, the model does not predict fraud. Meanwhile, raw data and imputation have sligthly less than 70% recall. It means that the model can correctly guess the fraud transaction around 70% of total fraud prediction.

- Precision measures how accurately the model identifies fraudulent transactions among all transactions it classifies as fraud. A high precision score means that most of the transactions flagged as fraud are actually fraudulent, minimizing false positives (FP). 

- TThe model trained on imputed and non-treatment datasets correctly identifies 83% of the fraudulent transactions among those predicted as fraud. In contrast, the model using Isolation Forest preprocessing fails to detect fraud, suggesting that key fraud indicators may have been removed during preprocessing.

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

- After applying the oversampling method, the Isolation Forest dataset experienced a significant shift, with accuracy dropping to 64%. This decline may be attributed to a high contamination level, which could have led to excessive removal of key fraud indicators during preprocessing.

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

- *Reminder Note*: Each dataset has different features as explained in the sub-chapter feature importances.

### ROC-AUC Curve

The Receiver Operating Characteristic (ROC) Curve helps evaluate a model’s performance by adjusting the decision threshold. The Area Under the Curve (AUC) measures how well the model distinguishes between legitimate and fraudulent transactions—a higher AUC indicates better accuracy. If the score is above 0.5 (or 50%), the model can effectively classify transactions as fraud or non-fraud. The model is set to `random_state = 1`. 

#### The Perfomance with Oversampling Method

<p align = "center">
  <img width = "800" height = "300" src = "figures/ROC Curve different dataset.png">
</p>

- The performance of model with each dataset mendapat nilai AUC lebih dari 0.5. The isolation-forest-treatmen dataset archive the lowest score compared to imputation and non-treatment data. 

#### The Performance of Oversampling with Hyperparameter

<p align = "center">
  <img widht = "800" height = "300" src = "figures/ROC Curve best parameter.png">
</p>

- Applying the Grid Search hyperparameter tuning did not result in any significant performance improvement across all approaches. 

## Conclusion
The conclusion of the research is

- The analysis of 48-hour transaction data shows that fraudulent activity is more frequent at night than during the day, as reflected in the pattern of detected fraud cases. Additionally, financial losses from nighttime fraud are higher compared to daytime, highlighting the increased risk during those hours. The fraudulent transactions over that period resulted in a total financial loss of \$60,000.

- Models trained on datasets from the raw and imputation approaches
 consistently outperform the data using the Isolation Forest.

## Futher Research

The research suggestion of this project:

- Try another machine learning model to predict the credit card fraud.
- Exploring alternative preprocessing models to enhance machine learning performance.

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

<p align = "center">
  <img width = "600" height "300" src = "figures/The%20Amount%20of%20Fraud%20Transaction%20(1).png">
</p>


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
<p align = "center">
  <img width = "400" height = "300" src = "figures/Countplot of CC Class.png" alt = "dist class figure">
</p>

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

line_plotting(x_axis = 'Hour', y_axis = 'Sum', title = 'Amount USD')
```

### The 48-hour Period of Transaction (Occurence)

```python
# How much amount in 48-hour recorded transaction?
line_plotting(x_axis = 'Hour', y_axis = 'Transaction', title = 'Transaction Made')
```
<p align = "center">
  <img width = "800" height = "300" src = "figures/transaction occured.png">
</p>


### The Fraud to Transaction Ratio (day vs night)

```python
# How often credit card transaction in the daylight and night?
#Filter the DataFrame for hours between 9 and 22 (inclusive)
transactions_9_to_22 = df_tmp[(df_tmp['Hour'] >= 9) & (df_tmp['Hour'] <= 22)]

# Filter the DataFrame for hours between 22 to 34
transactions_22_to_34 = df_tmp[(df_tmp['Hour'] >= 22) & (df_tmp['Hour'] <= 34)]


# Sum the 'Transaction' column for both classes
total_transactions_9_to_22 = transactions_9_to_22['Transaction'].sum() # morning
total_transactions_22_to_34 = transactions_22_to_34['Transaction'].sum() # night

print(f"Total transactions in Daylight: {total_transactions_9_to_22}") # daylight
print(f"Total transactions in Night: {total_transactions_22_to_34}") # night

# total fraud transaction
# Filter the DataFrame for fraud transactions between 9th and 22nd hour (inclusive)
fraud_transactions_9_to_22 = df_tmp[(df_tmp['Class'] == 1) & (df_tmp['Hour'] >= 9) & (df_tmp['Hour'] <= 22)]
fraud_transaction_22_to_34 = df_tmp[(df_tmp['Class'] == 1) & (df_tmp['Hour'] >= 22) & (df_tmp['Hour'] <= 34)]
# Sum the 'Transaction' column for fraud transactions
total_fraud_transactions_9_to_22 = fraud_transactions_9_to_22['Transaction'].sum()
total_fraud_transaction_22_to_34 = fraud_transaction_22_to_34['Transaction'].sum()

print(f"Total fraud transactions between hour 9 and 22: {total_fraud_transactions_9_to_22}")
print(f"Total fraud transactions between hour 22 and 34: {total_fraud_transaction_22_to_34}")

# percentage of fraud transaction
# at day
percentage_fraud_day = total_fraud_transactions_9_to_22*100/total_transactions_9_to_22
print('The probabilty of fraud at morning: ', percentage_fraud_day)

# at night
percentage_fraud_night = total_fraud_transaction_22_to_34*100/total_transactions_22_to_34
print('The probabilty of fraud at night: ', percentage_fraud_night)
```

```
Total transactions in Daylight: 115986
Total transactions in Night: 54165

Total fraud transactions between hour 9 and 22: 178
Total fraud transactions between hour 22 and 34: 106

The probabilty of fraud at morning:  0.15346679771696584
The probabilty of fraud at night:  0.19569832917935936
```

### The Amount of Fraud to Transaction Ratio (day vs night)
```python
# sum the 'Amount' Transaction column for both classes
total_amount_9_to_22 = transactions_9_to_22['Sum'].sum() # daylight
total_amount_22_to_34 = transactions_22_to_34['Sum'].sum() # night

print(f"Total amount in Daylight: {total_amount_9_to_22}") # daylight
print(f"Total amount in Night: {total_amount_22_to_34}") # night

# the amount of fraud transaction
#at day
amount_fraud_9_to_22 = fraud_transactions_9_to_22['Sum'].sum()
print('The amount of fraud transaction at morning (in $): ', amount_fraud_9_to_22)

#at night
amount_fraud_22_to_34 = fraud_transaction_22_to_34['Sum'].sum()
print('The amount of fraud transaction at night (in $): ', amount_fraud_22_to_34)

# percentage amount of fraud in transaction
# at day
percentage_amount_fraud_day = amount_fraud_9_to_22*100/total_amount_9_to_22
print('The percentage of amount of fraud transaction at morning: ', percentage_amount_fraud_day)

# at night
percentage_amount_fraud_night = amount_fraud_22_to_34*100/total_amount_22_to_34
print('The percentage of amount of fraud transaction at night: ', percentage_amount_fraud_night)
```

```
Total amount in Daylight: 11024995.419999998
Total amount in Night: 4258374.77

The amount of fraud transaction at morning (in $):  22378.44
The amount of fraud transaction at night (in $):  12084.440000000002

The percentage of amount of fraud transaction at morning:  0.20297913194053738
The percentage of amount of fraud transaction at night:  0.28378056541979757
```
### Outlier Treatment (Imputation)
```python
# Hanlde the outlier by imputation method
def outl_imputation(data, features):
  data_ = data.copy()

  for col in data_.columns:

    lower, upper = outlier_cols(data, col)
    data_.loc[data_[col] < lower, col] = lower
    data_.loc[data_[col] > upper, col] = upper

  return data_

df2 = df_new.drop(['Class'], axis = 1)

imputation_data = outl_imputation(df2, df2.columns)
imputation_data['Class'] = df_new['Class']
imputation_data.head()
```


### Outlier Treatment (Isolation Forest)

```python
# import model and create new data
from sklearn.ensemble import IsolationForest
datacopy = df_new.copy()
datacopy.drop(['Class'], axis = 1, inplace = True)

# Implement isolation forest model
# The 0.1 contamination is a default setting of isolation forest
isolationforest = IsolationForest(n_estimators = 100, contamination = 0.1)
isolationforest.fit(datacopy)

# Anomaly Score
score = isolationforest.decision_function(datacopy)
len(score)

# Find Anomaly
anomaly = isolationforest.predict(datacopy)
anomaly

# input it to new dataset
datacopy['anomaly'] = anomaly
datacopy['score'] = score
datacopy.head()
```

### Feature Selection Raw Data
```python
# Import decision tree model and random forest model
rf_feat_sel_raw = RandomForestClassifier(max_depth=3, random_state=1)

df_copy = df.copy()

X = df_copy.drop(['Class','Time'], axis = 1)
y = df_copy['Class']

rf_feat_sel_raw.fit(X,y)

rf_input3 = pd.DataFrame(rf_feat_sel_raw.feature_importances_, index = X.columns, columns = ['rf_importance']) # Random Forest + Imputation

rf_input3 = rf_input3.sort_values('rf_importance', ascending = False)

# Random Forest Feature Importances with Raw Data
fig = plt.figure(figsize = (12,7))

ax = sns.barplot(data = rf_input3, x = 'rf_importance', y = rf_input3.index, label = 'Random Forest') # dataset Decision Tree + Isolation Forest

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Random Forest Feature Importances (Raw Data)', fontsize = 15)
fig.tight_layout()
fig.set_facecolor('#ADD8E6')
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 12)
#plt.savefig('Decision Tree Feature Importances (raw data).png', dpi = 300)
#files.download('Decision Tree Feature Importances (raw data).png')
plt.show()

# Filter the importances score more than 0.050
important_cols = rf_input3[rf_input3.rf_importance > 0.050].index
df_raw_data = df_copy[important_cols] # As an input
df_raw_data['Class'] = df_copy['Class']
```
<p align = 'center'>
  <img width = "700" height = "500" src = "figures/random forest feature importances raw data.png">
</p>


### Feature Selection Imputation
```python
# Import decision tree model and random forest model
from sklearn.ensemble import RandomForestClassifier

rf_feat_sel_imp = RandomForestClassifier(max_depth=3, random_state=1)

X = imputation_data.drop(['Class','Time'], axis = 1)
y = imputation_data['Class']

rf_feat_sel_imp.fit(X,y)

rf_input1 = pd.DataFrame(rf_feat_sel_imp.feature_importances_, index = X.columns, columns = ['rf_importance']) # Random Forest + Imputation

rf_input1 = rf_input1.sort_values('rf_importance', ascending = False)

# Decision Tree Feature Importances
fig = plt.figure(figsize = (12,7))

ax = sns.barplot(data = rf_input1, x = 'rf_importance', y = rf_input1.index, label = 'Random Forest') # dataset Decision Tree + Isolation Forest

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Random Forest Feature Importances (Imputation)', fontsize = 15)
fig.tight_layout()
fig.set_facecolor('#ADD8E6')
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 12)
#plt.savefig('Random Forest Feature Importances (imputance).png', dpi = 300)
#iles.download('Random Forest Feature Importances (imputance).png')
plt.show()

# Filter the importances score more than 0.05
important_cols = rf_input1[rf_input1.rf_importance > 0.050].index
df_imputance = imputation_data[important_cols] # As an input
df_imputance['Class'] = imputation_data['Class']
df_imputance
```

<p align = "center">
  <img width = "700" height = "500" src = "figures/random forest feature importances imputation.png">
</p>


### Feature Selection Isolation Forest
```python
# Import decision tree model and random forest model
from sklearn.ensemble import RandomForestClassifier

rf_feat_sel_iso = RandomForestClassifier(max_depth=3, random_state=1)

X = isoforest_data.drop(['Class','Time'], axis = 1)
y = isoforest_data['Class']

rf_feat_sel_iso.fit(X,y)

rf_input2 = pd.DataFrame(rf_feat_sel_iso.feature_importances_, index = X.columns, columns = ['rf_importance']) # Random Forest + Imputation

rf_input2 = rf_input2.sort_values('rf_importance', ascending = False)

# Decision Tree Feature Importances
fig = plt.figure(figsize = (12,7))

ax = sns.barplot(data = rf_input1, x = 'rf_importance', y = rf_input1.index, label = 'Random Forest') # dataset Decision Tree + Isolation Forest

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Random Forest Feature Importances (Imputation)', fontsize = 15)
fig.tight_layout()
fig.set_facecolor('#ADD8E6')
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 12)
#plt.savefig('Random Forest Feature Importances (isolation forest).png', dpi = 300)
#iles.download('Random Forest Feature Importances (isolation forest).png')
plt.show()

# Filter the importances score more than 0.1
important_cols = rf_input1[rf_input1.rf_importance > 0.050].index
df_imputance = imputation_data[important_cols] # As an input
df_imputance['Class'] = imputation_data['Class']
df_imputance
```
<p align = "center">
  <img width = "700" height = "500" src = "figures/random forest feature importances isolation forest.png" >

### Random Forest with Raw Data (No Oversampled)
```python
# use raw data
from sklearn.model_selection import train_test_split

X_raw = df_raw_data.drop(['Class'], axis = 1)
y_raw = df_raw_data['Class']

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size = 0.2, random_state = 43)

rf_model = RandomForestClassifier(max_depth=3, random_state=1)

rf_model.fit(X_train_raw, y_train_raw)

# Credit Card Detection of raw data

from sklearn.metrics import classification_report, confusion_matrix

y_pred_rawdata = rf_model.predict(X_test_raw)

print('Classificaton Report by Using Raw Data')
print(classification_report(y_test_raw, y_pred_rawdata))
```
```
Classificaton Report by Using Raw Data
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56856
           1       0.83      0.67      0.74       106

    accuracy                           1.00     56962
   macro avg       0.91      0.83      0.87     56962
weighted avg       1.00      1.00      1.00     56962
```

### Random Forest with Imputation Data (No Oversample)
```python
# use imputation data
from sklearn.model_selection import train_test_split

X_imputance = df_imputance.drop(['Class'], axis = 1)
y_imputance = df_imputance['Class']

X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_imputance, y_imputance, test_size = 0.2, random_state = 43)

rf_model = RandomForestClassifier(max_depth=3, random_state=1)

rf_model.fit(X_train_imp, y_train_imp)

# Credit Card Detection

from sklearn.metrics import classification_report, confusion_matrix

y_pred_imputance = rf_model.predict(X_test_imp)

print('Classificaton Report by Using Imputation Data')
print(classification_report(y_test_imp, y_pred_imputance))
```

```
Classificaton Report by Using Imputation Data
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56644
           1       0.85      0.69      0.76       102

    accuracy                           1.00     56746
   macro avg       0.93      0.84      0.88     56746
weighted avg       1.00      1.00      1.00     56746
```

### Random Forest with Isolation Forest Data (No Oversample)

```python
# use isolation forest data
from sklearn.model_selection import train_test_split

X_iso = df_isoforest_data.drop(['Class'], axis = 1)
y_iso = df_isoforest_data['Class']

X_train_iso, X_test_iso, y_train_iso, y_test_iso = train_test_split(X_iso, y_iso, test_size = 0.2, random_state = 43)

rf_model = RandomForestClassifier(max_depth=3, random_state=1)

rf_model.fit(X_train_iso, y_train_iso)

# Credit Card Detection

from sklearn.metrics import classification_report, confusion_matrix

y_pred_isoforest = rf_model.predict(X_test_iso)

print('Classificaton Report by Using Isolation Forest')
print(classification_report(y_test_iso, y_pred_isoforest))
```

```
Classificaton Report by Using Isolation Forest
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     51063
           1       0.00      0.00      0.00         8

    accuracy                           1.00     51071
   macro avg       0.50      0.50      0.50     51071
weighted avg       1.00      1.00      1.00     51071
```


### Confusion Matrix of Three Different Approach (No Oversample)
```python
# Decision Tree Heatmap
fig, axs = plt.subplots(ncols=3, nrows=1 , figsize = (13,5), sharex = True, sharey = True)
sns.heatmap(confusion_matrix(y_test_raw, y_pred_rawdata), annot = True, fmt = 'd', ax = axs[0])
sns.heatmap(confusion_matrix(y_test_imp, y_pred_imputance), annot = True, fmt = 'd', ax = axs[1])
sns.heatmap(confusion_matrix(y_test_iso, y_pred_isoforest), annot = True, fmt = 'd', ax = axs[2])
axs[0].set_title('Raw data', fontsize = 14)
axs[0].set_xlabel('Predicted', fontsize = 14)
axs[0].set_ylabel('Truth', fontsize = 14)
axs[1].set_title('Imputation', fontsize = 14)
axs[1].set_xlabel('Predicted', fontsize = 14)
axs[2].set_title('Isolation Forest', fontsize = 14)
axs[2].set_xlabel('Predicted', fontsize = 14)
fig.suptitle('Confusion Matrix non-Oversampling', fontsize = 15)
fig.tight_layout()
#plt.savefig('rf dt conf mat no oversampling.png', dpi = 300)
#files.download('rf dt conf mat no oversampling.png')
plt.show()
```
<p align = "center">
  <img width = "800" height = "300" src = "figures/rf dt conf mat no oversampling.png">
</p>

### Oversampling SMOTE on Raw Dataset
```python
# import undesampling and oversampling
from imblearn.over_sampling import SMOTE

oversampling = SMOTE()

# Train dataset oversampling of raw data
X_train_raw_over, y_train_raw_over = oversampling.fit_resample(X_train_raw, y_train_raw)

# Test dataset oversampling of raw data
X_test_raw_over, y_test_raw_over = oversampling.fit_resample(X_test_raw, y_test_raw)
```

### Oversampling SMOTE on Imputation Data
```python
# Train Dataset oversampling of imputation data
X_train_imp_over, y_train_imp_over = oversampling.fit_resample(X_train_imp, y_train_imp)

# Test dataset oversampling of imputation data
X_test_imp_over, y_test_imp_over = oversampling.fit_resample(X_test_imp, y_test_imp)
```

### Oversampling SMOTE on Isolation Forest Data
```python
# Train Dataset oversampling of isolation forest data
X_train_iso_over, y_train_iso_over = oversampling.fit_resample(X_train_iso, y_train_iso)

# Test dataset oversampling of isolation forest data
X_test_iso_over, y_test_iso_over = oversampling.fit_resample(X_test_iso, y_test_iso)
```

### The Fraud Detection using Raw Data (Oversampled)
```python
# Train the data

rf_raw = RandomForestClassifier(max_depth=3, random_state=1)

rf_raw.fit(X_train_raw_over, y_train_raw_over)

# Credit Card Detection

from sklearn.metrics import classification_report, confusion_matrix

#y_pred_rawdata_over = rf_raw.predict(X_test_raw)
y_pred_rawdata_over = rf_raw.predict(X_test_raw_over) # Modified to predict on the oversampled test data

print('Classificaton Report by Using Raw Data with Oversampling')
print(classification_report(y_test_raw_over, y_pred_rawdata_over))
```
```
Classificaton Report by Using Raw Data with Oversampling
              precision    recall  f1-score   support

           0       0.88      0.98      0.93     56856
           1       0.98      0.86      0.92     56856

    accuracy                           0.92    113712
   macro avg       0.93      0.92      0.92    113712
weighted avg       0.93      0.92      0.92    113712
```

### The Fraud Detectin using Imputation Data (Oversampled)
```python
# Train the model

rf_imp = RandomForestClassifier(max_depth=3, random_state=1)
rf_imp.fit(X_train_imp_over, y_train_imp_over)

# Credit Card Fraud Detection

y_pred_imputance_over = rf_imp.predict(X_test_imp_over)

print('Classificaton Report by Using Imputation Data')
print(classification_report(y_test_imp_over, y_pred_imputance_over))
```

```
Classificaton Report by Using Imputation Data
              precision    recall  f1-score   support

           0       0.88      1.00      0.93     56644
           1       1.00      0.86      0.92     56644

    accuracy                           0.93    113288
   macro avg       0.94      0.93      0.93    113288
weighted avg       0.94      0.93      0.93    113288
```

### The Fraud Detection using Isolation Forest (Oversampled)
```python
# Train the model

rf_iso = RandomForestClassifier(max_depth=3, random_state=1)

rf_iso.fit(X_train_iso_over, y_train_iso_over)

# Credit Card Fraud Detection

y_pred_isodata_over = rf_iso.predict(X_test_iso_over)

print('Classificaton Report by Using Isolation Forest Data')
print(classification_report(y_test_iso_over, y_pred_isodata_over))
```

```
Classificaton Report by Using Isolation Forest Data
              precision    recall  f1-score   support

           0       0.59      0.87      0.70     51063
           1       0.75      0.41      0.53     51063

    accuracy                           0.64    102126
   macro avg       0.67      0.64      0.62    102126
weighted avg       0.67      0.64      0.62    102126
```

### Confusion Matrix of Three Different Approach (Oversample)
```python
# Decision Tree Heatmap
fig, axs = plt.subplots(ncols=3, nrows=1 , figsize = (13,5), sharex = True, sharey = True)
sns.heatmap(confusion_matrix(y_test_raw_over, y_pred_rawdata_over), annot = True, fmt = 'd', ax = axs[0])
sns.heatmap(confusion_matrix(y_test_imp_over, y_pred_imputance_over), annot = True, fmt = 'd', ax = axs[1])
sns.heatmap(confusion_matrix(y_test_iso_over, y_pred_isodata_over), annot = True, fmt = 'd', ax = axs[2])
axs[0].set_title('Raw data', fontsize = 14)
axs[0].set_xlabel('Predicted', fontsize = 14)
axs[0].set_ylabel('Truth', fontsize = 14)
axs[1].set_title('Imputation', fontsize = 14)
axs[1].set_xlabel('Predicted', fontsize = 14)
axs[2].set_title('Isolation Forest', fontsize = 14)
axs[2].set_xlabel('Predicted', fontsize = 14)
fig.suptitle('Confusion Matrix with Oversampling', fontsize = 15)
fig.tight_layout()
plt.savefig('rf dt conf mat oversampling.png', dpi = 300)
files.download('rf dt conf mat oversampling.png')
plt.show()
```
<p align = "center">
  <img width = "800" height = "300" src = "figures/rf dt conf mat oversampling.png">
</p>


### The AUC-ROC Curve of Three Approaches (Oversample)
```python
# Make roc curve of each model
from sklearn.metrics import roc_curve, roc_auc_score

# Predict probability of oversampled dataset
y_pred_rawdata_over_prob = rf_raw.predict_proba(X_test_raw_over)[:,1]
y_pred_isodata_over_prob = rf_iso.predict_proba(X_test_iso_over)[:,1]
y_pred_imputance_over_prob = rf_imp.predict_proba(X_test_imp_over)[:,1]

# Find false positive rate (specificity) and true positive rate (sensitivity)
fpr_raw, tpr_raw, _ = roc_curve(y_test_raw_over, y_pred_rawdata_over_prob)
fpr_iso, tpr_iso, _ = roc_curve(y_test_iso_over, y_pred_isodata_over_prob)
fpr_imputance, tpr_imputance, _ = roc_curve(y_test_imp_over, y_pred_imputance_over_prob)

auc_iso_over = roc_auc_score(y_test_iso_over, y_pred_isodata_over_prob)
auc_raw_over = roc_auc_score(y_test_raw_over, y_pred_rawdata_over_prob)
auc_imputance_over = roc_auc_score(y_test_imp_over, y_pred_imputance_over_prob)

# Plot ROC curve
fig = plt.figure(figsize = (13,5))

sns.lineplot(x = fpr_raw, y = tpr_raw, label = 'Raw Data (AUC = %0.2f)' % auc_raw_over)
sns.lineplot(x = fpr_iso, y = tpr_iso, label = 'Isolation Forest (AUC = %0.2f)' % auc_iso_over)
sns.lineplot(x = fpr_imputance, y = tpr_imputance, label = 'Imputation (AUC = %0.2f)' % auc_imputance_over)
sns.lineplot(x = [0,1], y = [0,1], color = 'black', linestyle = '--')

plt.xlabel('False Positive Rate(Specificity)', fontsize = 14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize = 14)
plt.title("ROC Curve of Random Forest with Diffrent Dataset", fontsize = 15)
plt.legend(loc = 'lower right', fontsize = 14)
fig.tight_layout()
plt.savefig('ROC Curve different dataset.png', dpi = 300)
files.download('ROC Curve different dataset.png')
plt.show()
```
<p align = "center">
  <img width = '800' height = "300" src = "figures/ROC Curve different dataset.png">
</p>

### Credit Card Fraud Detection using Grid Search (Raw Data)
```python
# Create random forest model enchanced by stratified k-fold
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'min_samples_split':[2,4],
    'min_samples_leaf':[1,2]
}

# Create Model
rf_raw = RandomForestClassifier(random_state = 1)

# Make GridSearchCV model
rf_gridsearch_raw = GridSearchCV(estimator = rf_raw, param_grid = rf_param_grid, cv = 3, n_jobs = -1, verbose = 2)

# Fit the model
rf_gridsearch_raw.fit(X_train_raw_over, y_train_raw_over)

# Best parameter
print('Random Forest Raw Data best Params: ')
print(rf_gridsearch_raw.best_params_)
print('-----------------')
print('Random Forest Raw Data best estimator: ')
print(rf_gridsearch_raw.best_estimator_)

# Get the best parameter model
best_rf_raw_over = rf_gridsearch_raw.best_estimator_

# Train the best model
y_pred_best_raw_over = best_rf_raw_over.predict(X_test_raw_over)

print('Classificaton Report by Using Imputation Data')
print(classification_report(y_test_raw_over, y_pred_best_raw_over))
```
```
Fitting 3 folds for each of 16 candidates, totalling 48 fits
Random Forest Raw Data best Params: 
{'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
-----------------
Random Forest Raw Data best estimator: 
RandomForestClassifier(max_depth=5, n_estimators=50, random_state=1)

Classificaton Report by Using Imputation Data
              precision    recall  f1-score   support

           0       0.88      0.98      0.93     56856
           1       0.98      0.87      0.92     56856

    accuracy                           0.92    113712
   macro avg       0.93      0.92      0.92    113712
weighted avg       0.93      0.92      0.92    113712

```
### Credit Card Fraud Detection using Grid Search (Imputation)
```python
# Create Model
rf_imp = RandomForestClassifier(random_state = 1)

# Make GridSearchCV model
rf_gridsearch_imp = GridSearchCV(estimator = rf_imp, param_grid = rf_param_grid, cv = 3, n_jobs = -1, verbose = 2)

# Fit the model
rf_gridsearch_imp.fit(X_train_imp_over, y_train_imp_over)

# Best parameter
print('Random Forest Imputation Data best Params: ')
print(rf_gridsearch_imp.best_params_)
print('-----------------')
print('Random Forest Imputation Data best estimator: ')
print(rf_gridsearch_imp.best_estimator_)

# Get the Best Parameter
best_rf_imp_over = rf_gridsearch_imp.best_estimator_

# predict the best model

y_pred_best_imp_over = best_rf_imp_over.predict(X_test_imp_over)

print('Classificaton Report by Using Imputation Data')
print(classification_report(y_test_imp_over, y_pred_best_imp_over))
```

```
Fitting 3 folds for each of 16 candidates, totalling 48 fits
Random Forest Imputation Data best Params: 
{'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
-----------------
Random Forest Imputation Data best estimator: 
RandomForestClassifier(max_depth=5, min_samples_leaf=2, n_estimators=50,
                       random_state=1)

Classificaton Report by Using Imputation Data
              precision    recall  f1-score   support

           0       0.89      0.99      0.94     56644
           1       0.98      0.88      0.93     56644

    accuracy                           0.93    113288
   macro avg       0.94      0.93      0.93    113288
weighted avg       0.94      0.93      0.93    113288
```
### Credit Card Fraud Detection Using Grid Search (Isolation Forest)
```python
# Create Model
rf_iso = RandomForestClassifier(random_state = 1)

# Make GridSearchCV model
rf_gridsearch_iso = GridSearchCV(estimator = rf_iso, param_grid = rf_param_grid, cv = 3, n_jobs = -1, verbose = 2)

# Fit the model
rf_gridsearch_iso.fit(X_train_iso_over, y_train_iso_over)

# Best parameter
print('Random Forest Isolation Forest Data best Params: ')
print(rf_gridsearch_iso.best_params_)
print('-----------------')
print('Random Forest Isolation Forest Data best estimator: ')
print(rf_gridsearch_iso.best_estimator_)

# Get the best parameter
best_rf_iso_over = rf_gridsearch_iso.best_estimator_

# Train the best model

y_pred_best_iso_over = best_rf_iso_over.predict(X_test_iso_over)

print('Classificaton Report by Using Imputation Data')
print(classification_report(y_test_iso_over, y_pred_best_iso_over))
```
```
Fitting 3 folds for each of 16 candidates, totalling 48 fits
Random Forest Isolation Forest Data best Params: 
{'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
-----------------
Random Forest Isolation Forest Data best estimator: 
RandomForestClassifier(max_depth=5, n_estimators=50, random_state=1)

Classificaton Report by Using Imputation Data
              precision    recall  f1-score   support

           0       0.60      0.91      0.72     51063
           1       0.81      0.38      0.52     51063

    accuracy                           0.65    102126
   macro avg       0.70      0.65      0.62    102126
weighted avg       0.70      0.65      0.62    102126
```

### The Grid Search Confusion Matrix of Three Approaches (Oversampled)
```python
# Decision Tree Heatmap
fig, axs = plt.subplots(ncols=3, nrows=1 , figsize = (13,5), sharex = True, sharey = True)
sns.heatmap(confusion_matrix(y_test_raw_over, y_pred_best_raw_over), annot = True, fmt = 'd', ax = axs[0])
sns.heatmap(confusion_matrix(y_test_imp_over, y_pred_best_imp_over), annot = True, fmt = 'd', ax = axs[1])
sns.heatmap(confusion_matrix(y_test_iso_over, y_pred_best_iso_over), annot = True, fmt = 'd', ax = axs[2])
axs[0].set_title('Raw data', fontsize = 14)
axs[0].set_xlabel('Predicted', fontsize = 14)
axs[0].set_ylabel('Truth', fontsize = 14)
axs[1].set_title('Imputation', fontsize = 14)
axs[1].set_xlabel('Predicted', fontsize = 14)
axs[2].set_title('Isolation Forest', fontsize = 14)
axs[2].set_xlabel('Predicted', fontsize = 14)
fig.suptitle('Confusion Matrix with Best Parameter', fontsize = 15)
fig.tight_layout()
#plt.savefig('rf dt conf mat best parameter.png', dpi = 300)
#files.download('rf dt conf mat best parameter.png')
plt.show()
```
<p align = "center">
<img width = "900" height = "300" src = "figures/rf dt conf mat Best Parameter.png">
</p>

### The Grid Search AUC-ROC Curve of Three Approaches (Oversampled)
```python
# Make roc curve of each model
from sklearn.metrics import roc_curve, roc_auc_score

# Predict probability of oversampled dataset
y_pred_rawdata_best_prob = best_rf_raw_over.predict_proba(X_test_raw_over)[:,1]
y_pred_isodata_best_prob = best_rf_iso_over.predict_proba(X_test_iso_over)[:,1]
y_pred_imputance_best_prob = best_rf_imp_over.predict_proba(X_test_imp_over)[:,1]

# Find false positive rate (specificity) and true positive rate (sensitivity)
fpr_raw, tpr_raw, _ = roc_curve(y_test_raw_over, y_pred_rawdata_best_prob)
fpr_iso, tpr_iso, _ = roc_curve(y_test_iso_over, y_pred_isodata_best_prob)
fpr_imputance, tpr_imputance, _ = roc_curve(y_test_imp_over, y_pred_imputance_best_prob)

auc_iso_over = roc_auc_score(y_test_iso_over, y_pred_isodata_best_prob)
auc_raw_over = roc_auc_score(y_test_raw_over, y_pred_rawdata_best_prob)
auc_imputance_over = roc_auc_score(y_test_imp_over, y_pred_imputance_best_prob)

# Plot ROC curve
fig = plt.figure(figsize = (13,5))

sns.lineplot(x = fpr_raw, y = tpr_raw, label = 'Raw Data (AUC = %0.2f)' % auc_raw_over)
sns.lineplot(x = fpr_iso, y = tpr_iso, label = 'Isolation Forest (AUC = %0.2f)' % auc_iso_over)
sns.lineplot(x = fpr_imputance, y = tpr_imputance, label = 'Imputation (AUC = %0.2f)' % auc_imputance_over)
sns.lineplot(x = [0,1], y = [0,1], color = 'black', linestyle = '--')

plt.xlabel('False Positive Rate(Specificity)', fontsize = 14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize = 14)
plt.title("ROC Curve of Random Forest with Diffrent Dataset", fontsize = 15)
plt.legend(loc = 'lower right', fontsize = 14)
fig.tight_layout()
#plt.savefig('ROC Curve best parameter.png', dpi = 300)
#files.download('ROC Curve best parameter.png')
plt.show()
```
<p align = "center">
<img width = "" height = "" src = "figures/ROC Curve best parameter.png">
</p>
