# <center>An application of SVM and Neural Network: <br> Bank Marketing Campaigns</center>

<a id="20"></a>
Prepared by: Punloeuvivorth ROM
 ### Table of Contents  
 
1. [Introduction](#1) <br>
2. [The dataset](#2) <br>
3. [Exploratory data analysis (EDA)](#3) <br>
    3.1 [Target variable](#4) <br>
    3.2 [Bank client data](#5) <br>
    3.3 [Last contact information](#6) <br>
    3.4 [Features correlation](#7) <br>  
4. [Data Preprocessing](#8) <br>
    4.1 [Outliers](#9) <br>
    4.2 [Resampling](#10) <br>
    4.3 [Dealing with categorical variables](#11) <br>
    4.4 [Splitting data](#12) <br>
    4.5 [Features scaling](#13) <br> 
5. [Modelling](#14) <br>
    5.1 [Logistic Regression](#15) <br>
    5.2 [SVM Classifier](#16) <br>
    5.3 [ANN](#17) <br>  
6. [Results Summary](#18)

<a id="1"></a>
# 1. Introduction

This project is focused on the use of machine learning algorithms for classification tasks, specifically the Support Vector Machines (SVM) and Neural Networks (NN) implemented using the Keras API with TensorFlow backend. The main objective of the project is to implement and compare the performance of these algorithms in accurately categorizing data into different classes based on input features. The most traditional model in classification tasks, Logistic Regression, will also be provided for benchmark purpose.

This project will also include optimization using GridSearchCV. Grid Search is a technique used to select the best hyperparameters for a model, which can greatly improve its performance. The optimization will be performed on both the SVM and NN models to determine the best set of hyperparameters for each algorithm.

The performance of the optimized SVM and NN models will be evaluated using various metrics, such as accuracy, F1 score, and ROC-AUC score. The results of this project will provide valuable insight on the performance of SVM and NN algorithms, as well as the impact of hyperparameter optimization on their performance.

<a id="2"></a>
# 2. The dataset
[Table of contents](#20) <br>  

The dataset used in this project is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The classification goal is thus to predict if the client will subscribe (yes/no) a term deposit (variable y).

The data contains 41188 observations with 20 inputs and one target. There is no missing values in this dataset if we treat the ‘unknown’ (in most of the categorical attributes) as a category class. The attribute information can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).


```python
df = pd.read_csv(path + '/bank-additional-full.csv', sep = ';')
df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>261</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>149</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>226</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>151</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>307</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>


<a id="3"></a>
# 3. Exploratory data analysis (EDA)
[Table of contents](#20) <br>  

  3.1 [Target variable](#4) <br>
  3.2 [Bank client data](#5) <br>
  3.3 [Last contact information](#6) <br>
  3.4 [Features correlation](#7) <br>  

<a id="4"></a>
## 3.1 Target variable

Since the objective is to predict the response from the clients whether they would subscribe to a term deposit or not, we may interest first with the target variables. 

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/1_y_dist.png" alt="1_y_dist.png" style="width:600px;"/>

With the total of 41188 observations, we note that the positive responses occur only 4640 times, that equals to 11.27%. From this insight, we now know that target variable is moderately imbalanced which may cause overfitting in model estimations. This problem may occur as the training model will spend most of its time on negative examples and not learn enough from positive ones. We need to take into account of this problem before building predictive models. 

<a id="5"></a>
## 3.2 Bank client data
[Table of contents](#20)

For the data about bank clients, ‘age’ was the only numeric variable while the rest are categorical variables. We note that most of the client ages are around 30-40, are married, have at least high school diploma, and mostly work as either admins or in blue-collar jobs (outside-the-office jobs). We have very few clients whose ages above 70 that seem to be outliers. We will further examine this later.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/1_y_dist.png" alt="1_y_dist.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/2_age.png" alt="2_age.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/3_client.png" alt="3_client.png" style="width:600px;"/>

We also have the data about the clients credit information, which show that most of the clients have no credit problems.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/4_credit.png" alt="4_credit.png" style="width:600px;"/>

<a id="6"></a>
## 3.3 Last contact information
[Table of contents](#20)

In this marketing campaign, more than one contact to the same client was required. That is why the information about the client previous contacts was collected. The contacts were made mainly via cellular during May throughout the weekdays. The duration of each call is also recorded, yet it may not be useful for our prediction. This is due to the fact that the duration is not known before a call is performed; and after the end of the call y is obviously known which make this attribute highly affects the output target (e.g., if duration=0 then y='no'). Thus, we only present this input for benchmark purposes only and will be discarded in our predictive models.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/5_contact.png" alt="5_contact.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/6_duration.png" alt="6_duration.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/7_poutcome.png" alt="7_poutcome.png" style="width:600px;"/>

The average number of contacts of 2-3 times performed during this campaign for each client (variable ‘campaign’). Since most of the clients were not been contacted during the last campaign (noted as 999 in ‘pdays’), the largest class in the outcome for previous marketing campaign (’poutcome’) were labeled ‘nonexistent’.

To get more insight related to the social and economic context, five macro indicators also given. Since they are all macro variables, the information remain the same for each individual at the time of recorded data. 

<a id="7"></a>
## 3.4 Features correlation
[Table of contents](#20)

As exspected, the variable ‘duration’ is the most correlated attribute to the target. The employment variation rate (’emp.var.rate’) is highly correlated to the euribor 3month rate (’euribor3m’) and the number of employees (’nr.employed’). Such high correlated variables should be excluded in most linear models, yet this is not the case in our project since we focus more on nonlinear modelling (deep learning).

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/8_cor.png" alt="8_cor.png" style="width:800px;"/>

<a id="8"></a>
# 4. Data preprocessing
[Table of contents](#20)

4.1 [Outliers](#9) <br>
4.2 [Resampling](#10) <br>
4.3 [Dealing with categorical variables](#11) <br>
4.4 [Splitting data](#12) <br>
4.5 [Features scaling](#13) <br> 

<a id="9"></a>
## 4.1 Outliers

As presented earlier, we found a few client with age above 70 that may be the outliers. 

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/9_age70.png" alt="9_age70.png" style="width:600px;"/>

With the total number of outliers (469 obs) compared to the total observation of 41188, it may not seem that much of a deal. One may consider removing them as it's only 1.14% of the whole dataset . Yet, if we take into account these outliers with the target variable, we may suffer from potential imformation loss (about half of the clients age above 70 response positively to the marketing campaign). We decided to keep these values by just keeping in mind that there exist some outliers in 'age'. 

There may also exist some outliers in other numerical variables, yet most of them are not related to the individual but to time.

<a id="10"></a>
## 4.2 Resampling
[Table of contents](#20)

In dealing with the imbalanced dataset, we decided to do the random undersampling method as we have such large dataset and the number of the minority class is considerable for undersampling. Despite the possible data leakage, we still perform the undersampling before train-test split since the aim of this project is to get the most accurate model (rather than the most generalized model) in predicting the outcome for this given dataset.


```python
## Resampling data with undersamppling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(df.iloc[:,:-1], df.y)

new_df = X_resampled.assign(y = y_resampled.values)
new_df.shape
```




    (9280, 21)



<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/10_resample.png" alt="10_resample.png" style="width:600px;"/>

<a id="11"></a>
## 4.3 Dealing with categorical variables
[Table of contents](#20)

Machine learning models can only work with numerical values. For this reason, it is necessary to transform the categorical values of the relevant features into numerical ones. In this step, we use LabelEncoder for the target variable and dummies encoding for the rest of the categorical variables. After this feature encoding, we get new datasets containing 52 variables (including target and dummies variables).


```python
# Source: https://towardsdatascience.com/encoding-categorical-variables-one-hot-vs-dummy-encoding-6d5b9c46e2db

# Creating new df (dropping 'duration')
new_df = new_df.drop('duration', axis=1)

# Label Encoding for target variable
label = LabelEncoder()
new_df['y'] = label.fit_transform(new_df['y'])

# Dummy Encoding for categorical features 
# 'drop_first=True : Dummy encoding
# 'drop_first=False : One-hot encoding
category_cols = new_df.select_dtypes(['object'])

for col in category_cols:
    new_df = pd.concat([new_df.drop(col, axis=1),
                        pd.get_dummies(new_df[col], prefix=col, prefix_sep='_',
                                       drop_first=True, dummy_na=False)], axis=1)
    
new_df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
      <th>job_blue-collar</th>
      <th>job_entrepreneur</th>
      <th>job_housemaid</th>
      <th>job_management</th>
      <th>job_retired</th>
      <th>...</th>
      <th>month_aug</th>
      <th>month_dec</th>
      <th>month_jul</th>
      <th>month_jun</th>
      <th>month_mar</th>
      <th>month_may</th>
      <th>month_nov</th>
      <th>month_oct</th>
      <th>month_sep</th>
      <th>day_of_week_mon</th>
      <th>day_of_week_thu</th>
      <th>day_of_week_tue</th>
      <th>day_of_week_wed</th>
      <th>poutcome_nonexistent</th>
      <th>poutcome_success</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.858</td>
      <td>5191.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44</td>
      <td>6</td>
      <td>12</td>
      <td>1</td>
      <td>-1.8</td>
      <td>92.893</td>
      <td>-46.2</td>
      <td>1.354</td>
      <td>5099.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>-1.8</td>
      <td>92.893</td>
      <td>-46.2</td>
      <td>1.291</td>
      <td>5099.1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>-1.8</td>
      <td>92.893</td>
      <td>-46.2</td>
      <td>1.281</td>
      <td>5099.1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>



<a id="12"></a>
## 4.4 Splitting data
[Table of contents](#20)

If we use the whole dataset which is imbalanced, we may interest in splitting the data using StratifiedKFold from sklearn to ensure the same proportion in each class after the split. Yet, since we already under-sampled the data, the simple train-test split is enough with the ratio of 80/20.


```python
X = new_df.drop('y', axis=1)
y = new_df.y

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)
print("X_train: ",X_train.shape)
print("X_test:  ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test:  ",y_test.shape)
```

    X_train:  (7424, 51)
    X_test:   (1856, 51)
    y_train:  (7424,)
    y_test:   (1856,)
    

<a id="13"></a>
## 4.5 Features scaling
[Table of contents](#20)

The goal of applying Feature Scaling is to make sure features are on almost the same scale so that each feature is equally important and make it easier to process by most ML algorithms. StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance.


```python
# Features scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

<a id="14"></a>
# 5. Modelling
[Table of contents](#20) <br> 

5.1 [Logistic Regression](#15) <br>
5.2 [SVM Classifier](#16) <br>
5.3 [ANN](#17) 

<a id="15"></a>
## 5.1 Logistic Regression


```python
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
lgr_pred = lgr.predict(X_test)

print('F1 Score: {:.5f}'.format(f1_score(y_test, lgr_pred)))
print('ROC-AUC Score: {:.5f}'.format(roc_auc_score(y_test, lgr_pred)))
print('Accuracy Score: {:.5f}'.format(accuracy_score(y_test, lgr_pred)))
```

    F1 Score: 0.69988
    ROC-AUC Score: 0.73571
    Accuracy Score: 0.73384
    

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/11_lgr.png" alt="11_lgr.png" style="width:400px;"/>


<a id="16"></a>
## 5.2 SVM Classifier
[Table of contents](#20)


```python
# Base model with default rbf kenel 
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print('F1 Score: {:.5f}'.format(f1_score(y_test, svc_pred)))
print('ROC-AUC Score: {:.5f}'.format(roc_auc_score(y_test, svc_pred)))
print('Accuracy Score: {:.5f}'.format(accuracy_score(y_test, svc_pred)))
```

    F1 Score: 0.69165
    ROC-AUC Score: 0.73155
    Accuracy Score: 0.72953
    


```python
# Polynomial kenel
svc_poly = SVC(kernel='poly', random_state=42)
svc_poly.fit(X_train, y_train)
svc_poly_pred = svc_poly.predict(X_test)

print('F1 Score: {:.5f}'.format(f1_score(y_test, svc_poly_pred)))
print('ROC-AUC Score: {:.5f}'.format(roc_auc_score(y_test, svc_poly_pred)))
print('Accuracy Score: {:.5f}'.format(accuracy_score(y_test, svc_poly_pred)))
```

    F1 Score: 0.68719
    ROC-AUC Score: 0.72834
    Accuracy Score: 0.72629
    


```python
# Linear kenel
svc_linear = SVC(kernel='linear', random_state=42)
svc_linear.fit(X_train, y_train)
svc_linear_pred = svc_linear.predict(X_test)

print('F1 Score: {:.5f}'.format(f1_score(y_test, svc_linear_pred)))
print('ROC-AUC Score: {:.5f}'.format(roc_auc_score(y_test, svc_linear_pred)))
print('Accuracy Score: {:.5f}'.format(accuracy_score(y_test, svc_linear_pred)))
```

    F1 Score: 0.68473
    ROC-AUC Score: 0.72619
    Accuracy Score: 0.72414
    

With these three kernels, we note that the model with 'rbf' kernel present the best result. We then use GridSearchCV to search for the optimal model. A more detailed explaination on the effect of the parameters 'gamma' and 'C' of the Radial Basis Function (RBF) kernel SVM can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).


```python
# hyperparameter tuning for rbf kernel
param_grid = {'C': [1,10,100,1000], 'gamma': [0.01,0.02,0.05,0.1,0.5,1]}

svc = SVC(random_state=42)
svc_grid = GridSearchCV(svc, param_grid, cv=5, verbose=0)

svc_grid.fit(X_train, y_train)
```





```python
print(svc_grid.best_estimator_)
```

    SVC(C=1, gamma=0.02, random_state=42)
    

    


```python
# Predicting with final SVC
svc = SVC(C=1, gamma=0.02, random_state=42, probability=True) #get probab. for roc curve
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print('F1 Score: {:.5f}'.format(f1_score(y_test, svc_pred)))
print('ROC-AUC Score: {:.5f}'.format(roc_auc_score(y_test, svc_pred)))
print('Accuracy Score: {:.5f}'.format(accuracy_score(y_test, svc_pred)))
```

    F1 Score: 0.69165
    ROC-AUC Score: 0.73155
    Accuracy Score: 0.72953
    


```python
print('Training score: {:.5f}'.format(svc.score(X_train, y_train)))
print('Test score: {:.5f}'.format(svc.score(X_test, y_test)))
```

    Training score: 0.76387
    Test score: 0.72953
    

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/12_svc.png" alt="12_svc.png" style="width:400px;"/>

It turns out that our base model with 'rbf' kernel was already the best model.





<a id="17"></a>
## 5.3 ANN
[Table of contents](#20)

We first define a sequentially connected network with three layers. To define the fully connected layer use the Dense class of Keras:
- The first layer with 51 inputs and activation function as relu
- The hidden layer with 25 neurons (follow the simplest rule of thumb: N.input/2)
- Finally, at the output layer, we use 1 unit and activation as sigmoid because it is a binary classification problem.

In compiling the model we must specify some additional parameters to better evaluate the model and to find the best set of weights to map inputs to outputs.
- Loss Function – one must specify the loss function to evaluate the set of weights on which model will be mapped. We will use cross-entropy as a loss function under the name of 'binary_crossentropy' which is used for binary classification.
- Optimizer – used to optimize the loss. We will use 'adam' which is a popular version of gradient descent and gives the best result in most problems.


```python
# Model building function
def build_model():
    model = Sequential()
    model.add(Dense(units = 25, activation = 'relu', input_dim = 51))
    model.add(Dense(units = 25, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
```


```python
# Fitting the model
model = KerasClassifier(build_fn = build_model,
                        batch_size = 10, epochs = 100, verbose=0)

model.fit(X_train, y_train, batch_size = 10, epochs = 100,verbose = 0)
```



```python
# Accuracy score on 10folds CV 
acc_CV = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
mean = acc_CV.mean()
variance = acc_CV.std()
```


```python
print('Train set CV: \n')
print('Accuracy mean: {:.5f}'.format(mean))
print('Accuracy deviation: {:.5f}'.format(variance))
```

    Train set CV: 
    
    Accuracy mean: 0.67363
    Accuracy deviation: 0.02069
    


```python
# Predicting on test set 
y_pred = model.predict(X_test)

print('Test set results: \n')
print('F1 Score: {:.5f}'.format(f1_score(y_test, y_pred)))
print('ROC-AUC Score: {:.5f}'.format(roc_auc_score(y_test, y_pred)))
print('Accuracy Score: {:.5f}'.format(accuracy_score(y_test, y_pred)))
```

    Test set results: 
    
    F1 Score: 0.66254
    ROC-AUC Score: 0.67808
    Accuracy Score: 0.67726
    


```python
# Tuning with GridSearchCV
model = KerasClassifier(build_fn = build_model)
parameters = {'batch_size': [10, 20, 30, 50],
              'epochs': [100, 150, 200]}

model_grid = GridSearchCV(estimator = model,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5, n_jobs = -1)

model_grid = model_grid.fit(X_train, y_train, verbose = 0)
```


```python
# summarize results
print("Best: %f using %s" % (model_grid.best_score_, model_grid.best_params_))
means = model_grid.cv_results_['mean_test_score']
stds = model_grid.cv_results_['std_test_score']
params = model_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

    Best: 0.699489 using {'batch_size': 50, 'epochs': 100} /n
    0.666220 (0.013719) with: {'batch_size': 10, 'epochs': 100}
    0.671473 (0.009030) with: {'batch_size': 10, 'epochs': 150}
    0.666490 (0.014819) with: {'batch_size': 10, 'epochs': 200}
    0.671739 (0.006997) with: {'batch_size': 20, 'epochs': 100}
    0.665816 (0.009836) with: {'batch_size': 20, 'epochs': 150}
    0.664467 (0.003112) with: {'batch_size': 20, 'epochs': 200}
    0.683324 (0.005481) with: {'batch_size': 30, 'epochs': 100}
    0.674434 (0.003481) with: {'batch_size': 30, 'epochs': 150}
    0.671874 (0.007042) with: {'batch_size': 30, 'epochs': 200}
    0.699489 (0.003130) with: {'batch_size': 50, 'epochs': 100}
    0.682649 (0.009580) with: {'batch_size': 50, 'epochs': 150}
    0.672144 (0.010606) with: {'batch_size': 50, 'epochs': 200}
    


```python
# Fitting the final model 
ann_grid = KerasClassifier(build_fn = build_model,
                             batch_size = 50, epochs = 100, verbose=0)

ann_grid = ann_grid.fit(X_train, y_train, verbose = 0)
ann_pred = ann_grid.predict(X_test)

print('F1 Score: {:.5f}'.format(f1_score(y_test, ann_pred)))
print('ROC-AUC Score: {:.5f}'.format(roc_auc_score(y_test, ann_pred)))
print('Accuracy Score: {:.5f}'.format(accuracy_score(y_test, ann_pred)))
```

    F1 Score: 0.67980
    ROC-AUC Score: 0.69423
    Accuracy Score: 0.69343
    
<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/13_nn.png" alt="13_nn.png" style="width:400px;"/>

We obtain a better result with this final model as compare to the previous one. Hence, the time we spent in performing the hyperparameter tuning is worth to wait. ;)




<a id="18"></a>
# 6. Results summary
[Table of contents](#20)

We already seen the application of the 3 models: Logistic regression, SVM Classifier, and ANN. We will present again the performance metrics for each (selected) model via classification report and ROC curve. 


```python
# Classification Report of the 3 model
print('Classification Report: \n')
print('1. Logistic Regression: \n')
print(classification_report(y_test, lgr_pred, digits=5))
print('2. SVM Classifier: \n')
print(classification_report(y_test, svc_pred, digits=5))
print('3. Neural Network: \n')
print(classification_report(y_test, ann_pred, digits=5))
```

    Classification Report: 
    
    1. Logistic Regression: 
    
                  precision    recall  f1-score   support
    
               0    0.68229   0.85996   0.76089       914
               1    0.81818   0.61146   0.69988       942
    
        accuracy                        0.73384      1856
       macro avg    0.75024   0.73571   0.73038      1856
    weighted avg    0.75126   0.73384   0.72992      1856
    
    2. SVM Classifier: 
    
                  precision    recall  f1-score   support
    
               0    0.67607   0.86543   0.75912       914
               1    0.82070   0.59766   0.69165       942
    
        accuracy                        0.72953      1856
       macro avg    0.74838   0.73155   0.72538      1856
    weighted avg    0.74948   0.72953   0.72487      1856
    
    3. Neural Network: 
    
                  precision    recall  f1-score   support
    
               0    0.66895   0.74726   0.70594       914
               1    0.72335   0.64119   0.67980       942
    
        accuracy                        0.69343      1856
       macro avg    0.69615   0.69423   0.69287      1856
    weighted avg    0.69656   0.69343   0.69267      1856
    
    

When comparing the performance of classification models, a ROC curve can provide valuable information on the trade-off between the true positive rate (TPR) and and the false positive rate (FPR). The ROC curve allows us to evaluate the models' performance over all possible threshold settings, providing a comprehensive view of their performance. In our case, the ROC curves of the logistic regression and SVC models are similar, it suggests that they perform similarly in terms of the TPR-FPR trade-off. 

The area under the ROC curve (AUC) represents the probability that a randomly chosen positive instance will be ranked higher than a randomly chosen negative instance by the model.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/14_roc.png" alt="14_roc.png" style="width:800px;"/>

Based on the results of the comparison between the three models, logistic regression and support vector machine (SVM) appear to perform similarly, offering similar accuracy and precision in their predictions. However, the artificial neural network (ANN) model seems to be the weakest performer in the comparison.

In conclusion, the main purpose of this project was originally to implement and compare the SVM classifier with ANN in classification tasks by using logistic regression as the benchmark. Yet, we found a surprising result where the most traditional model is found to be the best classifier. While logistic regression and SVM models seem to perform well, the ANN model may need further improvement to reach a level of performance comparable to the other two models. However, a less complex model is more preferable at anycase.





### References:

- Dataset: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- 7 Techniques to Handle Imbalanced Data: https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
- Build your first Neural Network model using Keras:<br> https://www.analyticsvidhya.com/blog/2021/05/develop-your-first-deep-learning-model-in-python-with-keras/
- Kaggle Notebooks:
    - [Bank Marketing + Classification + ROC,F1,RECALL...](https://www.kaggle.com/code/henriqueyamahata/bank-marketing-classification-roc-f1-recall)
    - [Credit Fraud || Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)
    - [SVM Classifier Tutorial](https://www.kaggle.com/code/prashant111/svm-classifier-tutorial)
    - [Deep Learning Tutorial for Beginners](https://www.kaggle.com/code/kanncaa1/deep-learning-tutorial-for-beginners)
