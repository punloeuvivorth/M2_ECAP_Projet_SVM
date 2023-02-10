# <center>L'application de SVM et de réseau de neurones : <br> Campagnes de marketing bancaire</center>

<a id="20"></a>
Punloeuvivorth ROM - M2 ECAP

 ### Table des matières
 
1. [Introduction](#1) <br>
2. [La base de données](#2) <br>
3. [Analyse exploratoire des données](#3) <br>
    3.1 [Variable target](#4) <br>
    3.2 [Données sur les clients](#5) <br>
    3.3 [Dernier contact](#6) <br>
    3.4 [Features correlation](#7) <br>  
4. [Prétraitement des données](#8) <br>
    4.1 [Outliers](#9) <br>
    4.2 [Echantillonnage](#10) <br>
    4.3 [Traitement des variables catégorielles](#11) <br>
    4.4 [Train-test split](#12) <br>
    4.5 [Features scaling](#13) <br> 
5. [Modelling](#14) <br>
    5.1 [Logistic Regression](#15) <br>
    5.2 [SVM Classifier](#16) <br>
    5.3 [ANN](#17) <br>  
6. [Résumé](#18)

<a id="1"></a>
# 1. Introduction

Ce projet est principalement basé sur l'utilisation d'algorithmes de machine learning pour des problèmes de classification, en particulier les Machines à Vecteur de Support (SVM) et les Réseaux de Neurones (NN) implémentés en utilisant l'API Keras avec le backend TensorFlow. L'objectif principal du projet est de mettre en œuvre et de comparer les performances de ces algorithmes pour classer avec précision des données dans différentes classes en fonction des features. Le modèle le plus traditionnel de classification, la régression logistique, sera également utilisé à des fins de benchmark.

Ce projet comprend également l'optimisation à l'aide de GridSearchCV. Le Grid Search est une technique utilisée pour sélectionner les meilleurs hyperparamètres pour un modèle, ce qui peut améliorer considérablement ses performances. L'optimisation sera effectuée sur les modèles SVM et NN afin de déterminer le meilleur ensemble d'hyperparamètres pour chaque algorithme.

Les performances des modèles SVM et NN optimisés seront évaluées à l'aide de diverses métriques, telles que l'accuracy, le F1 score et le ROC-AUC score. Les résultats de ce projet fournissent des informations utiles sur les performances des algorithmes SVM et NN, ainsi que sur l'impact de l'optimisation des hyperparamètres sur leurs performances.

<a id="2"></a>
# 2. La base de données
[Table des matières](#20) <br>  

Le jeu de données utilisé dans ce projet est celui du [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Les données sont liées aux campagnes de marketing direct d'une institution bancaire portugaise. Les campagnes de marketing étaient basées sur des appels téléphoniques. Souvent, plus d'un contact avec le même client était nécessaire pour savoir si le produit (dépôt bancaire) serait ('yes') ou non ('no') souscrit. L'objectif de la classification est donc de prédire si le client va souscrire (yes/no) un dépôt (variable y).

Les données contiennent 41188 observations avec 20 features et un target. Il n'y a pas de valeurs manquantes dans cette base de données si nous traitons les inconnus ("unknown" dans la plupart des attributs catégoriels) comme une classe de catégorie. Les informations sur les attributs peuvent être trouvées [ici](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).


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
[Table des matières](#20) <br>  

  3.1 [Variable target](#4) <br>
  3.2 [Données sur les clients](#5) <br>
  3.3 [Dernier contact](#6) <br>
  3.4 [Features correlation](#7) <br>   

<a id="4"></a>
## 3.1 Variable target

L'objectif étant de prédire la réponse des clients s'ils souscriraient ou non à un dépôt à terme, on peut s'intéresser d'abord à la variable cible.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/1_y_dist.png" alt="1_y_dist.png" style="width:600px;"/>

Avec un total de 41188 observations, nous constatons que les réponses positives ne se produisent que 4640 fois, soit 11,27%. À partir de ces informations, on sait maintenant que la variable cible est déséquilibrée, ce qui peut conduire à un surajustement dans les estimations du modèle. Ce problème peut se produire parce que le modèle d'apprentissage passera la plupart de son temps sur la classe négative et n'apprendra pas suffisamment de la classe positive. Ce problème doit être pris en compte avant de construire des modèles prédictifs. 


<a id="5"></a>
## 3.2 Données sur les clients
[Table des matières](#20)

En ce qui concerne les données relatives aux clients, l'âge est la seule variable numérique, les autres étant des variables catégorielles. On constate que la plupart des clients ont entre 30 et 40 ans, sont mariés, ont au moins un baccalauréat et travaillent principalement comme administrateurs ou comme blue-collar (emplois en dehors du bureau). Nous avons très peu de clients dont l'âge dépasse 70 ans, ce qui semble être une valeur aberrante. On y reviendra plus tard.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/1_y_dist.png" alt="1_y_dist.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/2_age.png" alt="2_age.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/3_client.png" alt="3_client.png" style="width:600px;"/>

On dispose également des données sur les informations de crédit des clients, qui montrent que la plupart des clients n'ont pas de problèmes de crédit.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/4_credit.png" alt="4_credit.png" style="width:600px;"/>

<a id="6"></a>
## 3.3 Derniers contacts
[Table des matières](#20)

Dans cette campagne de marketing, plusieurs contacts avec le même client étaient nécessaires. C'est pourquoi les informations sur les contacts précédents du client ont été collectées. Les contacts ont été effectués principalement par téléphone cellulaire au mois de mai, durant les jours de la semaine. La durée de chaque appel est également enregistrée, mais elle peut ne pas être utile pour notre prédiction. Cela est dû au fait que la durée n'est pas connue avant qu'un appel soit effectué ; et après la fin de l'appel y est évidemment connu ce qui fait que cet attribut affecte fortement le target  (par exemple, si la durée=0 alors y='no'). Par conséquent, on ne présente cette feature qu'à des fins de référence et elle ne sera pas prise en compte dans nos modèles prédictifs.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/5_contact.png" alt="5_contact.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/6_duration.png" alt="6_duration.png" style="width:600px;"/>

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/7_poutcome.png" alt="7_poutcome.png" style="width:600px;"/>

En moyenne, le nombre de contacts de 2-3 fois effectués pendant cette campagne pour chaque client (variable 'campaign'). Étant donné que la plupart des clients n'ont pas été contactés lors de la dernière campagne (noté 999 dans 'pdays'), la classe la plus importante dans le résultat de la campagne de marketing précédente ('poutcome') a été notée 'nonexistent'.

Pour obtenir un meilleur aperçu du contexte social et économique, cinq variables macro sont également disponibles. Comme il s'agit de variables macro, les informations restent les mêmes pour chaque individu au moment de l'enregistrement des données. 

<a id="7"></a>
## 3.4 Features correlation
[Table des matières](#20)

Comme prévu, la variable "duration" est la plus corrélée au target. Le taux de variation de l'emploi ('emp.var.rate') est fortement corrélé au taux euribor a 3 mois ('euribor3m') et au nombre de salariés ('nr.employed'). De telles variables hautement corrélées devraient être exclues dans la plupart des modèles linéaires, mais ce n'est pas le cas dans notre projet puisqu'on se concentre plus sur la modélisation non linéaire (deep learning).

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/8_cor.png" alt="8_cor.png" style="width:800px;"/>

<a id="8"></a>
# 4. Prétraitement des données
[Table des matières](#20)

  4.1 [Outliers](#9) <br>
  4.2 [Echantillonnage](#10) <br>
  4.3 [Traitement des variables catégorielles](#11) <br>
  4.4 [Train-test split](#12) <br>
  4.5 [Features scaling](#13) <br> 

<a id="9"></a>
## 4.1 Outliers

Comme présenté précédemment, on a trouvé quelques clients âgés de plus de 70 ans qui pourraient être des valeurs aberrantes. 

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/9_age70.png" alt="9_age70.png" style="width:600px;"/>

Le nombre total de valeurs aberrantes (469 obs) par rapport à l'observation totale de 41188 peut ne pas sembler si important. On peut envisager de les supprimer car il ne s'agit que de 1,14 % de l'ensemble de données. Cependant, si on prend en compte ces valeurs aberrantes avec la variable cible, on peut souffrir d'une potentielle perte d'information (environ la moitié des clients âgés de plus de 70 ans répondent positivement à la campagne marketing). J'ai décidé de garder ces valeurs en gardant à l'esprit qu'il existe des valeurs aberrantes dans 'age'. 

Il peut également exister des valeurs aberrantes dans d'autres variables numériques, mais la plupart d'entre elles ne sont pas liées à l'individu mais au temps.

<a id="10"></a>
## 4.2 Echantillonnage
[Table des matières](#20)

En traitant la base de données déséquilibrée, on a décidé de faire la méthode de sous-échantillonnage aléatoire car nous avons des données si volumineuses et le nombre de la classe minoritaire est considérable pour le sous-échantillonnage. Malgré la possible "data leakage", on effectue toujours le sous-échantillonnage avant la séparation train-test puisque le but de ce projet est d'obtenir le modèle le plus précis (plutôt que le modèle le plus généralisé) pour prédire le résultat pour cette base de données.


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
## 4.3 Traitement des variables catégorielles
[Table des matières](#20)


Les modèles de machine learning ne peuvent fonctionner qu'avec des valeurs numériques. Pour cette raison, il est nécessaire de transformer les valeurs catégorielles des features en valeurs numériques. On utilise LabelEncoder pour la variable target et le dummies encodeur pour le reste des variables catégorielles. Après cet encodage de features, nous obtenons de nouveaux base de données contenant 52 variables (y compris les variables target et les dummies).


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
## 4.4 Train-test split
[Table des matières](#20)

Si l'on utilise l'ensemble de données qui est déséquilibré, on peut avoir intérêt à diviser les données en utilisant StratifiedKFold de sklearn pour assurer la même proportion dans chaque classe après la division. Cependant, comme on a déjà sous-échantillonné les données, le simple train-test split suffit avec le ratio de 80/20.


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
[Table des matières](#20)

L'objectif de l'application de Feature Scaling est de s'assurer que les features sont à peu près à la même échelle afin que chaque feature soit d'égale importance et de faciliter son traitement par la plupart des algorithmes de ML. On utilise StandardScaler pour standardiser les fonctionnalités.


```python
# Features scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

<a id="14"></a>
# 5. Modelling
[Table des matières](#20) <br> 

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
[Table des matières](#20)


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
    

Avec ces trois noyaux, on constate que le modèle avec noyau 'rbf' présente le meilleur résultat. On utilise alors GridSearchCV pour rechercher le modèle optimal. Une explication plus détaillée sur l'effet des paramètres 'gamma' et 'C' du SVM du noyau de la Radial Basis Function (RBF) peut être trouvée [ici](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).


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

Il s'avère que notre modèle de base avec le noyau 'rbf' était déjà le meilleur modèle!



<a id="17"></a>
## 5.3 ANN
[Table des matières](#20)

On définit d'abord un réseau séquentiellement connecté avec trois couches. Pour définir la couche entièrement connectée, on utilise la classe Dense de Keras :
- La première couche avec 25 neurons, 51 inputs et fonction d'activation comme "relu"
- La deuxième couche cachée avec 25 neurones
- Enfin, au niveau de la couche output, on utilise 1 unité et fonction d'activation comme "sigmoid" car il s'agit d'un problème de classification binaire.

Pour compiler le modèle, il faut spécifier quelques paramètres supplémentaires pour mieux évaluer le modèle et trouver le meilleur ensemble de poids pour mapper les inputs aux outputs.
- Fonction de perte - il faut spécifier la fonction de perte pour évaluer l'ensemble de poids sur lequel le modèle sera mappé. On utilise l'entropie croisée comme fonction de perte sous le nom de 'binary_crossentropy' qui est utilisé pour la classification binaire.
- Optimiseur - utilisé pour optimiser la perte. On utilise 'adam' qui est une version populaire de la descente de gradient et donne le meilleur résultat dans la plupart des problèmes.


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

On obtient un meilleur résultat avec ce modèle final par rapport au précédent. Le temps passé à effectuer le réglage des hyperparamètres vaut la peine d'attendre. ;)




<a id="18"></a>
# 6. Rrésumé
[Table des matières](#20)

Nous avons déjà présenté l'application pour les 3 modèles : régression logistique, SVC et ANN. On présente à nouveau les métriques de performance pour chaque modèle (optimisé) via le rapport de classification et la courbe ROC.


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
    
    

Lors de la comparaison des performances des modèles de classification, une courbe ROC peut fournir des informations précieuses sur le compromis entre le taux de vrais positifs (TPR) et le taux de faux positifs (FPR). La courbe ROC nous permet d'évaluer les performances des modèles sur tous les seuils possibles, offrant une vue complète de leurs performances. Dans notre cas, les courbes ROC des modèles de régression logistique et SVC sont similaires, cela suggère qu'ils fonctionnent de manière similaire en termes de compromis TPR-FPR. 

L'AUC (Area Under the ROC curve) représente la probabilité qu'une instance positive choisie au hasard soit mieux classée qu'une instance négative choisie au hasard par le modèle.

<img src="https://github.com/punloeuvivorth/M2_ECAP_Projet_SVM/blob/main/img/14_roc.png" alt="14_roc.png" style="width:800px;"/>

Selon la comparaison entre les trois modèles, la régression logistique et le SVC semblent fonctionner de manière similaire, offrant une précision similaire dans leurs prédictions. Cependant, le modèle de réseau de neurones artificiels (ANN) semble être le moins performant dans la comparaison.

En conclusion, l'objectif principal de ce projet était à l'origine d'implémenter et de comparer le SVM Classifier avec ANN dans le problème de classification en utilisant la régression logistique comme référence. Cependant, on a trouvé un résultat surprenant où le modèle le plus traditionnel s'avère être le meilleur classificateur. Alors que les modèles de régression logistique et SVM semblent bien fonctionner, le modèle ANN peut nécessiter des améliorations supplémentaires pour atteindre un niveau de performance comparable aux deux autres modèles. Il est important de noter qu'un modèle moins complexe est toujours plus préférable.




### Références:

- Dataset: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- 7 Techniques to Handle Imbalanced Data: https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
- Build your first Neural Network model using Keras:<br> https://www.analyticsvidhya.com/blog/2021/05/develop-your-first-deep-learning-model-in-python-with-keras/
- Kaggle Notebooks:
    - [Bank Marketing + Classification + ROC,F1,RECALL...](https://www.kaggle.com/code/henriqueyamahata/bank-marketing-classification-roc-f1-recall)
    - [Credit Fraud || Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)
    - [SVM Classifier Tutorial](https://www.kaggle.com/code/prashant111/svm-classifier-tutorial)
    - [Deep Learning Tutorial for Beginners](https://www.kaggle.com/code/kanncaa1/deep-learning-tutorial-for-beginners)
