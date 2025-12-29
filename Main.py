#importing required python classes and packages
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization #importing byaseian optimization class
import matplotlib.pyplot as plt #use to visualize dataset vallues
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE#loading SMOTE class to deal with imbalance data
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import svm
import os
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
#loading and displaying BOTNET 2018 dataset
dataset = pd.read_csv("Dataset/UNSW_2018_IoT_Botnet_Full5pc_4.csv", low_memory=False)
dataset
#finding & plotting graph of normal and attacks instances
#visualizing class labels count found in dataset
labels, count = np.unique(dataset['attack'].ravel(), return_counts = True)
print("Normal Records : "+str(count[0]))
print("Attack Records : "+str(count[1]))
height = count
bars = labels
y_pos = np.arange(len(bars))
plt.figure(figsize = (4, 3)) 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Dataset Class Label Graph")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()
#finding and displaying count of missing or null values
dataset.isnull().sum()
#visualizing product quality as number of Low, high and medium quality
#describe and plotting graph of various Product Current Quality %  found in dataset 
dataset.groupby("proto").size().plot.pie(autopct='%.0f%%', figsize=(6, 6))
plt.title("Different Protocol Graph")
plt.xlabel("Protocol Name")
plt.ylabel("Usage %")
plt.show()
#visualizing different protocols used for attack and normal request 
data = dataset[['proto', 'attack']]
plt.figure(figsize=(6,4))
sns.boxplot(data=data, x='proto', y='attack', palette='rainbow')
plt.title("Different Protocols Used by Attacks")
plt.show()
#drop irrelevant attributes
dataset.drop(['pkSeqID', 'category', 'subcategory'], axis = 1,inplace=True)
#using label encoder to convert non-numeric values to numeric values
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
encoder4 = LabelEncoder()
encoder5 = LabelEncoder()
encoder6 = LabelEncoder()
encoder7 = LabelEncoder()
dataset['flgs'] = pd.Series(encoder1.fit_transform(dataset['flgs'].astype(str)))#encode all str columns to numeric
dataset['proto'] = pd.Series(encoder2.fit_transform(dataset['proto'].astype(str)))#encode all str columns to numeric
dataset['saddr'] = pd.Series(encoder3.fit_transform(dataset['saddr'].astype(str)))#encode all str columns to numeric
dataset['sport'] = pd.Series(encoder4.fit_transform(dataset['sport'].astype(str)))#encode all str columns to numeric
dataset['daddr'] = pd.Series(encoder5.fit_transform(dataset['daddr'].astype(str)))#encode all str columns to numeric
dataset['dport'] = pd.Series(encoder6.fit_transform(dataset['dport'].astype(str)))#encode all str columns to numeric
dataset['state'] = pd.Series(encoder7.fit_transform(dataset['state'].astype(str)))#encode all str columns to numeric
dataset
#using PCA we can identify and plot dataset imbalance ratio
pca = PCA(n_components=2)
temp = dataset.values
X = temp[:,0:temp.shape[1]-1]
Y = temp[:,temp.shape[1]-1]
X = pca.fit_transform(X)
plt.figure(figsize=(6,4))
plt.scatter(X[:, 0], X[:, 1], c=Y) 
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Dataset Imbalance Graph")
plt.show() 
#dataset preprocessing like shuffling and normalization
Y = dataset['attack'].ravel()#represents attack or normal
data = dataset.values
X = data[:,0:dataset.shape[1]-1]
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffling dataset values
X = X[indices]
Y = Y[indices]
thr = 50
#normalizing dataset values using min max scaler
scaler = MinMaxScaler(feature_range = (0, 1))
X = scaler.fit_transform(X)#normalize train features
print("Normalize Training Features")
print(X)
#now apply smote algorithm on imbalance dataset for balancing
smote = SMOTE()
X, Y = smote.fit_resample(X, Y)
#dataset class labels after applying smote
labels, count = np.unique(Y, return_counts = True)
print("Normal Records After Smote : "+str(count[0]))
print("Attack Records After Smote: "+str(count[1]))
height = count
bars = ['Normal', 'Attack']
y_pos = np.arange(len(bars))
plt.figure(figsize = (4, 3)) 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Dataset Class Label Graph After Applying SMOTE")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()
#split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print("Total records found in dataset = "+str(X.shape[0]))
print("Total features found in dataset= "+str(X.shape[1]))
print("80% dataset for training : "+str(X_train.shape[0]))
print("20% dataset for testing  : "+str(X_test.shape[0]))
#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []
#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+" Accuracy  : "+str(a))
    print(algorithm+" Precision : "+str(p))
    print(algorithm+" Recall    : "+str(r))
    print(algorithm+" FSCORE    : "+str(f))
    labels = ['Normal', 'Attack']
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 3))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()
#training and evaluating performance of default decision tree algorithm
dt_cls = DecisionTreeClassifier(max_depth=1)
dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict =dt_cls.predict(X_test)#perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("Default Decision Tree Algorithm", y_test, predict)
#will call this function to optimize decision tree hyperparameters using BOGP
def gaussianProcess(max_depth, min_samples_split, max_features):
    params_dt = {}
    params_dt['max_depth'] = max_depth
    params_dt['min_samples_split'] = min_samples_split
    params_dt['max_features'] = max_features
    scores = cross_val_score(DecisionTreeClassifier(random_state=123, **params_dt),
                             X_train, y_train, cv=5).mean()
    score = scores.mean()
    return score
#giving different parameters as input to bayesian optimization and by using this parameters will process using BOGP to
#select best parameters
params_dt = {'max_depth':(5, 10), 'min_samples_split':(0.1, 0.9), 'max_features':(0.1, 0.9)}
dt_bo = BayesianOptimization(gaussianProcess, params_dt, random_state=111)
dt_bo.maximize(init_points=5, n_iter=2)
params_dt = dt_bo.max['params']
print("Best Optimized Parameters Selected by BOGP = "+str(params_dt))
#training and evaluating performance of BOGP optimizied decision tree algorithm
bogp_dt_cls = DecisionTreeClassifier(max_depth=params_dt['max_depth'], max_features=params_dt['max_features'],
                                     min_samples_split=params_dt['min_samples_split'])
bogp_dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
predict = bogp_dt_cls.predict(X_test)#perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("BOGP Optimized Decision Tree", y_test, predict)
#training and evaluating performance of SVM algorithm
svm_cls = svm.SVC()
svm_cls.fit(X_train[0:thr], y_train[0:thr])#train algorithm using training features and target value
predict = svm_cls.predict(X_test) #perform prediction on test data
#call this function with true and predicted values to calculate accuracy and other metrics
calculateMetrics("SVM Algorithm", y_test, predict)
#training CNN deep learning algorithm to predict factory maintenaance
#converting dataset shape for CNN comptaible format as 4 dimension array
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
#creating deep learning cnn model object
cnn_model = Sequential()
#defining CNN layer wwith 32 neurons of size 1 X 1 to filter dataset features 32 times
cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
#defining maxpool layet to collect relevant filtered features from previous CNN layer
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
#creating another CNN layer with 16 neurons to optimzed features 16 times
cnn_model.add(Convolution2D(16, (1, 1), activation = 'relu'))
#max layet to collect relevant features
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
#convert multidimension features to single flatten size
cnn_model.add(Flatten())
#define output prediction layer
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
#compile, train and load CNN model
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train1, y_train1, batch_size = 32, epochs = 5, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")
#perform prediction on test data   
predict = cnn_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
#call this function to calculate accuracy and other metrics
calculateMetrics("Extension CNN", y_test1, predict)
#comparison graph between all algorithms
df = pd.DataFrame([['Default Decision Tree','Accuracy',accuracy[0]],['Default Decision Tree','Precision',precision[0]],['Default Decision Tree','Recall',recall[0]],['Default Decision Tree','FSCORE',fscore[0]],
                   ['Optimized Decision Tree','Accuracy',accuracy[1]],['Optimized Decision Tree','Precision',precision[1]],['Optimized Decision Tree','Recall',recall[1]],['Optimized Decision Tree','FSCORE',fscore[1]],
                   ['SVM','Accuracy',accuracy[2]],['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','FSCORE',fscore[2]],
                   ['Extension CNN','Accuracy',accuracy[3]],['Extension CNN','Precision',precision[3]],['Extension CNN','Recall',recall[3]],['Extension CNN','FSCORE',fscore[3]],
                  ],columns=['Parameters','Algorithms','Value'])
df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
plt.title("All Algorithms Performance Graph")
plt.show()
#display all algorithm performnace
algorithms = ['Default Decision Tree', 'Optimized Decision Tree', 'SVM', 'Extension CNN']
data = []
for i in range(len(accuracy)):
    data.append([algorithms[i], accuracy[i], precision[i], recall[i], fscore[i]])
data = pd.DataFrame(data, columns=['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE'])
data   
