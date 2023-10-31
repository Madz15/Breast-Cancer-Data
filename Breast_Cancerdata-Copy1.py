#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf


# In[2]:


import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics


# In[3]:


df = pd.read_csv('breast-cancer.csv')


# In[4]:


df[df.isnull().any(axis = 1)]
CD = df.copy()


# In[5]:


CD.shape


# In[6]:


CD.columns


# In[7]:


##CHANGING OUR DATA INTO BINARY NUMBERS 
CD['node.caps']= (CD['node.caps']=='yes').astype(int)
CD['irradiat']= (CD['irradiat']=='yes').astype(int)
CD['Class']= (CD['Class']=='recurrence-events').astype(int)


# In[8]:


CD.head()


# In[9]:


quad = {'left_up':1, 'left_low': 2, 'right_up':3, 'right_low':4, 'central':5} 
CD.head()


# In[10]:


age = {'20-29':24.5, '30-39':34.5,'40-49':44.5,'50-59':54.5, '60-69':64.5,'70-79':74.5,'80-89':84.5,'90-99':94.5}
CD = CD.replace({'Age': age})
CD.head()


# In[11]:


Menopause = {'premeno':1, 'ge40': 2, 'lt40':3} 
CD = CD.replace({'Menopause': Menopause})


# In[12]:


tumor = {'0-4':2, '5-9':7,'10-14':12,'15-19':17, '20-24':22,'25-29':27,'30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52}
CD = CD.replace({'tumor.size': tumor})
CD.head()


# In[13]:


nodes = {'0-2':1, '3-5':4,'6-8':7,'9-11':10, '12-14':13,'15-17':16,'18-20':19,'21-23':22,'24-26':25,'27-29':28,'30-32':31,'33-35':34,
        '36-38':37,'39':39}
CD = CD.replace({'inv.nodes': nodes})
(CD['inv.nodes'].describe)
CD.head()


# In[14]:


quad = {'left_up':1, 'left_low': 2, 'right_up':3, 'right_low':4, 'central':5} 
CD = CD.replace({'breast.quad': quad})
CD['breast.quad'] = CD['breast.quad']


CD.head()


# In[15]:


breast = {'left':1, 'right':2} 
CD = CD.replace({'breast': breast})
CD.head()


# In[16]:


CD.groupby('Class').hist(figsize=(10, 10))


# In[17]:


CD[CD.isnull().any(axis = 1)]
CD = CD.dropna()
CD.head()


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics


# In[19]:


CD


# In[20]:


##Factors affecting risk of metastasis
figure= px.histogram(df, x = "Class",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Class")

figure.show()


# In[21]:


figure = px.histogram(df, x = "Age",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Age")

figure.show()


# In[22]:


figure = px.histogram(df, x = "Menopause",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Menopause")

figure.show()


# In[23]:


figure = px.histogram(df, x = "tumor.size",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Tumor size")

figure.show()


# In[24]:


figure = px.histogram(df, x = "inv.nodes",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Tumor size")

figure.show()


# In[25]:


figure = px.histogram(df, x = "deg.malig",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Tumor size")

figure.show()


# In[26]:


figure = px.histogram(df, x = "breast",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Tumor size")

figure.show()


# In[27]:


##Factors affecting risk of metastasis
figure= px.histogram(df, x = "breast.quad",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Class")

figure.show()


# In[28]:


figure= px.histogram(df, x = "irradiat",
color =  "node.caps",
title = "Factors influencing the likelihood of lymph node metastasis: Class")

figure.show()


# In[29]:


from scipy.stats import chi2_contingency


# In[30]:


chisqt = pd.crosstab(df["Age"], df["node.caps"])
print(chisqt)


# In[31]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[32]:


significance_level = 0.05
print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject HYPOTHESIS') 
else: 
    print('ACCEPT HYPOTHESIS')


# In[33]:


chisqt = pd.crosstab(df["Class"], df["node.caps"])
print(chisqt)


# In[34]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject HYPOTHESIS') 
else: 
    print('ACCEPT HYPOTHESIS') 


# In[35]:


chisqt = pd.crosstab(df["Menopause"], df["node.caps"])
print(chisqt)


# In[36]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[37]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject ') 
else: 
    print('ACCEPT ') 


# In[38]:


chisqt = pd.crosstab(df["tumor.size"], df["node.caps"])
print(chisqt)


# In[39]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[40]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject ') 
else: 
    print('ACCEPT')


# In[41]:


chisqt = pd.crosstab(df["inv.nodes"], df["node.caps"])
print(chisqt)


# In[42]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[43]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject ') 
else: 
    print('ACCEPT ')


# In[44]:


chisqt = pd.crosstab(df["deg.malig"], df["node.caps"])
print(chisqt)


# In[45]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[46]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject ') 
else: 
    print('ACCEPT ') 


# In[47]:


chisqt = pd.crosstab(df["breast"], df["node.caps"])
print(chisqt)


# In[48]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[49]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject ') 
else: 
    print('ACCEPT ') 


# In[50]:


chisqt = pd.crosstab(df["breast"], df["node.caps"])
print(chisqt)


# In[51]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[52]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject ') 
else: 
    print('ACCEPT ') 


# In[53]:


chisqt = pd.crosstab(df["breast.quad"], df["node.caps"])
print(chisqt)


# In[54]:


stat, p, dof, expected=chi2_contingency(chisqt)


# In[55]:


print("p value: " + str(p)) 
if p <= significance_level: 
    print('Reject ') 
else: 
    print('ACCEPT')


# In[56]:


df["nod.-caps"] = df["node.caps"].map({"no":0, "yes":1})
df.value_counts("node.caps")


# In[57]:


df["breast"] = df["breast"].map({"right":0, "left":1})
df.value_counts("breast")


# In[58]:


df["irradiat"] = df["irradiat"].map({"no":0, "yes":1})
df.value_counts("irradiat")


# In[59]:


df["Class"] = df["Class"].map({"no-recurrence-events":0, "recurrence-events":1})
df.value_counts("Class")


# In[60]:


df["breast.quad"] = df["breast.quad"].map({"right.up":0, "left-up":1, "right-low":2, "left-low":3, "central":4})
df.value_counts("breast.quad")


# In[61]:


df["tumor.size"] = df["tumor.size"].map({"30-34": 32 , "25-29":27, "20-24":22, "15-19":17, "10-14":12, "40-44":42, "35-39":37, "0-4":2, "50-54":52, "5-9":7, "45-49":47})
df.value_counts("tumor.size")


# In[62]:


df["Menopause"] = df["Menopause"].map({"premeno": 0 , "ge40":50, "lt40":30 })
df.value_counts("Menopause")


# In[63]:


df["node.caps"] = df["node.caps"].map({"no":0, "yes":1})
df.value_counts("node.caps")


# In[64]:


df["Age"] = df["Age"].map({"50-59": 54.5 , "40-49":44.5, "60-69":64.5, "30-39":34.5, "70-79":74.5, "20-29":24.5 })
df.value_counts("Age")


# In[65]:


df["inv.nodes"] = df["inv.nodes"].map({"0-2": 1 , "3-5":4, "6-8":7,"9-11":10,"15-17":16, "12-14":13,"24-26":25 })
df.value_counts("inv.nodes")


# In[66]:


df=df.dropna(how='all')

# Reset index after drop
df=df.dropna().reset_index(drop=True)


# In[67]:


df


# In[68]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics


# In[69]:


x = np.array(df[["Class", 
                   "Age" , 
                   "Menopause" , 
                   "tumor.size" , 
                   "inv.nodes" , 
                   "deg.malig" , 
                   "breast" , 
                   "breast.quad" , 
                   "irradiat"]])
y = np.array(df["node.caps"])


# In[70]:


print(x)


# In[71]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics 


# In[72]:


df


# In[73]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[74]:


from keras.utils.np_utils import to_categorical


# In[75]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=0)


# In[76]:


test, train = train_test_split(df, test_size = 0.8)


# In[77]:


classes = 2
batch_size = 2
input_shape = (9,)


# In[78]:


ytrain = to_categorical(ytrain, classes)
ytest = to_categorical(ytest, classes)


# In[79]:


classifier = Sequential()


# In[80]:


classifier.add(Dense(128, activation = 'relu', input_shape = input_shape))
classifier.add(Dropout(.2))
classifier.add(Dense(classes, activation = 'softmax'))


# In[81]:


classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[82]:


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(name='precision'),
     keras.metrics.Recall(name='recall'),])
   # Fitting the ANN to the Training set


# In[83]:


history=classifier.fit(xtrain, ytrain,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              validation_data=(xtest, ytest))


# In[84]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[85]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[86]:


# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Precision Curve')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[87]:


# summarize history for precision
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Recall Curve')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[88]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=0)


# In[89]:


#Tree = DecisionTreeClassifier()
modelDTree = DecisionTreeClassifier(criterion="entropy",
                                     random_state=0)


# In[90]:


from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
      ''


# In[91]:


sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)


# In[92]:


from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()


# In[93]:


modelNB.fit(xtrain, ytrain)


# In[94]:


from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }


# In[95]:


modelNB_result = cross_validation(modelNB, x, y, 3)
print(modelNB_result)


# In[96]:


modelDT = DecisionTreeClassifier(criterion="entropy",
                                     random_state=0)


# In[97]:


modelDT.fit(xtrain,ytrain)


# In[98]:


ypred = modelDT.predict(xtest)
probas1_ = modelDT.fit(xtrain, ytrain).predict_proba(xtest)
ypred 


# In[99]:


def plot_result(x_label, y_label, plot_title, train_data, val_data):
        
        
        # Set size of plot
        plt.figure(figsize=(10,6))
        labels = ["1st ", "2nd ", "3rd ", "4th ", "5th "]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='black', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='purple', label='Validation')
        plt.title(plot_title, fontsize=20)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()


# In[100]:


modelDecisionT_result = cross_validation(modelDT, x, y, 5)
print(modelDecisionT_result)


# In[101]:


model_name = "Decision Tree"
plot_result(model_name,
                "Accuracy",
                "Accuracy scores in 5 Folds",
                modelDecisionT_result["Training Accuracy scores"],
                modelDecisionT_result["Validation Accuracy scores"])


# In[102]:


import seaborn as sns
cm = confusion_matrix(ytest, ypred)
sns.heatmap(cm,annot=True)
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('h.png')


# In[103]:


from sklearn.metrics import classification_report

# printing the report
print(classification_report(ytest, ypred))


# In[104]:


modelDT.fit(xtrain, ytrain)
print(f"Decision tree training set accuracy: {format(modelDT.score(xtrain, ytrain), '.4f')} ")
print(f"Decision tree testing set accuracy: {format(modelDT.score(xtest, ytest), '.4f')} ")


# In[105]:


modelNB.fit(xtrain, ytrain)
print(f"GaussianNB training set accuracy: {format(modelNB.score(xtrain, ytrain), '.4f')} ")
print(f"GaussianNB testing set accuracy: {format(modelNB.score(xtest, ytest), '.4f')} ")


# In[ ]:




