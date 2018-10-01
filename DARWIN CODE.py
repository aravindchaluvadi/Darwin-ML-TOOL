# importing required packages
import os , datetime
from tkinter import *
from tkinter import filedialog
import os, sys
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix,roc_curve,auc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from pandas import Series,DataFrame
import random 
from random import sample
import pandas as pd
# setting the path of working directory
root =Tk()
path = filedialog.askdirectory()
root.destroy()
os.chdir(path)

#creating a dirctory for darwin with timestamp
if not os.path.exists(path+"/Darwin"+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')):
    os.makedirs(path+"/Darwin"+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
os.chdir(path+"/Darwin"+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
path_Darwin =os.getcwd()

#selecting the required file
root = Tk()
df1_original = pd.read_csv(filedialog.askopenfile(),na_values=[""," ","?","#","NaN"])
root.destroy()

#specifying the target column by the user 
target = input("Enter Target Column name:")

#renaming the target column
df1_original= df1_original.rename(columns={target: 'target'})
cols=df1_original.columns

#selecting the categorical variables by the user
import tkinter as tk
cat_vars=[]
class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        label1 = tk.Label(self, text="Double-Click all Categorical Variables")
        label1.pack(side=tk.TOP)
        lb = tk.Listbox(self,height=30, width=50)
        
        for item in cols:
            lb.insert("end", item)
            lb.bind("<Double-Button-1>", self.OnDouble)
            lb.pack(side="top", fill="both", expand=True)
       

    def OnDouble(self, event):
        widget = event.widget
        selection=widget.curselection()
        value = widget.get(selection[0])
        cat_vars.append(value)
       
        print("selection:", selection, ": '%s'" % value)

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()

#removing the duplicates selected by the user    
cat_vars = list(set(cat_vars) & set(cols)) 
print(cat_vars)

#changing the sleceted variables into categorical datatype
for col in cat_vars:
    df1_original[col]=df1_original[col].astype('category')
    

#cat_vars = list(set(cat_vars) & set(cols))    


# creating directory for layer 1
if not os.path.exists(path_Darwin+"/Layer_1"):
    os.makedirs(path_Darwin+"/Layer_1")
os.chdir(path_Darwin+"/Layer_1")
path_layer_1 =os.getcwd()


#NA's removal

if  df1_original.isnull().any().any() == True:
    by=input("Type of NA removal --- row or column") # user input to select the type of removal
    if by=='column':
            
################ clumns wise removal
        pct_null = df1_original.isnull().sum() / len(df1_original)
        missing_features = pct_null[pct_null > 0.30].index
        df1_original.drop(missing_features,axis=1,inplace=True)
        df1_original.to_csv("nacolumnremoved..csv",index=False)
    else:
        
#########removing rrow wise na
        pct_null = df1_original.isnull().sum() / len(df1_original)
        missing_features = pct_null[pct_null > 0.30].index
        df1_original.drop(missing_features,axis=0,inplace=True)
        df1_original.to_csv("narowremoved..csv",index=False)
else:
    df1_original.to_csv("nonapresent..csv",index=False)  #writing the original file which donot have NA's
    missing_features = []
    
    file_names = []
for root, dirs, files in os.walk(path_layer_1):  
    for filename in files:
        file_names.append(filename)
#print(file_names)

# creating directory for layer 2
if not os.path.exists(path_Darwin+"/Layer_2"):
    os.makedirs(path_Darwin+"/Layer_2")
path_layer_2 = path_Darwin+"/Layer_2"

os.getcwd()

# removing the columns dropped in the layer 1 from the catergory list
col0= list(set(cat_vars) - set(missing_features))

# Imputation for the missing values
for file in file_names:
    print(file)
    df1_layer2_original = pd.read_csv(path_layer_1+"/"+file)
    for col in col0:
        df1_layer2_original[col]=df1_layer2_original[col].astype('category')
    if  df1_layer2_original.isnull().any().any() == True: 
        #central impputation
        df1_layer2_original[df1_layer2_original.select_dtypes(['category']).columns] = df1_layer2_original.select_dtypes(['category']).apply(lambda x:x.fillna(x.value_counts().index[0]))
        df1_layer2_original[df1_layer2_original.select_dtypes(['int64','float64']).columns] = df1_layer2_original.select_dtypes(['int64','float64']).apply(lambda x:x.fillna(x.mean()))
        df1_layer2_original.to_csv(path_layer_2+"/"+"centralImpute.{}".format(file),index=False)
        #NA omit
        df1_layer2_omitna =df1_layer2_original.dropna(how='any')
        df1_layer2_omitna.to_csv(path_layer_2+"/"+"naomit.{}".format(file),index=False)
    else:
        df1_layer2_original.to_csv(path_layer_2+"/"+"original.{}".format(file),index=False) # writing the original file which donot have NA's
        
#layer 3

#reading the files from layer 2 directory
file_names = []
for root, dirs, files in os.walk(path_layer_2):  
    for filename in files:
        file_names.append(filename)
#print(file_names)

# creating directory for layer 3
if not os.path.exists(path_Darwin+"/Layer_3"):
    os.makedirs(path_Darwin+"/Layer_3")
path_layer_3 = path_Darwin+"/Layer_3"

from sklearn import preprocessing 

# Normalizing the data by Z-Score and range
for file in file_names:
    print(file)
    df1_layer3_original = pd.read_csv(path_layer_2+"/"+file)
    for col in col0:
        df1_layer3_original[col]=df1_layer3_original[col].astype('category')
    #Normalizing the data by Z-Score    
    df1_layer3_scale =df1_layer3_original.select_dtypes(['int64','float64']).apply(lambda x:((x)-x.mean())/(x.std()))
    df1_layer3_scale_merge = pd.merge(df1_layer3_scale,df1_layer3_original.select_dtypes(['category']), left_index=True, right_index=True)
    df1_layer3_scale_merge.to_csv(path_layer_3+"/"+"scaled.{}".format(file),index=False)
    
    #Normalizing the data by range
    df1_layer3_range =df1_layer3_original.select_dtypes(['int64','float64']).apply(lambda x:((x)-min(x))/((max(x))-(min(x))))
    df1_layer3_range_merge = pd.merge(df1_layer3_range,df1_layer3_original.select_dtypes(['category']), left_index=True, right_index=True)
    df1_layer3_range_merge.to_csv(path_layer_3+"/"+"range.{}".format(file),index=False)
    
    df1_layer3_original.to_csv(path_layer_3+"/"+"notscaled.{}".format(file),index=False) #writing the not normalized data
#LAYER 4

#reading the files from layer 3 directory
file_names = []
for root, dirs, files in os.walk(path_layer_3):  
    for filename in files:
        file_names.append(filename)
#print(file_names)

# creating directory for layer 4
if not os.path.exists(path_Darwin+"/Layer_4"):
    os.makedirs(path_Darwin+"/Layer_4")
path_layer_4 = path_Darwin+"/Layer_4"

# creating an empty list 
col_drop=[]

#dropping the columns with more than 10 levels or having 1 level
for file in file_names:
    print(file)
    df1_layer4_original = pd.read_csv(path_layer_3+"/"+file)
    for col in col0:
        df1_layer4_original[col]=df1_layer4_original[col].astype('category')
    uni_col = df1_layer4_original.select_dtypes(['category']).nunique()
    df1_layer4_keypair=uni_col.to_dict()
    df1_layer4_keypair.items()
    for values in df1_layer4_keypair.items():
        if(values[1]>10 or values[1]==1):
            #print(values[0])
            col_drop.append(values[0])  # appending the columns with levels more than threshold or only one level
    df1_layer4_original.drop(col_drop,axis=1,inplace=True)
    df1_layer4_original.to_csv(path_layer_4+"/"+"factorcontrolled.{}".format(file),index=False)

#LAYER 5

#reading the files from layer 4 directory
file_names = []
for root, dirs, files in os.walk(path_layer_4):  
    for filename in files:
        file_names.append(filename)
# print(file_names)

# creating directory for layer 5
if not os.path.exists(path_Darwin+"/Layer_5"):
    os.makedirs(path_Darwin+"/Layer_5")
path_layer_5 = path_Darwin+"/Layer_5"

#removing the columns dropped in the layer 4 from the catergory list
cat1=list(set(col0) - set(col_drop))
print(cat1)

# setting the threshold by user
ci_threshold = int(input("Enter the Class Imbalance threshold ")) 

#handeling class imbalance
for file in file_names:
    print(file)
    df1_layer5_original = pd.read_csv(path_layer_4+"/"+file)
    for col in cat1:
        df1_layer5_original[col]=df1_layer5_original[col].astype('category')
    df1_layer5_original.to_csv(path_layer_5+"/"+"classimbalancenothandled.{}".format(file),index=False)
    uni =df1_layer5_original.target.nunique() # number of uniques in the target column
    #print(uni)
    if(ci_threshold==0):
        ci_threshold = math.trunc((100 /uni)/4) #setting the threshold if user has not specified
    t= pd.crosstab(index = df1_layer5_original.target,
                             columns="count")
    dist=(t/t.sum())*100
    dist['count']
    #print(type(dist))
    ci_check=dist['count' ]< ci_threshold
#     print(type(ci_check))
#     print(len(ci_check))
    count = 0
    for i in ci_check:
        if i == True:
            count = count + 1
    if(count>0):
        target_count = df1_layer5_original.target.value_counts()
        #print('Class',i, target_count[i])
        req_rows=math.trunc((1 /uni) * df1_layer5_original.shape[0])
        n=df1_layer5_original.target.unique()
        for i in range(0,(uni+1)):
            df_class_i =n[1:i]
            count_class_i= df1_layer5_original.target.value_counts()
            df1_layer5_keypair=count_class_i.to_dict()
        count_over_sample=max(df1_layer5_keypair.values())
        print(count_over_sample)
        column=df1_layer5_original.columns
        print(column)
        df_test_over = pd.DataFrame(columns=column)
        df_test_over.head()
        for i in range(0,uni):
            #df_class_i = pd.DataFrame(columns=column)
            class_i=n[1:i]
            df_class_i=df1_layer5_original[df1_layer5_original.target == df1_layer5_original.target.unique()[i]]
            count_class_i= df1_layer5_original.target.value_counts()
            if (count_class_i) is not (count_over_sample):
                #df_class_i_over = df_class_i.sample(n=count_over_sample, replace=True)
                df_class_i_over = df_class_i.sample(count_over_sample, replace=True)
                df_c_i=df_class_i_over.sample(n=req_rows,replace=True)
                df_test_over = df_test_over.append(df_c_i,ignore_index=True)
            df_test_over.append((count_class_i) == (count_over_sample))
            print(df_test_over.head())
            print('Random over-sampling:')
            print(df_test_over.target.value_counts())
            print(df_test_over.shape)
            df_test_over.to_csv(path_layer_5+"/"+"classimbalancehandled.{}".format(file),index=False)
#LAYER 6

from sklearn.preprocessing import LabelEncoder
file_names = []
for root, dirs, files in os.walk(path_layer_5):
    for filename in files:
        file_names.append(filename)
print(file_names)


if not os.path.exists(path_Darwin+"/Layer_6"):
    os.makedirs(path_Darwin+"/Layer_6")
path_layer_6 = path_Darwin+"/Layer_6"

column=['Algoritm','Dataset','l1','l2','l3','l4','l5','Train Accuracy','Test Accuracy','Train Precesion','Test Precesion','Train Recall','Test Recall','Train F1_Score','Test F1_Score']
df_output = pd.DataFrame(columns=column)
df_output.head()

for file in file_names:
    print(file)
    a=file.split(".")
    b=a[:len(a)-1]
    df1_layer6_original = pd.read_csv(path_layer_5+"/"+file)
    for col in cat1:
        df1_layer6_original[col]=df1_layer6_original[col].astype('category')        
    cat_vars = df1_layer6_original.select_dtypes(['category'])
    d = {}
    # Iterate over the categorical variables and convert them to levels
    for i in cat_vars:
        #print(i)
        d[i] = LabelEncoder()
        d[i].fit(df1_layer6_original[i])
        df1_layer6_original[i] = d[i].fit_transform(df1_layer6_original[i])
    #print(df1_layer6_original.head())
    
    # train and test split
    X_df=df1_layer6_original.drop(df1_layer6_original.columns[-1], axis=1)              # Features 
    y_df = df1_layer6_original.target
    X_train,X_test,y_train,y_test = train_test_split(X_df, y_df,random_state=1234)
    
    logreg = LogisticRegression()
    model=logreg.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    confusion_matrix_test = confusion_matrix(y_test, test_pred)
    confusion_matrix_train = confusion_matrix(y_train, train_pred)

    print(confusion_matrix_train)
    print(confusion_matrix_test)

    #score's for metrics for train

    algorithm='logistic'
    Train_Accuracy=accuracy_score(y_train, train_pred)
    Train_Precision=precision_score(y_train, train_pred,average='macro')
    Train_Recall=recall_score(y_train, train_pred,average='macro')
    Train_F1_Score=f1_score(y_train, train_pred,average='macro')
    print('\n',algorithm)
    print("accuracy for train: {}".format(Train_Accuracy))
    print("precision for train: {}".format(Train_Precision))
    print("recall for train: {}".format(Train_Recall))
    print("f1 for train: {}".format(Train_F1_Score))
    
    #score's for metrics for test

    Test_Accuracy=accuracy_score(y_test, test_pred)
    Test_Precision=precision_score(y_test, test_pred,average='macro')
    Test_Recall=recall_score(y_test, test_pred,average='macro')
    Test_F1_Score=f1_score(y_test, test_pred,average='macro')
    print("accuracy for test: {}".format(Test_Accuracy))
    print("precision for test: {}".format(Test_Precision))
    print("recall for test: {}".format(Test_Recall))
    print("f1 for test: {}".format(Test_F1_Score))
    
    
    
    output= pd.DataFrame([[algorithm,'{}'.format(file),b[4],b[3],b[2],b[1],b[0],Train_Accuracy,Test_Accuracy,Train_Precision,Test_Precision,Train_Recall,Test_Recall,Train_F1_Score,Test_F1_Score]],columns=column)
    df_output=df_output.append(output,ignore_index=False)
    
    
    gnb = GaussianNB()
    nbmodel=gnb.fit(X_train, y_train)
    train_pred = nbmodel.predict(X_train)
    test_pred = nbmodel.predict(X_test)
    confusion_matrix_test = confusion_matrix(y_test, test_pred)
    confusion_matrix_train = confusion_matrix(y_train, train_pred)

    print(confusion_matrix_train)
    print(confusion_matrix_test)

    algorithm1='Naive Bayes'
    Train_Accuracy=accuracy_score(y_train, train_pred)
    Train_Precision=precision_score(y_train, train_pred,average='macro')
    Train_Recall=recall_score(y_train, train_pred,average='macro')
    Train_F1_Score=f1_score(y_train, train_pred,average='macro')
    print('\n',algorithm1)
    print("accuracy for train: {}".format(Train_Accuracy))
    print("precision for train: {}".format(Train_Precision))
    print("recall for train: {}".format(Train_Recall))
    print("f1 for train: {}".format(Train_F1_Score))

    #score's for metrics for test

    Test_Accuracy=accuracy_score(y_test, test_pred)
    Test_Precision=precision_score(y_test, test_pred,average='macro')
    Test_Recall=recall_score(y_test, test_pred,average='macro')
    Test_F1_Score=f1_score(y_test, test_pred,average='macro')
    print("accuracy for test: {}".format(Test_Accuracy))
    print("precision for test: {}".format(Test_Precision))
    print("recall for test: {}".format(Test_Recall))
    print("f1 for test: {}".format(Test_F1_Score))
    
   
    
    output1= pd.DataFrame([[algorithm1,'{}'.format(file),b[4],b[3],b[2],b[1],b[0],Train_Accuracy,Test_Accuracy,Train_Precision,Test_Precision,Train_Recall,Test_Recall,Train_F1_Score,Test_F1_Score]],columns=column)
    df_output=df_output.append(output1,ignore_index=False)
    
    #Dcecsion Tree##
    
    param_grid = {'max_depth': np.arange(3, 10)}
    tree= GridSearchCV(DecisionTreeClassifier(), param_grid)
    dtmodel=tree.fit(X_train,y_train)
    train_pred = dtmodel.predict(X_train)
    test_pred = dtmodel.predict(X_test)
    confusion_matrix_test = confusion_matrix(y_test, test_pred)
    confusion_matrix_train = confusion_matrix(y_train, train_pred)

    print(confusion_matrix_train)
    print(confusion_matrix_test)

    algorithm2='Decision Tree'
    #score's for metrics for train
    Train_Accuracy=accuracy_score(y_train, train_pred)
    Train_Precision=precision_score(y_train, train_pred,average='macro')
    Train_Recall=recall_score(y_train, train_pred,average='macro')
    Train_F1_Score=f1_score(y_train, train_pred,average='macro')
    print('\n',algorithm2)
    print("accuracy for train: {}".format(Train_Accuracy))
    print("precision for train: {}".format(Train_Precision))
    print("recall for train: {}".format(Train_Recall))
    print("f1 for train: {}".format(Train_F1_Score))
    
    #score's for metrics for test
    Test_Accuracy=accuracy_score(y_test, test_pred)
    Test_Precision=precision_score(y_test, test_pred,average='macro')
    Test_Recall=recall_score(y_test, test_pred,average='macro')
    Test_F1_Score=f1_score(y_test, test_pred,average='macro')
    print("accuracy for test: {}".format(Test_Accuracy))
    print("precision for test: {}".format(Test_Precision))
    print("recall for test: {}".format(Test_Recall))
    print("f1 for test: {}".format(Test_F1_Score))
    
    
    
    output2= pd.DataFrame([[algorithm2,'{}'.format(file),b[4],b[3],b[2],b[1],b[0],Train_Accuracy,Train_Precision,Train_Recall,Train_F1_Score,Test_Accuracy,Test_Precision,Test_Recall,Test_F1_Score]],columns=column)
    df_output=df_output.append(output2,ignore_index=False)
    
    ### Random Forest ###
    
    rfc = RandomForestClassifier(n_jobs=-1, max_features='log2') 
 
    # Use a grid over parameters of interest
    param_grid = { 
               "n_estimators" : [9, 18, 27, 36, 45, 54, 63],
               "max_depth" : [5, 10],
               "min_samples_leaf" : [2, 4]}

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
    rfmodel=CV_rfc.fit(X=X_train, y=y_train)
    print (rfmodel.best_score_, rfmodel.best_params_) 
    train_pred = rfmodel.predict(X_train)
    test_pred = rfmodel.predict(X_test)
    confusion_matrix_test = confusion_matrix(y_test, test_pred)
    confusion_matrix_train = confusion_matrix(y_train, train_pred)

    print(confusion_matrix_train)
    print(confusion_matrix_test)

    algorithm3='Random Forest'
    #score's for metrics for train
    Train_Accuracy=accuracy_score(y_train, train_pred)
    Train_Precision=precision_score(y_train, train_pred,average='macro')
    Train_Recall=recall_score(y_train, train_pred,average='macro')
    Train_F1_Score=f1_score(y_train, train_pred,average='macro')
    print('\n',algorithm3)
    print("accuracy for train: {}".format(Train_Accuracy))
    print("precision for train: {}".format(Train_Precision))
    print("recall for train: {}".format(Train_Recall))
    print("f1 for train: {}".format(Train_F1_Score))

    #score's for metrics for test
    Test_Accuracy=accuracy_score(y_test, test_pred)
    Test_Precision=precision_score(y_test, test_pred,average='macro')
    Test_Recall=recall_score(y_test, test_pred,average='macro')
    Test_F1_Score=f1_score(y_test, test_pred,average='macro')
    print("accuracy for test: {}".format(Test_Accuracy))
    print("precision for test: {}".format(Test_Precision))
    print("recall for test: {}".format(Test_Recall))
    print("f1 for test: {}".format(Test_F1_Score))
    
    

    output3= pd.DataFrame([[algorithm3,'{}'.format(file),b[4],b[3],b[2],b[1],b[0],Train_Accuracy,Train_Precision,Train_Recall,Train_F1_Score,Test_Accuracy,Test_Precision,Test_Recall,Test_F1_Score]],columns=column)
    df_output=df_output.append(output3,ignore_index=False)
    
    ### svm ###
    
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, n_jobs=4)
    svm_model =clf.fit(X=X_train, y=y_train)
    print (svm_model.best_score_,svm_model.best_params_) 
    train_pred = svm_model.predict(X_train)
    test_pred = svm_model.predict(X_test)
    confusion_matrix_test = confusion_matrix(y_test, test_pred)
    confusion_matrix_train = confusion_matrix(y_train, train_pred)
    
    algorithm4='svm'
    
    print(confusion_matrix_train)
    print(confusion_matrix_test)
    
    Train_Accuracy=accuracy_score(y_train, train_pred)
    Train_Precision=precision_score(y_train, train_pred,average='macro')
    Train_Recall=recall_score(y_train, train_pred,average='macro')
    Train_F1_Score=f1_score(y_train, train_pred,average='macro')
    print('\n',algorithm4)
    print("accuracy for train: {}".format(Train_Accuracy))
    print("precision for train: {}".format(Train_Precision))
    print("recall for train: {}".format(Train_Recall))
    print("f1 for train: {}".format(Train_F1_Score))

    #score's for metrics for test

    Test_Accuracy=accuracy_score(y_test, test_pred)
    Test_Precision=precision_score(y_test, test_pred,average='macro')
    Test_Recall=recall_score(y_test, test_pred,average='macro')
    Test_F1_Score=f1_score(y_test, test_pred,average='macro')
    print("accuracy for test: {}".format(Test_Accuracy))
    print("precision for test: {}".format(Test_Precision))
    print("recall for test: {}".format(Test_Recall))
    print("f1 for test: {}".format(Test_F1_Score))
    
   
    
    output4= pd.DataFrame([[algorithm4,'{}'.format(file),b[4],b[3],b[2],b[1],b[0],Train_Accuracy,Train_Precision,Train_Recall,Train_F1_Score,Test_Accuracy,Test_Precision,Test_Recall,Test_F1_Score]],columns=column)
    df_output=df_output.append(output4,ignore_index=False)
    
    
    ###KNN##
    from sklearn.neighbors import KNeighborsClassifier
    parameters= {'n_neighbors': np.arange(3, 11)}
    knn = KNeighborsClassifier(algorithm = 'brute')
    knn_mod = GridSearchCV(knn, parameters)
     
    knnmodel=knn_mod.fit(X_train, y_train)
   
    print(knn_mod.fit(X_train, y_train).best_params_.items())
   
    train_pred = knnmodel.predict(X_train)
    test_pred = knnmodel.predict(X_test)
    confusion_matrix_test = confusion_matrix(y_test, test_pred)
    confusion_matrix_train = confusion_matrix(y_train, train_pred)

    print(confusion_matrix_train)
    print(confusion_matrix_test)

    algorithm5='KNN'
    #score's for metrics for train
    Train_Accuracy=accuracy_score(y_train, train_pred)
    Train_Precision=precision_score(y_train, train_pred,average='macro')
    Train_Recall=recall_score(y_train, train_pred,average='macro')
    Train_F1_Score=f1_score(y_train, train_pred,average='macro')
    print('\n',algorithm5)
    print("accuracy for train: {}".format(Train_Accuracy))
    print("precision for train: {}".format(Train_Precision))
    print("recall for train: {}".format(Train_Recall))
    print("f1 for train: {}".format(Train_F1_Score))
    
    #score's for metrics for test
    Test_Accuracy=accuracy_score(y_test, test_pred)
    Test_Precision=precision_score(y_test, test_pred,average='macro')
    Test_Recall=recall_score(y_test, test_pred,average='macro')
    Test_F1_Score=f1_score(y_test, test_pred,average='macro')
    print("accuracy for test: {}".format(Test_Accuracy))
    print("precision for test: {}".format(Test_Precision))
    print("recall for test: {}".format(Test_Recall))
    print("f1 for test: {}".format(Test_F1_Score))
    
    
    output5= pd.DataFrame([[algorithm5,'{}'.format(file),b[4],b[3],b[2],b[1],b[0],Train_Accuracy,Train_Precision,Train_Recall,Train_F1_Score,Test_Accuracy,Test_Precision,Test_Recall,Test_F1_Score]],columns=column)
    df_output=df_output.append(output5,ignore_index=False)

df_output.to_csv(path_layer_6+"/"+"output.csv",index=False)
