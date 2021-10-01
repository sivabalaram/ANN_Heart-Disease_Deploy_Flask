
from flask import Flask, render_template, request
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


df=pd.read_csv('Churn_Modelling.csv')
df.drop(['RowNumber','CustomerId','Surname','Geography','Gender'],inplace=True,axis=1)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#Encoding categorical data
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#X[:, 2] = le.fit_transform(X[:, 2])
#print(X)

#one hot encoder for "geography" colum
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
#print(X)

sc=StandardScaler()
X=sc.fit_transform(X)



# app.py
 

#import pickle
#from sklearn.linear_model import LogisticRegression
 
model=tf.keras.models.load_model('ANN_Model.h5')
app = Flask(__name__)
 
 
@app.route('/')
def index():
    return render_template('index.html')
 
 
 
@app.route('/predict',methods=['POST'])
def predict():
 
    if request.method == 'POST':
 
        CreditScore = request.form['CreditScore']
        #Geography = request.form['Geography']
        #Gender= request.form['Gender']
        Age = request.form['Age']
        Tenure = request.form['Tenure']
        Balance = request.form['Balance']
        NumOfProducts = request.form['NumOfProducts']
        HasCrCard = request.form['HasCrCard']
        IsActiveMember = request.form['IsActiveMember']
        EstimatedSalary = request.form['EstimatedSalary']

    x=model.predict(sc.transform([[CreditScore, Age,Tenure,Balance , NumOfProducts,HasCrCard, IsActiveMember,EstimatedSalary]]))
    
    if x>0.50:
        x=1
    else:
        x=0
       
    return render_template('sub.html',prediction=x)

 
 
 
if __name__ == '__main__':
    app.run(debug=True)