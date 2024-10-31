import numpy as np
import pandas as pd # For data cleaning
from sklearn.model_selection import train_test_split  # Data split 
from sklearn.linear_model import LogisticRegression  #
from sklearn.metrics import accuracy_score # For checking model accuracy
import streamlit as st

credited_card_df = pd.read_csv('creditcard.csv')
credited_card_df.head()


legit = credited_card_df[credited_card_df.Class == 0]
fraud = credited_card_df[credited_card_df.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

#split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

#train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

#Web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter All Required Features Values')
input_df_splited = input_df.split(',')

submit = st.button("Submit")

if submit:
    features = np.asarray(input_df_splited,dtype=np.float64)
    prediction = model.predict(features.reshape(1, -1))

    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fradulant Transaction")    

