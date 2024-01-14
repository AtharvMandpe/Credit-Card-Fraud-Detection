import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv("C:/Users/athar/Desktop/dataset/creditcard.csv")

# print(credit_card_data.head())
# print(credit_card_data.tail())

# print(credit_card_data.info())

# #missing values in each col
# print(credit_card_data.isnull().sum())

# #no of fraud and legit transac
# print(credit_card_data['Class'].value_counts())

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# print(legit.shape)
# print(legit.Amount.describe())
# print(fraud.Amount.describe())

#building sample dataset
legit_sample = legit.sample(n=492)

#concat 2 dataframes
new_dataset = pd.concat([legit_sample, fraud], axis=0)

#print(new_dataset['Class'].value_counts())

#splitting into targets and features
x = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify=y, random_state=2)

#print(x.shape, x_train.shape, x_test.shape)

#model
model = LogisticRegression()

model.fit(x_train, y_train)
#print("done training")

#accuracy
x_train_pred = model.predict(x_train)
training_accuracy = accuracy_score(x_train_pred, y_train)
print("Accuracy on training data : ", training_accuracy)

x_test_pred = model.predict(x_test)
testing_accuracy = accuracy_score(x_test_pred, y_test)
print("Accuracy on testing data : ", testing_accuracy)
