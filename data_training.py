import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def preprocessData():
    # Read in CSV with loan data
    data = pd.read_csv("loan_sanction_train.csv")
    # Fill in missing data
    data.fillna({'Gender': data['Gender'].mode()[0]}, inplace=True)
    data.fillna({'Married': data['Married'].mode()[0]}, inplace=True)
    data.fillna({'Dependents': data['Dependents'].mode()[0]}, inplace=True)
    data.fillna({'Self_Employed': data['Self_Employed'].mode()[0]}, inplace=True)
    data.fillna({'Credit_History': data['Credit_History'].mode()[0]}, inplace=True)
    data.fillna({'Loan_Amount_Term': data['Loan_Amount_Term'].mode()[0]}, inplace=True)
    data.fillna({'LoanAmount': data['LoanAmount'].median()}, inplace=True)
    data.replace({'Dependents':'3+'}, '3', inplace=True)
    # Combine income fields into one variable
    data['Total_Income']=data['ApplicantIncome']+data['CoapplicantIncome']
    # Reduce impact of outliers with logarithmic transformation
    data['Total_Income_log'] = np.log(data['Total_Income'])
    data = data.drop(['ApplicantIncome','CoapplicantIncome'], axis=1)
    # Remove Loan ID
    data = data.drop('Loan_ID', axis=1)
    # Label encoder transforms non-numerical data
    label_encoder = preprocessing.LabelEncoder()
    obj = (data.dtypes == 'object')
    for col in list(obj[obj].index):
        data[col] = label_encoder.fit_transform(data[col])

    return data

def predict(dataFrame):
    data = preprocessData()

    #Separate target variable
    X = data.drop('Loan_Status', axis=1)
    y = data.Loan_Status

    x_train,x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)
    model.fit(x_train, y_train)
    Y_pred = model.predict(dataFrame)
    if Y_pred == 1:
        return 'Yes'
    if Y_pred == 0:
        return 'No'

# Converts submission values into non-numerical values for loan results screen
def convert(dataFrame):
    dataFrame.replace({'Gender':1}, 'Male', inplace=True)
    dataFrame.replace({'Gender':0}, 'Female', inplace=True)
    dataFrame.replace({'Married':0}, 'No', inplace=True)
    dataFrame.replace({'Married':1}, 'Yes', inplace=True)
    dataFrame.replace({'Dependents':3}, '3+', inplace=True)
    dataFrame.replace({'Education': 0}, 'Graduated', inplace=True)
    dataFrame.replace({'Education':1}, 'Not Graduated', inplace=True)
    dataFrame.replace({'Self_Employed':0}, 'No', inplace=True)
    dataFrame.replace({'Self_Employed':1}, 'Yes', inplace=True)
    dataFrame.replace({'Credit_History':0}, '300-669', inplace=True)
    dataFrame.replace({'Credit_History':1}, '670-850', inplace=True)
    dataFrame.replace({'Property_Area':0}, 'Rural', inplace=True)
    dataFrame.replace({'Property_Area':1}, 'Semiurban', inplace=True)
    dataFrame.replace({'Property_Area':2}, 'Urban', inplace=True)
    return dataFrame

# Function to test models for final selection
def testModel():
    data = preprocessData()
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rfc = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)
    knn = KNeighborsClassifier(n_neighbors=3)
    svc = SVC(kernel='linear')
    lr = LogisticRegression(max_iter=10000)


    for model in [rfc, knn, svc, lr]:
            model.fit(x_train, y_train)
            Y_pred = model.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, Y_pred)
            print('Accuracy of', model.__class__.__name__,':', 100*accuracy)

if __name__ == '__main__':
    testModel()



