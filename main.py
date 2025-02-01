from tkinter import messagebox
from tkinter import *
import pandas as pd
import csv
import data_training
from data_training import predict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Function to submit values for loan application
def submitClick():
    df = pd.read_csv("submission.csv")
    gender = genderVar.get()
    married = marriedVar.get()
    dependents = dependentsVar.get()
    education = educationVar.get()
    selfEmployed = selfEmployedVar.get()
    income = incomeSlider.get()
    coIncome = coIncomeSlider.get()
    loanAmount = loanAmountSlider.get()
    loanTerm = loanTermVar.get()
    credit = creditVar.get()
    area = propertyAreaVar.get()
    new_row = {'Gender': gender, 'Married': married, 'Dependents': dependents, 'Education': education,
               'Self_Employed': selfEmployed, 'ApplicantIncome': income, 'CoapplicantIncome': coIncome,
               'LoanAmount': loanAmount, 'Loan_Amount_Term': loanTerm, 'Credit_History': credit, 'Property_Area': area}
    df.loc[len(df)] = new_row
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Total_Income_log'] = np.log(df['Total_Income'])
    df = df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)
    result = data_training.predict(df)
    print(df)
    df = data_training.convert(df)
    loanInfo = df.iloc[0]


    messagebox.showinfo("Application Result", message=f'Gender: {df.loc[0]['Gender']}\n'
                                                      f'Married: {df.loc[0]['Married']}\n'
                                                      f'Dependents: {df.loc[0]['Dependents']}\n'
                                                      f'Education: {df.loc[0]['Education']}\n'
                                                      f'Self Employed: {df.loc[0]['Self_Employed']}\n'
                                                      f'Total Income: {df.loc[0]['Total_Income']}\n'
                                                      f'Loan Amount: {df.loc[0]['LoanAmount']},000\n'
                                                      f'Loan Term: {df.loc[0]['Loan_Amount_Term']}\n'
                                                      f'Credit Score: {df.loc[0]['Credit_History']}\n'
                                                      f'Property Area: {df.loc[0]['Property_Area']}\n'
                                                      f'Loan Approved: {result}')

# Functions for pop-up windows with visualizations
def correlationClick():
    data = data_training.preprocessData()
    newWindow = Toplevel(root)
    newWindow.title("Correlation Matrix")
    fig = Figure(figsize=(10, 10), dpi=100)
    fig_canvas = FigureCanvasTkAgg(fig, newWindow)
    ax = fig.add_subplot(111)
    ax.figure.subplots_adjust(left=0.25, bottom=0.25)
    sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f',linewidths=2,annot=True, ax=ax)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(side=TOP, expand=1)
    closeButton=Button(newWindow, text="Close", command=newWindow.destroy)
    closeButton.pack()

def outlierClick():
    data = pd.read_csv("loan_sanction_train.csv")
    newWindow = Toplevel(root)
    newWindow.title("Outlier Detection")
    outlierData = pd.DataFrame(data, columns=['ApplicantIncome', 'CoapplicantIncome'])
    fig = Figure(figsize=(10, 10), dpi=100)
    fig_canvas = FigureCanvasTkAgg(fig, newWindow)
    plot = fig.add_subplot(111)
    plot.boxplot(outlierData, tick_labels=['Applicant Income', 'Coapplicant Income'])
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(side=TOP, expand=1)
    closeButton=Button(newWindow, text="Close", command=newWindow.destroy)
    closeButton.pack()

def feature_importanceClick():
    data = data_training.preprocessData()
    newWindow = Toplevel(root)
    newWindow.title("Feature Importance")
    X = data.drop('Loan_Status', axis=1)
    y = data.Loan_Status
    model = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    importance.plot(kind='barh', ax=ax)
    ax.figure.subplots_adjust(left=0.25)
    canvas = FigureCanvasTkAgg(fig, newWindow)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, expand=1)
    closeButton=Button(newWindow, text="Close", command=newWindow.destroy)
    closeButton.pack()


# Create GUI
root = Tk()
root.title('Goliath National Bank')

headLabel = Label(root, text="Loan Approval Prediction")
headLabel.config(anchor=CENTER)
headLabel.grid(row=0, columnspan=4)

genderLabel = Label(root, text="Gender")
genderLabel.grid(row=1, column=0, sticky=E)
genderVar = IntVar()
g1 = Radiobutton(root, text="Male", variable=genderVar, value=1)
g1.grid(row=1,column=1)
g2 = Radiobutton(root, text="Female", variable=genderVar, value=0)
g2.grid(row=1,column=2)
marriedLabel = Label(root, text="Marital Status")
marriedLabel.grid(row=2, column=0, sticky=E)
marriedVar = IntVar()
m1 = Radiobutton(root, text="Single", variable=marriedVar, value=0)
m1.grid(row=2, column=1)
m2 = Radiobutton(root, text="Married", variable=marriedVar, value=1)
m2.grid(row=2, column=2)
dependentsLabel = Label(root, text="Number of Dependents")
dependentsLabel.grid(row=3, column=0, sticky=E)
dependentsVar = IntVar()
dependentsMenu = OptionMenu(root, dependentsVar,0, 1,2,3)
dependentsMenu.grid(row=3, column=1)
educationLabel = Label(root, text="College Education")
educationLabel.grid(row=4, column=0, sticky=E)
educationVar = IntVar()
e1 = Radiobutton(root, text="Graduated", variable=educationVar, value=0)
e1.grid(row=4, column=1)
e2 = Radiobutton(root, text="Not Graduated", variable=educationVar, value=1)
e2.grid(row=4, column=2)
selfEmployedLabel = Label(root, text="Self Employed")
selfEmployedLabel.grid(row=5, column=0, sticky=E)
selfEmployedVar = IntVar()
s1 = Radiobutton(root, text="No", variable=selfEmployedVar, value=0)
s1.grid(row=5, column=1)
s2 = Radiobutton(root, text="Yes", variable=selfEmployedVar, value=1)
s2.grid(row=5, column=2)
incomeLabel = Label(root, text="Monthly Income")
incomeLabel.grid(row=6, column=0, sticky=E)
incomeSlider = Scale(root, from_=0, to=25000, length=400,orient=HORIZONTAL)
incomeSlider.grid(row=6, column=1, columnspan=3)
coIncomeLabel = Label(root, text="Co-Applicant Income")
coIncomeLabel.grid(row=7, column=0, sticky=E)
coIncomeSlider = Scale(root, from_=0, to=25000, length=400,orient=HORIZONTAL)
coIncomeSlider.grid(row=7, column=1, columnspan=4)
loanAmountLabel = Label(root, text="Loan Amount in Thousands")
loanAmountLabel.grid(row=8, column=0, sticky=E)
loanAmountSlider = Scale(root, from_=0, to=700, length=400,orient=HORIZONTAL)
loanAmountSlider.grid(row=8, column=1, columnspan=4)
loanTermLabel = Label(root, text="Loan Term")
loanTermLabel.grid(row=9, column=0, sticky=E)
loanTermVar = IntVar()
loanTermVar.set(360)
loanTermMenu = OptionMenu(root, loanTermVar, 120, 240, 360)
loanTermMenu.grid(row=9, column=1)
creditHistoryLabel = Label(root, text="Credit Score")
creditHistoryLabel.grid(row=10, column=0, sticky=E)
creditVar = IntVar()
c1 = Radiobutton(root, text="300 - 669", variable=creditVar, value=0)
c1.grid(row=10, column=1)
c2 = Radiobutton(root, text="670 - 850", variable=creditVar, value=1)
c2.grid(row=10, column=2)
propertyAreaLabel = Label(root, text="Property Area")
propertyAreaLabel.grid(row=11, column=0, sticky=E)
propertyAreaVar = IntVar()
p1 = Radiobutton(root, text="Rural", variable=propertyAreaVar, value=0)
p1.grid(row=11, column=1)
p2 = Radiobutton(root, text="Semiurban", variable=propertyAreaVar, value=1)
p2.grid(row=11, column=2)
p3 = Radiobutton(root, text="Urban", variable=propertyAreaVar, value=2)
p3.grid(row=11, column=3)
submitButton = Button(root, text="Submit", command=submitClick)
submitButton.config(anchor=CENTER)
submitButton.grid(row=12, columnspan=4)
correlationButton = Button(root, text="Correlation Matrix", command=correlationClick)
correlationButton.grid(row=15, column=0)
outlierButton = Button(root, text="Outliers", command=outlierClick)
outlierButton.grid(row=15, column=1)
importanceButton = Button(root, text="Feature Importance", command=feature_importanceClick)
importanceButton.grid(row=15, column=2)

dataLabel = Label(root, text="Data Visualizations")
dataLabel.config(anchor=CENTER)
dataLabel.grid(row=13, columnspan=4)

quitButton = Button(root, text="Quit", command=root.destroy)
quitButton.config(anchor=CENTER)
quitButton.grid(row=16, columnspan=4)



root.mainloop()
