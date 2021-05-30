import pandas

ds = pandas.read_csv("SalaryData.csv")
x = ds['YearsExperience'].values.reshape(-1,1)
y = ds['Salary']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
print(model.coef_)
print(model.intercept_)
print(model.predict([[1.1]]))
print(model.predict([[1.1]])/39343)

import joblib

joblib.dump(model, "salary_model.pk1")
