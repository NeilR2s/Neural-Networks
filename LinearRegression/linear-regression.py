import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

'''
x = [21, 15, 10, 16, 18, 14, 16, 20, 19, 13]
y = [95, 86, 84, 88, 90, 75, 89, 91, 84, 83]
'''

class LinearRegression:
    def calculate(X_train, y_train):
        x_mean = sum(X_train)/len(X_train)
        y_mean = sum(y_train)/len(y_train)       
        x_deviation = [(x - x_mean) for x in X_train]
        y_deviation = [(y - y_mean) for y in y_train]
        x_sqr_sum = sum([((x - x_mean)**2) for x in X_train])
        y_sqr_sum = sum([((y - y_mean)**2) for y in y_train])
        product_sum = sum([(x*y) for x,y in zip(x_deviation, y_deviation)])
        r = product_sum / (math.sqrt(x_sqr_sum * y_sqr_sum))
        Sy = math.sqrt(y_sqr_sum/(len(y_train) - 1))
        Sx = math.sqrt(x_sqr_sum/(len(X_train) -1))
        slope = r*(Sy/Sx)
        y_intercept = y_mean - slope * x_mean 
        return {
            "x_mean": x_mean,
            "y_mean": y_mean,
            "slope": slope,
            "y_intercept": y_intercept,
            "correlation_coefficient": r
        }
    
df = pd.read_csv('housing.csv')


model = LinearRegression
df = df[['median_income', 'median_house_value']]
df.drop(df[df['median_house_value']>500000].index, inplace=True)
df.plot.scatter('median_income', 'median_house_value')
X = df.iloc[:, 0].values. reshape(-1,1)
y = df. iloc[:, 1]. values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(model.calculate(X_train, y_train))

