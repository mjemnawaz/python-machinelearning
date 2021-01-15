
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# MULTIPLE REGRESSION is like linear regression, except with multiple independent variables
# 	for example, we can predict the speed of a car based on the time if day as the independent variable
#	or, to be more accurate, we can take into account other independent variables too such as the weight of the car

# importing a csv file listing cars with their brand, model, volume, weight, and CO2 emissions
df = pandas.read_csv("cars.csv")

# collect the relevant dependent and independent variables from the pandas dataframe
DEPENDENT_VAR = df['CO2']
INDEPENDENT_VARS = df[['Weight', 'Volume']]

# get data of the relationship between the dependent variable and the multiple independent variables
REGRESSION_OBJ = linear_model.LinearRegression()
REGRESSION_OBJ.fit(INDEPENDENT_VARS, DEPENDENT_VAR)


# we can use the regression object to accurately predict the CO2 emissions given the weight and volume
print("If the weight is 2300 and the volume is 1300, the CO2 will be {}.".format(REGRESSION_OBJ.predict([[2300,1300]])[0]))

# with COEFFICIENTS we can identify the relationship between the dependent variable and each independent ones
print("If weight increases by 1 unit and all else stays the same, CO2 emissions shift {}.".format(REGRESSION_OBJ.coef_[0]))
print("If volume increases by 1 unit and all else stays the same, CO2 emissions shift {}.".format(REGRESSION_OBJ.coef_[1]))

# the issue with collecting data on each of these relationships is that it is hard to compare them
# STANDARDIZATION makes it easy to compare different units of measurement
# standardization equation: new value = ( original value - mean ) / standard deviation
SCALE = StandardScaler()

# here we take the original "Weight" and "Volume" and conert them to new columns of standardized values
SCALED_INDEPENDENT_VARS = SCALE.fit_transform(INDEPENDENT_VARS)
print("Original weight of {} standardized to {}.".format(INDEPENDENT_VARS['Weight'][0], SCALED_INDEPENDENT_VARS[0][0]))
print("Original volume of {} standardized to {}.".format(INDEPENDENT_VARS['Volume'][0], SCALED_INDEPENDENT_VARS[0][1]))

# these scaled amounts should give us the same predictions as the non-scaled amounts after we scale our input as well:
REGRESSION_OBJ_SCALED = linear_model.LinearRegression()
REGRESSION_OBJ_SCALED.fit(SCALED_INDEPENDENT_VARS, DEPENDENT_VAR)
SCALED_VALUES = SCALE.transform([[2300,1300]])
print("If the weight is 2300 and the volume is 1300, the CO2 will be {}.".format(REGRESSION_OBJ_SCALED.predict(SCALED_VALUES)[0]))

# to predict if a model is good enough, we use the TRAIN/TEST method
#	we split the data into a training set (80%) and a testing set (20%)
#	we use the training set to create our model, and the testing set to... test it!

# lets create a dataset of 100 customers
np.random.seed(2)
MINUTES_BEFORE_PURCHASE = np.random.normal(3, 1, 100)
MONEY_SPENT_ON_PURCHASE = np.random.normal(150,40,100)/MINUTES_BEFORE_PURCHASE

# and split the training and testing sets
TRAIN_MINUTES_BEFORE_PURCHASE = MINUTES_BEFORE_PURCHASE[:80]
TRAIN_MONEY_SPENT_ON_PURCHASE = MONEY_SPENT_ON_PURCHASE[:80]
TEST_MINUTES_BEFORE_PURCHASE = MINUTES_BEFORE_PURCHASE[80:]
TEST_MONEY_SPENT_ON_PURCHASE = MONEY_SPENT_ON_PURCHASE[80:]

# create multiple graphs to show the full dataset, and then the training and testing ones
fig, axs = plt.subplots(3)

# now put in the data for each scatterplot
#	for the testing and training sets to be fair, they should look similar to the full dataset
axs[0].scatter(MINUTES_BEFORE_PURCHASE, MONEY_SPENT_ON_PURCHASE)
axs[1].scatter(TRAIN_MINUTES_BEFORE_PURCHASE, TRAIN_MONEY_SPENT_ON_PURCHASE)
axs[2].scatter(TEST_MINUTES_BEFORE_PURCHASE, TEST_MONEY_SPENT_ON_PURCHASE)

# now, should we use linear regression, polynomial regression, or multiple regression?
#	based on the shape of the graph and us only having one independent variable, polynomial is best
POLYNOMIAL_MODEL = np.poly1d(np.polyfit(TRAIN_MINUTES_BEFORE_PURCHASE, TRAIN_MONEY_SPENT_ON_PURCHASE, 4))
POLYNOMIAL_LINE = np.linspace(0, 6, 100)
axs[0].plot(POLYNOMIAL_LINE, POLYNOMIAL_MODEL(POLYNOMIAL_LINE))

# by looking at the resulting line, we can see that overfitting has decresed accuracy of our model beyond a certain range
# lets check the strength of the relationship between the variables using R^2
print("\nR^2 Score for Training Data: {}".\
	format(r2_score(TRAIN_MONEY_SPENT_ON_PURCHASE, POLYNOMIAL_MODEL(TRAIN_MINUTES_BEFORE_PURCHASE))))
print("R^2 Score for Testing Data: {}".\
	format(r2_score(TEST_MONEY_SPENT_ON_PURCHASE, POLYNOMIAL_MODEL(TEST_MINUTES_BEFORE_PURCHASE))))
print("If a customer spends five minutes, we predict they'll spend {} dollars.".format(POLYNOMIAL_MODEL(5)))

plt.show()

