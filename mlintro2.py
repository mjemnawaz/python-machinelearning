
import pandas
from sklearn import linear_model
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