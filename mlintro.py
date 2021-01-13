
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# machine learning is for analyzing data and predicting the outcome!

# dataset - any collection of data
DATA_SET = [1, 2, 3, 4, 14, 45, 16, 87, 2, 35]
print("Dataset:\t\t {}".format(DATA_SET))

# there are 3 types of data: numerical (discrete or continuous), categorical, and ordinal

# 3 relevant values: MEAN (average value), MEDIAN (midpoint value), MODE (most common value)
#	to find the mean, add all values in the relevant dataset and divide by the length of the dataset
print("Mean:\t\t\t {}".format(np.mean(DATA_SET)))
#	to find the median, sort the values of the relevant dataset and locate the middle one
#	if the dataset is of an even length, return the sum divided by 2 of the two middle numbers
print("Median:\t\t\t {}".format(np.median(DATA_SET)))
#	for mode, return the number that occurs most often in the dataset
print("Mode:\t\t\t {}".format(stats.mode(DATA_SET)[0][0]))


# the VARIANCE and STANDARD DEVIATION measure the spread of the values in a dataset
#	to get the variance, take the mean of the squared difference from the mean of each value in the dataset
print("Variance:\t\t {}".format(np.var(DATA_SET)))
#	the standard deviation is the square root of the variance
print("Standard Deviation:\t {}".format(np.std(DATA_SET)))

# a PERCENTILE returns the value that a given percent of the values in the dataset are less than
#	so, if the 75th percentile is 23, 75% of the values in the dataset are less than 23
print("25th Percentile:\t {}".format(np.percentile(DATA_SET, 25)))
print("50th Percentile:\t {}".format(np.percentile(DATA_SET, 50)))
print("75th Percentile:\t {}".format(np.percentile(DATA_SET, 75)))

# machine learning often deals with large amounts of data, so it is useful to be able to generate large datasets for testing
RAND_DATA_SET_20 = np.random.uniform(0,10,20)
print("Randomly Generated Dataset of 50 Values: \n{}".format(RAND_DATA_SET_20))

# set up multiple charts
fig, axs = plt.subplots(2)

# sometimes it is useful to have a dataset based around a certain mean values with a certain standard deviation
# this is a NORMAL DISTRIBUTION. see the histograms to see how 250 values in a normal distribution are distinct from 250 random ones

# to visualize the values in a dataset, we can use a HISTOGRAM
# the histogram is a bar chart that shows how many values in the dataset fall into each interval
#	below, we make a histogram with 15 intervals dividing the range from 0 to 10
RAND_DATA_SET_250 = np.random.uniform(0,10,250)
NORM_DATA_SET_250 = np.random.normal(5,1,250)
axs[0].hist(RAND_DATA_SET_250, 15, alpha=0.5)
axs[0].hist(NORM_DATA_SET_250, 15, alpha=0.5)

# another useful visualization, a SCATTER PLOT
axs[0].scatter(RAND_DATA_SET_250, NORM_DATA_SET_250)

# the term REGRESSION is used when we try to find the relationship between 2 variables
# this relationship can be used to predict future trends

# LINEAR REGRESSION, for example, is when we find a linear relationship between the x and y variables
#	below we get the slope and y inercept of the linear relationship to chart the line
#	we also get the R-VALUE, from -1 to 1, where 0 indicates no relationship
#		in such a case, there may be no relationship or linear regression may not be the best regression model
slope, intercept, r_value, p_value, std_error = stats.linregress(RAND_DATA_SET_250, NORM_DATA_SET_250)
print("Relationship between random values on the x-axis and normal values on the y axis:\n\t {}\
	 [close to zero]".format(r_value))

# if there is a strong relationship, we can roughly predict future y-values given an x-value w the function below:
def getPoint(x):
	return slope * x + intercept
print("We predict that when x is 3, y is {}.\t[close to five]".format(getPoint(3)))

#	now we will create a list of points that fall on the linear relationship within the range of our scatterplot
LINE = list(map(getPoint, RAND_DATA_SET_250))
axs[0].plot(LINE, RAND_DATA_SET_250)

# POLYNOMIAL REGRESSION can be used for nonlinear relationships, as with the example below:
time = list(range(0,16))
speed = [90,87,82,89,76,65,40,34,46,49,52,40,56, 67, 89, 92]
axs[1].scatter(time, speed)
POLYNOMIAL_MODEL = np.poly1d(np.polyfit(time, speed, 3))
POLYNOMIAL_LINE = np.linspace(time[0], len(time), 100)
axs[1].plot(POLYNOMIAL_LINE, POLYNOMIAL_MODEL(POLYNOMIAL_LINE))
print("\n\nScatterplot Values: ")

for x in range(0,16):
	print("[{}, {}] ".format(x, speed[x]), end="")

print("\n\nPolynomial regression line equation:\n{}\n".format(POLYNOMIAL_MODEL))

# the strength of a relationship in polyomial regression is measured using the R^2 VALUE [-1 to 1, 0 = no relationship]
print("R^2 Value:\t\t {}".format(r2_score(speed, POLYNOMIAL_MODEL(time))))

# if the relationship is strong enough we can accurately predict future values, like so:
print("At time 16, the speed should be around {}".format(POLYNOMIAL_MODEL(16)))

plt.show()
