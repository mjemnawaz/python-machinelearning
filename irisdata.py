	
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape
print("Entries and attributes in this dataset: {}".format(dataset.shape))

# head
print("Check the first 5 entries: \n{}".format(dataset.head(5)))

# descriptions
print("Summary of each numeric attribute: \n{}".format(dataset.describe()))

# class distribution
print("Number of entries for each {}".format(dataset.groupby('class').size()))

# UNIVARIATE PLOT
# create 4 box and whisker subplots on a 2x2 grid for each numerica independent variable
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# we can also create corresponding histograms to visualize the spread of each variable
dataset.hist()
pyplot.show()

# now we can use a scatter plot matrix to identify relationships between the individual variable
# we can see clear relationships between some variables, implying a predictable correlation
scatter_matrix(dataset)
pyplot.show()

# split the dataset into training and testing datasets
array = dataset.values
X = array[:,0:4]
y = array[:,4]
# the testing set is 20% of the original dataset
# setting the random_state ensures our output will be reproducible, but is otherwise trivial
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# since we do not know which algrithms to use to build our models, we'll experiment w 6 different ones
# 	Logisitc Regression [linear]
#	Linear Discriminant Analysis [linear]
#	K-Nearest Neighbors [non-linear]
#	Classification and Regression Tress [non-linear]
#	Guassian Naive Bayes [non-linear]
#	Support Vector Machines [non-linear]

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
print()
for name, model in models:
	# we will be using k-fold cross validation to ensure accuracy
	# 	in this case we will use the usual default value for k, which is 10
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# rerunning the code may offer slightly different results
# overall, it appears that the most accurate model is the Support Vector Machines with 98% accuracy

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# the graph further supports the assumption that the Support Vector Machines is the best algorithm for us
#	so, let us use SVM to build a model that we use to make predictions
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions w accuracy score to know how close our predictions are to the data
#	the confusion matrix provides an indication of errors made
#  	the classification report provides a breakdown of each class by precision, recall, f1-score and support
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

