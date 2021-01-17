
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# for the final intro topic, let us make a DECISION TREE
#	--a flow chart that lets you make decisions based on previous experience

# lets read in some data about shows
df = pandas.read_csv("shows.csv")
print(df)

# this dataset contains some non-numeric values in the NATIONALITY and GO columns
# lets convert these values into numbers using dictionaries to map to numeric values:
NATIONALITY_DICT = {'UK':0, 'USA':1, 'N':2}
GO_DICT = {'YES':1, 'NO':0}
df['Nationality'] = df['Nationality'].map(NATIONALITY_DICT)
df['Go'] = df['Go'].map(GO_DICT)
print(df)

# now we'll separate the FEATURE columns from the TARGET columns
#	AKA the independent variables from the dependent one
FEATURES = df[['Age', 'Experience', 'Rank', 'Nationality']]
TARGET = df['Go']

# now we are prepared to create the decision tree, as below:
dtree = DecisionTreeClassifier()
dtree = dtree.fit(FEATURES, TARGET)
data = tree.export_graphviz(dtree, out_file=None, feature_names=['Age', 'Experience', 'Rank', 'Nationality'])

# and we'll save the decision tree to a png file
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

# and also show it when we run the code
img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show() 