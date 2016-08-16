# import dataset
from sklearn import datasets

iris = datasets.load_breast_cancer()

X = iris.data  # features
y = iris.target  # labels

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# Here you can change classifier to your will.

# from sklearn.ensemble import RandomForestClassifier
# my_classifier = RandomForestClassifier()

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

print(y_test)
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score

# prints how accurate the predictions are
print(accuracy_score(y_test, predictions))
