import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

# Location of dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset

# read dataset for training purposes
titanicTrain = pd.read_csv("A5_train.csv")  

X_train = titanicTrain
Y_train = titanicTrain

X = X_train.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
Y = Y_train.Survived

#Testing material---------------------------------
titanicTest = pd.read_csv("A5_test.csv")

X_test = titanicTest
Y_test = titanicTest

X_test = X_test.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
Y_test = Y_test.drop(['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

#change the sex to 0 and 1, Female and Male
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform) 
X_test = X_test.apply(le.fit_transform)
scaler = StandardScaler()  
scaler.fit(X)

X_train = scaler.transform(X)  
X_test = scaler.transform(X_test)  
#----------------------------------------------

print("X equals\n" ,X)
print("Y equals\n", Y)
print("X_test equals\n",X_test)
print("Y_test equals\n",Y_test)

mlp = MLPClassifier(hidden_layer_sizes=(30, 25, 15), max_iter=1000)  
mlp.fit(X, Y)

predictions = mlp.predict(X_test)
print(classification_report(Y_test,predictions)) 


#nnYTest = nnYTest.drop(['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

#3X_train = scaler.transform(X_train)  
#nnXTest = scaler.transform(X_test) 


##This splits 80% of the dataset into our training set and the other 20% in to test data.
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
#
#
##Before making actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated.
#
#scaler = StandardScaler()  
#scaler.fit(X_train)
#
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)  
#
##And now it's finally time to do what you have been waiting for, train a neural network that can actually make predictions:
#
#mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
#mlp.fit(X_train, y_train.values.ravel())
#
##The first parameter, hidden_layer_sizes, is used to set the size of the hidden layers. In our script we will create three layers of 10 nodes each. There is no standard formula for choosing the number of layers and nodes for a neural network and it varies quite a bit depending on the problem at hand. The best way is to try different combinations and see what works best.
#
##The second parameter to MLPClassifier specifies the number of iterations, or the epochs, that you want your neural network to execute. Remember, one epoch is a combination of one cycle of feed-forward and back propagation phase.
#
##By default the 'relu' activation function is used with 'adam' cost optimizer. However, you can change these functions using the activation and solver parameters, respectively.
#
##In the third line the fit function is used to train the algorithm on our training data i.e. X_train and y_train.
#
##The final step is to make predictions on our test data. To do so, execute the following script:
#predictions = mlp.predict(X_test)  
#
##We created our algorithm and we made some predictions on the test dataset. Now is the time to evaluate how well our algorithm performs. To evaluate an algorithm, the most commonly used metrics are a confusion matrix, precision, recall, and f1 score. The confusion_matrix and classification_report methods of the sklearn.metrics library can help us find these scores.
#
#print(confusion_matrix(y_test,predictions))  
#print(classification_report(y_test,predictions))  
#
##You can see from the confusion matrix that our neural network only misclassified one plant out of the 30 plants we tested the network on. Also, the f1 score of 0.97 is very good, given the fact that we only had 150 instances to train.
#
##Your results can be slightly different from these because train_test_split randomly splits data into training and test sets, so our networks may not have been trained/tested on the same data. But overall, the accuracy should be greater than 90% on your datasets as well.
