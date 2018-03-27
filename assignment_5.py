import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model

#------------------------------Survival predictions---------------------------
#-----------------------------------------------------------------------------
#
##----------------------------Neural Network----------------------------------
##----------------------------------------------------------------------------
#
## read dataset for training purposes
titanicTrain = pd.read_csv("A5_train.csv")  
titanicTest = pd.read_csv("A5_test.csv")

X_train = titanicTrain
Y_train = titanicTrain

X = X_train.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
Y = Y_train.Survived

#Testing material---------------------------------


X_test = titanicTest
Y_test = titanicTest

X_test = X_test.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
Y_test = Y_test.drop(['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
X = X.fillna(0)

#change the sex to 0 and 1, Female and Male
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform) 
X_test = X_test.apply(le.fit_transform)
scaler = StandardScaler()  
scaler.fit(X)

X = scaler.transform(X)  
X_test = scaler.transform(X_test)  
#----------------------------------------------

mlp = MLPClassifier(hidden_layer_sizes=(30, 25, 10), max_iter=1000)  
mlp.fit(X, Y)

predictions = mlp.predict(X_test)
#print the right information for the assignment Neural network portion
right = 0
count = 0
for i in Y_test.Survived:
    if i == predictions[count]:
        right +=1
        count +=1
    else:
        count+=1
print("Survived prediction: Neural Networks:",right,"/ 91")
#
##-----------------------------------Decision tree---------------------------
##---------------------------------------------------------------------------

db = pd.read_csv("A5_train.csv") 
dbT = pd.read_csv("A5_test.csv")

features = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

#---------------------training information
dty = db["Survived"]
dtx = db[features]
dtx.is_copy = False
#change the features to numbers
targetS = dtx["Sex"].unique()
targetA = dtx["Age"].unique()
targetT = dtx["Ticket"].unique()
targetC = dtx["Cabin"].unique()
targetE = dtx["Embarked"].unique()
targetN = dtx["Name"].unique()

map_to_intS = {name: n for n, name in enumerate(targetS)}
map_to_intA = {name: n for n, name in enumerate(targetA)}
map_to_intT = {name: n for n, name in enumerate(targetT)}
map_to_intC = {name: n for n, name in enumerate(targetC)}
map_to_intE = {name: n for n, name in enumerate(targetE)}
map_to_intN = {name: n for n, name in enumerate(targetN)}

dtx["Sex"] = dtx["Sex"].replace(map_to_intS)
dtx["Age"] = dtx["Age"].replace(map_to_intA)
dtx["Ticket"] = dtx["Ticket"].replace(map_to_intT)
dtx["Cabin"] = dtx["Cabin"].replace(map_to_intC)
dtx["Embarked"] = dtx["Embarked"].replace(map_to_intE)
dtx["Name"] = dtx["Name"].replace(map_to_intN)

#----------------------Testing information-----------------------------
dtyT = dbT["Survived"]
dtxT = dbT[features]
dtxT.is_copy = False
#change the features to numbers
TtargetS = dtxT["Sex"].unique()
TtargetA = dtxT["Age"].unique()
TtargetT = dtxT["Ticket"].unique()
TtargetC = dtxT["Cabin"].unique()
TtargetE = dtxT["Embarked"].unique()
TtargetN = dtxT["Name"].unique()

Tmap_to_intS = {name: n for n, name in enumerate(TtargetS)}
Tmap_to_intA = {name: n for n, name in enumerate(TtargetA)}
Tmap_to_intT = {name: n for n, name in enumerate(TtargetT)}
Tmap_to_intC = {name: n for n, name in enumerate(TtargetC)}
Tmap_to_intE = {name: n for n, name in enumerate(TtargetE)}
Tmap_to_intN = {name: n for n, name in enumerate(TtargetN)}

dtxT["Sex"] = dtxT["Sex"].replace(Tmap_to_intS)
dtxT["Age"] = dtxT["Age"].replace(Tmap_to_intA)
dtxT["Ticket"] = dtxT["Ticket"].replace(Tmap_to_intT)
dtxT["Cabin"] = dtxT["Cabin"].replace(Tmap_to_intC)
dtxT["Embarked"] = dtxT["Embarked"].replace(Tmap_to_intE)
dtxT["Name"] = dtxT["Name"].replace(Tmap_to_intN)

#-----------------------------Decision tree setup
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(dtx, dty)

predictions = dt.predict(dtxT[features])

#-----------------------------Print the results-----------------------
right = 0
count = 0
for i in dtyT:
    if i == predictions[count]:
        right +=1
        count +=1
    else:
        count+=1
print("Survived prediction: Decision tree:",right,"/ 91")

#----------------------------------------------------------------------
#-------------------------------Fare predictions-----------------------
#----------------------------------------------------------------------

#-------------------------------Neural Network
#---------Training data
##read dataset for training purposes

titanicTrain = pd.read_csv("A5_train.csv")  
titanicTest = pd.read_csv("A5_test.csv")

X_train = titanicTrain
Y_train = titanicTrain

X = X_train.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)
Y = Y_train.drop(['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)

#Testing material---------------------------------


X_test = titanicTest
Y_test = titanicTest

X_test = X_test.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)
Y_test = Y_test.drop(['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)
X = X.fillna(0)

#change the sex to 0 and 1, Female and Male
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform) 
X_test = X_test.apply(le.fit_transform)
scaler = StandardScaler()  
scaler.fit(X)

X = scaler.transform(X)  
X_test = scaler.transform(X_test)  
#----------------------------------------------

mlp = MLPClassifier(hidden_layer_sizes=(30, 25, 10), max_iter=1000)  
Y = Y.astype('str')

mlp.fit(X, Y.values.ravel())

predictions = mlp.predict(X_test)
#print the right information for the assignment Neural network portion
right = 0
count = 0
#print("y: ",Y)
#print("predictions",predictions)
#print("Y_test",Y_test)
for i in Y_test.Fare:
    max = i + 5
    min = i - 5
    if ((float(predictions[count]) < max) and (float(predictions[count]) > min)):
        right += 1
        count += 1
    else:
        count += 1
print("Fare Predictions: Neural Network:", right, "/91")



#---------------------------Linear Regression----------------------------------
#------------------------------------------------------------------------------
#titanicTrain = pd.read_csv("A5_train.csv")  
#titanicTest = pd.read_csv("A5_test.csv")
lrData = titanicTrain
lrTX = titanicTest
# define the data/predictors as the pre-set feature names  
lrX = lrData.drop(['PassengerId','Name','SibSp','Parch','Cabin','Fare','Embarked'],axis=1)
lrY = lrData.drop(['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)
# Put the target (housing value -- MEDV) in another DataFrame
lrTX = titanicTest.drop(['PassengerId','Name','SibSp','Parch','Cabin','Fare','Embarked'],axis=1)
lrTY = titanicTest.drop(['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)

lrTX = lrTX.fillna(0)

#change the sex to 0 and 1, Female and Male
le = preprocessing.LabelEncoder()
lrX = lrX.apply(le.fit_transform) 
lrTX = lrTX.apply(le.fit_transform)


lm = linear_model.LinearRegression()
model = lm.fit(lrX,lrY)

predictions = lm.predict(lrTX)

preds = list(predictions)

y_test_list = list(lrTY)

count = 0
right = 0
for i in lrTY.Fare:
    max = i + 5
    min = i - 5
    if ((float(preds[count]) < max) and (float(preds[count]) > min)):
        right +=1
        count += 1
    else:
        count += 1
print("Fare Predictions: Neural Network:", right, "/91")

#------------------------Decision Tree-----------------------------------------
#------------------------------------------------------------------------------

db = pd.read_csv("A5_train.csv") 
dbT = pd.read_csv("A5_test.csv")

features = ["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Cabin","Embarked"]

#---------------------training information
dty = db.drop(["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1)
dtx = db[features]
dtx.is_copy = False
#change the features to numbers
targetS = dtx["Sex"].unique()
targetA = dtx["Age"].unique()
targetT = dtx["Ticket"].unique()
targetC = dtx["Cabin"].unique()
targetE = dtx["Embarked"].unique()
targetN = dtx["Name"].unique()

map_to_intS = {name: n for n, name in enumerate(targetS)}
map_to_intA = {name: n for n, name in enumerate(targetA)}
map_to_intT = {name: n for n, name in enumerate(targetT)}
map_to_intC = {name: n for n, name in enumerate(targetC)}
map_to_intE = {name: n for n, name in enumerate(targetE)}
map_to_intN = {name: n for n, name in enumerate(targetN)}

dtx["Sex"] = dtx["Sex"].replace(map_to_intS)
dtx["Age"] = dtx["Age"].replace(map_to_intA)
dtx["Ticket"] = dtx["Ticket"].replace(map_to_intT)
dtx["Cabin"] = dtx["Cabin"].replace(map_to_intC)
dtx["Embarked"] = dtx["Embarked"].replace(map_to_intE)
dtx["Name"] = dtx["Name"].replace(map_to_intN)

#----------------------Testing information-----------------------------
dtyT = dbT["Fare"]
dtxT = dbT[features]
dtxT.is_copy = False
#change the features to numbers
TtargetS = dtxT["Sex"].unique()
TtargetA = dtxT["Age"].unique()
TtargetT = dtxT["Ticket"].unique()
TtargetC = dtxT["Cabin"].unique()
TtargetE = dtxT["Embarked"].unique()
TtargetN = dtxT["Name"].unique()

Tmap_to_intS = {name: n for n, name in enumerate(TtargetS)}
Tmap_to_intA = {name: n for n, name in enumerate(TtargetA)}
Tmap_to_intT = {name: n for n, name in enumerate(TtargetT)}
Tmap_to_intC = {name: n for n, name in enumerate(TtargetC)}
Tmap_to_intE = {name: n for n, name in enumerate(TtargetE)}
Tmap_to_intN = {name: n for n, name in enumerate(TtargetN)}

dtxT["Sex"] = dtxT["Sex"].replace(Tmap_to_intS)
dtxT["Age"] = dtxT["Age"].replace(Tmap_to_intA)
dtxT["Ticket"] = dtxT["Ticket"].replace(Tmap_to_intT)
dtxT["Cabin"] = dtxT["Cabin"].replace(Tmap_to_intC)
dtxT["Embarked"] = dtxT["Embarked"].replace(Tmap_to_intE)
dtxT["Name"] = dtxT["Name"].replace(Tmap_to_intN)

#-----------------------------Decision tree setup


dty = dty["Fare"].astype(int)

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(dtx, dty)

predictions = dt.predict(dtxT[features])

#-----------------------------Print the results-----------------------
right = 0
count = 0
for i in dtyT:
    if i == predictions[count]:
        right +=1
        count +=1
    else:
        count+=1
print("Fare prediction: Decision tree:",right,"/ 91")




















