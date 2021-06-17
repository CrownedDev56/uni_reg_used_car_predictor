import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

## pd.read_csv: imports the carAi csv file from the saved folder
car = pd.read_csv('carAi.csv')

## pandas dataframe with car data and columns labeled 
data  = pd.DataFrame(car, columns = ['price', 'make', 'fuel',
                                     'model', 'transmission',
                                     'cylinders','litres',
                                     'year','body_type', 
                                     'odometer', 'economy', 'category'])

## .head: This function returns the first 5 rows for the object based on position.
print(data.head(5))
## .isnull: lists how many null values are in each feature
print(data.isnull().sum())
print(data.dtypes)

## value.counts: displaying the statistics of individual objects in a feature 

print(data['make'].value_counts())
print(data['fuel'].value_counts())
print(data['model'].value_counts())
print(data['transmission'].value_counts())
print(data['category'].value_counts())
print(data['odometer'].value_counts())
print(data['body_type'].value_counts())

## nunique:represents unique value of objects in a feature
print(data['make'].nunique())
print(data['fuel'].nunique())
print(data['model'].nunique())
print(data['transmission'].nunique())
print(data['category'].nunique())
print(data['odometer'].nunique())
print(data['body_type'].nunique())

#Dropna: remove records with null values from the data frame
data = data.dropna(subset = ['category','odometer','economy'])

## le.fit_transform: converts string data in dataset to positive integers
le = preprocessing.LabelEncoder()

data['make']= le.fit_transform(data['make'])
data['economy']= le.fit_transform(data['economy'])
data['fuel']= le.fit_transform(data['fuel'])
data['model']= le.fit_transform(data['model'])
data['transmission']= le.fit_transform(data['transmission'])
data['category']= le.fit_transform(data['category'])
## fillna: fills records with missing data
data.fillna({'body_type':'SUV'}, inplace=True)
data['body_type']= le.fit_transform(data['body_type'])

pd.set_option('display.max_columns',18)


## iloc: integer location based indexing for selection by position pointing to 9 features 1-10 and one target 0 
features = data.iloc[:,1:10]
target = data.iloc[:,0]

## preprocessing.scale() standardizing the features
x_features = preprocessing.scale(features)

features_train, features_test, target_train, target_test = train_test_split(x_features,
target, test_size = 0.33, random_state = 10)

## building model
model = Sequential()

## adding input hidden and output layers
model.add(Dense(18, input_dim=9,kernel_initializer='normal', activation='relu'))
model.add(Dense(20,kernel_initializer='normal', activation='relu'))
model.add(Dense(20,kernel_initializer='normal', activation='relu'))
model.add(Dense(18,kernel_initializer='normal', activation='relu'))
model.add(Dense(1, activation= 'linear')) 


model.compile(loss='mean_absolute_error', optimizer='adam')
 
model.fit(features_train,target_train, epochs=100, batch_size=801, verbose=1)

pred = model.predict(features_test)




sns.pairplot(features)

##############################################################################################
                                        ##Graphs##
#from the scattergraph, we noticed outliers that didnt fit the model.
#set the upper bound of price column to 60000
data['price'] = data['price'].clip(upper=60000)
data['odometer']=data['odometer'].clip(upper=500000)




plt.scatter (data['year'], data['odometer'], color='red')
plt.title('Milage vs Year', fontsize = 12)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Milage', fontsize = 12)
plt.grid(True)
plt.show()
plt.scatter (data['year'], data['price'], color='red')
plt.title('Price vs Year', fontsize = 12)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Price', fontsize = 12)
plt.grid(True)
plt.show()
                                        
var = data.groupby('fuel').price.sum()
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('fuel') #diesel is 1, unleaded is 2
ax1.set_ylabel('price')
ax1.set_title("Price vs Fuel")
var.plot(kind='bar')

var = data.groupby('make').price.sum()
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('make') #Subaru is 1, Toyota is 2
ax1.set_ylabel('price')
ax1.set_title("Price vs Make")
var.plot(kind='bar')

var = data.groupby('model').price.sum()
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('model') #1 is forester, 2 is impreza, 3 is rav4
ax1.set_ylabel('price')
ax1.set_title("Price vs Model")
var.plot(kind='bar')

var = data.groupby('transmission').price.sum()
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('transmission') #1 is Automatic, 2 is Manual
ax1.set_ylabel('price')
ax1.set_title("Price vs Transmission")
var.plot(kind='bar')

var = data.groupby('category').price.sum()
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('category')
ax1.set_ylabel('price')
ax1.set_title("Price vs Category")
var.plot(kind='bar')

var = data.groupby('body_type').price.sum()
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.set_xlabel('body_type')
ax1.set_ylabel('price')
ax1.set_title("Price vs Body_type")
var.plot(kind='bar')
 

plt.show()
plt.scatter (pred, target_test, color='red')
plt.title('Predicted vs Actual', fontsize = 12)
plt.xlabel('Predicted Values', fontsize = 12)
plt.ylabel('Actual Values', fontsize = 12)
plt.grid(True)
             
                    
                                        
##############################################################################################

print('Mean Absolute Error Score of:', mean_absolute_error(target_test, pred))
print('Mean Squared Error Score of:', mean_squared_error(target_test, pred))
print('R2 square value Score of:', r2_score(target_test, pred)*100)




 