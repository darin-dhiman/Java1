

import pandas as pd                # importing libraries
import seaborn as sns
import matplotlib.pyplot as plt    # importing plotting lib

path = "NYC-Accidents-2020.csv"    # the path, where computer thinks your file is. In the same folder as your ipynb
df = pd.read_csv(path)             # reading the .csv file, csv = comma sep

df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)   # .rename(columns), rename columns to replace ' ', or spaces, with '_' underscores
df.head(3)

### CONVERT DATE TO NUMERICAL VALUE USING PANDAS

df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'])
reference_date = df['CRASH_DATE'].min()
df['CRASH_DATE'] = (df['CRASH_DATE'] - reference_date).dt.days


### CAN ALSO DEFINE A HOME-BREWED FUNCTION TO DO THIS
def new_time(time_str):
    """convert without datetime"""
    h, m, s = map(int, time_str.split(':'))
    return h + m/60 + s/3600

df['CRASH_TIME'] = df['CRASH_TIME'].apply(new_time)

df.head(3)

df.columns

fig, ax = plt.subplots(3, 1, figsize=(5, 8), sharex=True)

ax[0].scatter(df['ZIP_CODE'], df['NUMBER_OF_CYCLIST_INJURED'])
ax[0].set_title('Cyclists Injured')
ax[1].scatter(df['ZIP_CODE'], df['NUMBER_OF_CYCLIST_KILLED'])
ax[1].set_title('Cyclists Killed')
ax[2].scatter(df['ZIP_CODE'], df['NUMBER_OF_MOTORIST_KILLED'])
ax[2].set_title('Motorists Killed')

# plt.xlim([40.5, 40.95])

df.columns

print(df['LONGITUDE'].mean())
print(df['LATITUDE'].mean())

plt.scatter(df['LONGITUDE'],df['LATITUDE'], alpha=0.1, s=.2)
plt.ylim([40.5, 40.95])
plt.xlim([-74.4, -73.6])

fig, axs = plt.subplots(2, 1, figsize=(10, 12))    # establish our fig plot

borough = df[['CRASH_DATE','CRASH_TIME','BOROUGH']].dropna()  # make small dataframe and drop NaN (non input values)

sns.histplot(data=borough, x='CRASH_DATE', bins=48, hue="BOROUGH", stat="density", multiple="stack", ax=axs[0]).set_title('Crashs per date')
sns.histplot(data=borough, x='CRASH_TIME', bins=48, hue="BOROUGH", stat="density", multiple="stack",ax=axs[1]).set_title('Crashs per time')

plt.tight_layout()

plt.show()

"""## PREPROCESSING
"""

df['FATAL'] = df['NUMBER_OF_PERSONS_KILLED'].apply(lambda x: 0 if x == 0 else 1)

df = df.drop(['LOCATION',
              'ON_STREET_NAME',
              'CROSS_STREET_NAME',
              'OFF_STREET_NAME',
              'CONTRIBUTING_FACTOR_VEHICLE_3',
                'CONTRIBUTING_FACTOR_VEHICLE_4',
                'CONTRIBUTING_FACTOR_VEHICLE_5',
              'VEHICLE_TYPE_CODE_3',
              'VEHICLE_TYPE_CODE_4',
              'VEHICLE_TYPE_CODE_5',
              'COLLISION_ID'], axis=1)
df.shape

"""IDENTIFYING CATEGORICAL DATA AND SEPARATING"""

### IDENTIFY ALL NON NUMERICAL-CONTAINING COLUMNS

cc = list() ## cc stands for categorical-column

for col in df.columns:
  if df[col].dtype == 'object':
    cc.append(col)

cc

"""USING ONE-HOT ENCODINGS WITH GET DUMMIES"""

### GET DUMMIES FOR ALL CATEGORICALS

df_categorical = df[cc]
dummies = pd.get_dummies(df_categorical, dtype=int)

### ADD TO ORIGINAL DF
df.drop(cc, inplace=True, axis=1)
df = pd.concat([df, dummies], axis=1)

df.head()

"""## EDA

Start with some basic things:
- View dimensions of dataset
- Preview the dataset
- View column names
- View statistical properties of dataset
- Check for missing values | if yes, how to clean up missing values?
- View the frequency distribution of values
- Missing values in categorical variables

1) Find the shape of your training and testing datasets (Hint: use the shape() function)
"""

# Your code here
df.shape


plt.hist(df['ZIP_CODE'], bins=50)
plt.title('Crashes per zipcode')


# Your code here
plt.scatter(df['CRASH_DATE'] , df['NUMBER_OF_PERSONS_KILLED'], s=1)
plt.title('number persons killed per hour')

fig, ax = plt.subplots()
sns.scatterplot(data=df , x='CRASH_TIME', y='FATAL', ax=ax).set_title('FATALITIES PER HOUR')

df[['CRASH_TIME', 'FATAL']]

fig, ax = plt.subplots()
sns.histplot(data=df[['CRASH_TIME', 'FATAL']], x='CRASH_TIME', ax=ax).set_title('FATALITIES PER HOUR')

fig, ax = plt.subplots()
sns.histplot(data=df[['CRASH_DATE', 'FATAL']], x='CRASH_DATE', bins=241, ax=ax).set_title('FATAL ACCIDENTS PER DATE')

"""4) Try to think of one more step on your own here. What else would you like to know about the data or how it is arranged?

5) Make correlation matrix to observe correlations
"""

# Compute the correlation matrix
correlation_matrix = df[['CRASH_DATE',
                         'CRASH_TIME',
                         'ZIP_CODE',
                         'NUMBER_OF_PERSONS_INJURED',
                         'NUMBER_OF_PERSONS_KILLED',
                         'NUMBER_OF_PEDESTRIANS_INJURED',
                         'NUMBER_OF_PEDESTRIANS_KILLED',
                         'NUMBER_OF_CYCLIST_INJURED',
                         'NUMBER_OF_CYCLIST_KILLED',
                         'NUMBER_OF_MOTORIST_INJURED',
                         'NUMBER_OF_MOTORIST_KILLED',
                        ]].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', center=0)
plt.title('Correlation Matrix')
plt.show()

"""6) More scatterplots showing cool geographic visualizations"""

fig, axs = plt.subplots(2, 1, figsize=(4, 10))

sns.scatterplot(data=df,
                x='LONGITUDE',
                y='LATITUDE',
                size=df['NUMBER_OF_PERSONS_INJURED'],
                sizes=(0.5,10),
                hue=df['NUMBER_OF_PERSONS_INJURED'],
                ax=axs[0]).set_title('NYC Injuries')

axs[0].set_xlim(-74.25, -73.7)
axs[0].set_ylim(40.5,40.95)
axs[0].legend(title='number')

sns.scatterplot(data=df,
                x='LONGITUDE',
                y='LATITUDE',
                size=df['NUMBER_OF_PEDESTRIANS_INJURED'],
                 sizes=(0.5,10),
                hue=df['NUMBER_OF_PEDESTRIANS_INJURED'],
                ax=axs[1]).set_title('NYC Pedestrian Injuries')

axs[1].set_xlim(-74.25, -73.7)
axs[1].set_ylim(40.5,40.95)
axs[1].legend(title='number')

fig, axs = plt.subplots(2, 1, figsize=(4, 10))

sns.scatterplot(data=df[df['BOROUGH_MANHATTAN']==1],
                x='LONGITUDE',
                y='LATITUDE',
                size=df['NUMBER_OF_PERSONS_INJURED'],
                sizes=(3,100),
                hue=df['NUMBER_OF_PERSONS_INJURED'],
                ax=axs[0]).set_title('Manhattan Injuries')

axs[0].set_xlim(-74.05, -73.9)
axs[0].set_ylim(40.68,40.9)
axs[0].legend(title='number')

sns.scatterplot(data=df[df['BOROUGH_MANHATTAN']==1],
                x='LONGITUDE',
                y='LATITUDE',
                size=df['NUMBER_OF_PEDESTRIANS_INJURED'],
                sizes=(3,100),
                hue=df['NUMBER_OF_PEDESTRIANS_INJURED'],
                ax=axs[1]).set_title('Manhattan Pedestrian Injuries')

axs[1].set_xlim(-74.05, -73.9)
axs[1].set_ylim(40.68,40.9)
axs[1].legend(title='number')

"""## Baseline Model

Once you defined the explanatory variables and the outcome variables, you can go ahead and build a linear regression model as a baseline model.
"""

df.shape

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import numpy as np

data = df.loc[:, ['LATITUDE',
                  'LONGITUDE',
                  'CRASH_DATE',
                  'CRASH_TIME',
                  'NUMBER_OF_PERSONS_INJURED']].dropna()

# Dependent and Independent vars
X = data.drop('NUMBER_OF_PERSONS_INJURED', axis=1)
y = data['NUMBER_OF_PERSONS_INJURED']

data.shape

"""# Define data split

Split data into 30% for test set
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

# Randomly split into train and test sets (30% of data in test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Reshape data for model fitting
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1,1)

print(X_train.shape, X_test.shape)

"""# LINEAR REGRESSION, SIMPLE model

Simple linear regression model with a few parameters
"""

# fit on training data
linear_model = LinearRegression() # builds up model package
linear_model.fit(X_train, y_train) # trains model using training x and y

y_pred = linear_model.predict(X_test) #makes the prediactions

# PREDICTIONS
mse = mean_squared_error(y_test, linear_model.predict(X_test))
print("Simple input MSE:", mse)

"""## LOGISTIC REGRESSION Model

For improved models, you can explore more advanced models such as neural network or even recurrent neural network.
"""

### LOGISTIC REG, SAME MODEL

logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)

# print("Coefficients:", logit_model.coef_)
# print("Intercept:", logit_model.intercept_)
# Evaluation
preds = logit_model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# PREDICTIONS
# print("Predictions:", preds)
print("Simple input MSE:", mse)

"""# COMPLEX LINEAR MODEL"""

data = df.dropna()

# Dependent and Independent vars
X = data.drop(['NUMBER_OF_PERSONS_INJURED',

               # OTHER OBVIOUS CORRELATES

              'NUMBER_OF_PERSONS_KILLED', 'NUMBER_OF_PEDESTRIANS_INJURED',
              'NUMBER_OF_PEDESTRIANS_KILLED', 'NUMBER_OF_CYCLIST_INJURED',
              'NUMBER_OF_CYCLIST_KILLED', 'NUMBER_OF_MOTORIST_INJURED',
              'NUMBER_OF_MOTORIST_KILLED'],

              axis=1)
y = data['NUMBER_OF_PERSONS_INJURED']

# Randomly split into train and test sets (30% of data in test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Reshape data for model fitting
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1,1)

print(X_train.shape, X_test.shape)

# fit on training data
linear_model = LinearRegression() # builds up model package
linear_model.fit(X_train, y_train) # trains model using training x and y

y_pred = linear_model.predict(X_test) #makes the prediactions

# PREDICTIONS
mse = mean_squared_error(y_test, linear_model.predict(X_test))
print("COMPLEX MODEL MSE:", mse)

"""# WRONG WAY TO RUN COMPLEX MODEL


We should not run model with injuries for motorists, pedestrians, etc because then we have data leakage! MSE 2e-26 is unreasonably good.

"""

data = df.dropna()

# Dependent and Independent vars
X = data.drop(['NUMBER_OF_PERSONS_INJURED'],

              axis=1)
y = data['NUMBER_OF_PERSONS_INJURED']

# Randomly split into train and test sets (30% of data in test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Reshape data for model fitting
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1,1)

print(X_train.shape, X_test.shape)

# fit on training data
linear_model = LinearRegression() # builds up model package
linear_model.fit(X_train, y_train) # trains model using training x and y

y_pred = linear_model.predict(X_test) #makes the prediactions

# PREDICTIONS
mse = mean_squared_error(y_test, linear_model.predict(X_test))
print("COMPLEX MODEL MSE:", mse)

"""# Neural Net Model

Implement simple perceptron or simple model
"""

import tensorflow as tf

"""### Preprocessing: clean and scale data"""

data = df.dropna()

# Dependent and Independent vars
X = data.drop(['NUMBER_OF_PERSONS_INJURED',

               # OTHER OBVIOUS CORRELATES

              'NUMBER_OF_PERSONS_KILLED', 'NUMBER_OF_PEDESTRIANS_INJURED',
              'NUMBER_OF_PEDESTRIANS_KILLED', 'NUMBER_OF_CYCLIST_INJURED',
              'NUMBER_OF_CYCLIST_KILLED', 'NUMBER_OF_MOTORIST_INJURED',
              'NUMBER_OF_MOTORIST_KILLED'],

              axis=1)
y = data['NUMBER_OF_PERSONS_INJURED']

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Rejoin scaled data and use that as our X (scaled explanatory variables)
X = pd.DataFrame(X, columns = df.drop(['NUMBER_OF_PERSONS_INJURED',
              'NUMBER_OF_PERSONS_KILLED', 'NUMBER_OF_PEDESTRIANS_INJURED',
              'NUMBER_OF_PEDESTRIANS_KILLED', 'NUMBER_OF_CYCLIST_INJURED',
              'NUMBER_OF_CYCLIST_KILLED', 'NUMBER_OF_MOTORIST_INJURED',
              'NUMBER_OF_MOTORIST_KILLED'], axis = 1).columns) # convert back to pandas
X.head()

"""### Preprocessing: train test split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def plot_graphs(history, metric):
  """ plot history of model train """
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel("MSE")
  plt.legend(["Train MSE", "Test MSE"])

"""### Instantiate and train model"""

# define model architecture
nn_model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(30, activation=tf.nn.relu), # 30 neurons in this layer, activation is ReLU(x) = max(x, 0)
     tf.keras.layers.Dense(1) # output layer -> one neuron to get the final predicted value
  ])

# compile model
nn_model.compile(optimizer = "adam", # don't worry about this, just use "adam"
              loss = 'mse')

# fit our model
history = nn_model.fit(
    X_train, y_train, epochs=10,
    validation_data=(X_test, y_test)
)

plt.figure(figsize=(16, 6))
plot_graphs(history, 'loss')

"""### Complex architecture model"""

# define model architecture
nn_model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(16, activation=tf.nn.relu),
     tf.keras.layers.Dense(32, activation=tf.nn.relu),
     tf.keras.layers.Dense(16, activation=tf.nn.relu),
     tf.keras.layers.Dense(8, activation=tf.nn.relu),
     tf.keras.layers.Dense(4, activation=tf.nn.relu),
     tf.keras.layers.Dense(1) # output layer -> one neuron to get the final predicted value
  ])

# compile model
nn_model.compile(optimizer = "adam", # don't worry about this, just use "adam"
              loss = 'mse')

# fit our model
history = nn_model.fit(
    X_train, y_train, epochs=10,
    validation_data=(X_test, y_test)
)

plt.figure(figsize=(16, 6))
plot_graphs(history, 'loss')

"""## Tuning

Tuning is the process of maximizing a model's performance without overfitting or creating too high of a variance. In machine learning, this is accomplished by selecting appropriate “hyperparameters.” Hyperparameters can be thought of as the “dials” or “knobs” of a machine learning model.
"""

def build_model(n_hidden=1, n_neurons=30):
    model = tf.keras.models.Sequential()

    # Hidden layers
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))

    # Output layer for regression — 1 neuron, no activation
    model.add(tf.keras.layers.Dense(1))  # No activation for regression

    # Compile with regression-appropriate loss
    model.compile(loss="mean_squared_error",
                  optimizer='adam')

    return model

"""### Build validation set (not final test set)

Validation set will be used as a "pre-test" test, like an SAT practice test.
"""

X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)
X_train_cv.shape, X_valid.shape, y_train_cv.shape, y_valid.shape

# architecture parameters to test out
num_layers = [1,3,5,8] # numbers of layers
num_neurons = [10,30,50,70] # numbers of neurons per layer

# loop over different numbers of hidden layers
for n_hidden in num_layers:

  # inner loop over numbers of neurons per layer
  for n_neurons in num_neurons:

    # construct model using helpfer function
    model = build_model(n_hidden, n_neurons)

    # fit model using train set and evaluate using validation set
    print('Number of hidden layers: ', n_hidden)
    print('Number of neurons per layer: ', n_neurons)
    history = model.fit(X_train_cv, y_train_cv, epochs=10,
                    validation_data=[X_valid, y_valid])
