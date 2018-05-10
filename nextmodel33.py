import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt2
from sklearn.ensemble import BaggingRegressor
import pickle

df = pd.read_csv('modified_stock.csv')

# df=df.values
# X=df[:,:-1]
# y=df[:,-1]
# print y
#
#
# model=BaggingRegressor(verbose=2,n_estimators=1000000)
# model.fit(X,y)
#
#
# # Dump the trained decision tree classifier with Pickle
# decision_tree_pkl_filename = 'decision_tree_classifier_20170212.pkl'
# # Open the file to save as pkl file
# decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
# pickle.dump(model, decision_tree_model_pkl)
# # Close the pickle instances
# decision_tree_model_pkl.close()

# X=df[:,:-1]
# y=df[:,-1].values
# print y
# print X
# #
# # print df[:: -1]

def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape

    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))

    return X_train, X_test

def preprocess_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    print stock.columns
    print amount_of_features

    data = stock.as_matrix()
    print data[0]


    sequence_length = seq_len + 1
    print sequence_length

    print len(data)
    result = []
    for index in range(len(data) - sequence_length):
        # print data[index: index + sequence_length]

        result.append(data[index: index + sequence_length])

    print len(result)
    result = np.array(result)
    # print result


    print result.shape
    row = round(0.9 * result.shape[0])



    train = result[: int(row), :]

    train, result = standard_scaler(train, result)

    X_train = train[:, : -1]

    y_train = train[:, -1][:, -1]

    X_test = result[int(row):, : -1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]

def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model
#
window = 20
X_train, y_train, X_test, y_test = preprocess_data(df[:: -1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_testdfrvgrr ", X_test.shape)



print("y_test", y_test.shape)

model = build_model([X_train.shape[2], window, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=768,
    nb_epoch=300,
    validation_split=0.1,
    verbose=2)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
# serialize model to json
model_json = model.to_json()
with open("model133.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model133.h5")
print("Saved model to disk")

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
trainPredict = model.predict(X_test[1:20])
print trainPredict

diff = []
ratio = []
pred = model.predict(X_test[1:5])
for u in range(len(y_test[1:5])):
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

import matplotlib.pyplot as plt2

plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_test[1:5], color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()


