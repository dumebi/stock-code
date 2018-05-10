import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from keras.models import model_from_json
import pickle

df = pd.read_csv('input.csv',usecols=['open','high','low','volume','close'])
# print df.head()
df=df.fillna('999')
#
with open(r"decision_tree_classifier_20170212.pkl", "rb") as input_file:
    scaler = pickle.load(input_file)

# print scaler
#
def standard_scaler( X_test):

    test_samples, test_nx, test_ny = X_test.shape
    # print X_test.shape

    X_test = X_test.reshape((test_samples, test_nx * test_ny))

    # preprocessor = prep.StandardScaler().fit(X_test)
    # X_test = preprocessor.transform(X_test)
    X_test= scaler.fit_transform(X_test)

    X_test = X_test.reshape((test_samples, test_nx, test_ny))
    # print X_test
    return X_test


def preprocess_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[: int(row), :]

    # print ('row ',row)
    # print train


    result = standard_scaler(result)
    # print result
    X_test = np.reshape(result, (result.shape[0], result.shape[1], amount_of_features))
    # print X_test
    return  X_test


def loadSaved_Model(x_test):
  json_file = open('model133.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model133.h5")
  print("Loaded model from disk")

  # evaluate loaded model on test data
  loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  trainPredict = loaded_model.predict(x_test)
  # # #serialize model to json
  return trainPredict

# X_train = np.reshape(X_train.shape[0],X_train.shape[1],1))

X_test=preprocess_data(df[:: -1],0)

# print X_test.shape
pred=loadSaved_Model(X_test)
print pred


import matplotlib.pyplot as plt2

plt2.plot(pred, color='blue', label='Prediction')
plt2.legend()
plt2.show()
