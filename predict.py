import pandas
import yfinance as yf
import pandas as pd
import numpy as np
from ta import momentum
from datetime import datetime,timedelta
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
import sys
import os
#get stock training data
scaler = MinMaxScaler()
args = []
#getting stock
for arg in sys.argv:
    args.append(arg)
def get_inputs(sys_args):
    args = []
    for ticker in sys_args:
        args.append(str(ticker))
        print(type(args[-1]))
    return args
tickers = get_inputs(sys.argv[1:])
print(tickers)
#creating folder path
folder = 'Predictions ' + str(datetime.now().date())
desktop = os.path.join(os.path.expanduser('~'))
folder_path = os.path.join(desktop, folder)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def get_data(symbol, period):

    ticker = yf.Ticker(symbol)
    if period[-1] == 'y':

        #data = ticker.history(period=period)
        data = ticker.history(start=datetime.date(datetime.now() - timedelta(days=int(period[:-1])*365)), end = datetime.date(datetime.now() - timedelta(days = 2)))
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['Std_50'] = data['Close'].rolling(window=50).std()
        data['Band_1'] = data['MA50'] - data['Std_50']
        data['Band_2'] = data['MA50'] + data['Std_50']
        data['ON_returns'] = data['Close'].shift(1) - data['Open'].shift()
        data['ON_returns_signal'] = np.where(data['ON_returns'] > 0, 1, 0)  # 1 is up, 0 is down
        data['dist_from_mean'] = data['Close'] - data['MA50']
        rsi = momentum.RSIIndicator(data['Close'], window=14)
        data['RSI'] = rsi.rsi()
        data['EMA20'] = data['Close'].ewm(span=20).mean()
        temp = pd.DataFrame()
        temp['P*V'] = data['Close'] * data['Volume']
        temp['V'] = data['Volume']
        temp['P*V_sum'] = temp['P*V'].rolling(window=50).sum()
        temp['V_sum'] = temp['V'].rolling(window=50).sum()
        data['VWAP'] = temp['P*V_sum'] / temp['V_sum']

        data['future_returns'] = data['Close'].shift(-1) - data['Close']  # tommorow's return
        #print( data['future_returns'],np.where(data['future_returns'] > 0, 1, 0))
        data['future_returns'] = np.where(data['future_returns'] > 0, 1, 0)  # 1 = up, 0 = down
        data = data.drop(data.index[:200])
    elif period == 't':
        data = ticker.history(period='2y')
        #data = ticker.history(start = datetime.date(datetime.now() - timedelta(days = 2)), end = datetime.date(datetime.now()-timedelta(days = 1)) )
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['Std_50'] = data['Close'].rolling(window=50).std()
        data['Band_1'] = data['MA50'] - data['Std_50']
        data['Band_2'] = data['MA50'] + data['Std_50']
        data['ON_returns'] = data['Close'].shift(1) - data['Open'].shift()
        data['ON_returns_signal'] = np.where(data['ON_returns'] > 0, 1, 0)  # 1 is up, 0 is down
        data['dist_from_mean'] = data['Close'] - data['MA50']
        rsi = momentum.RSIIndicator(data['Close'], window=14)
        data['RSI'] = rsi.rsi()
        data['EMA20'] = data['Close'].ewm(span=20).mean()
        temp = pd.DataFrame()
        temp['P*V'] = data['Close'] * data['Volume']
        temp['V'] = data['Volume']
        temp['P*V_sum'] = temp['P*V'].rolling(window=50).sum()
        temp['V_sum'] = temp['V'].rolling(window=50).sum()
        data['VWAP'] = temp['P*V_sum'] / temp['V_sum']
        data['future_returns'] = data['Close'].shift(-1) - data['Close']  # tommorow's return

        data['future_returns'] = np.where(data['future_returns'] > 0, 1, 0)  # 1 = up, 0 = down
        data = data.tail(-2)
        data = data.tail(1)

    else:
        data = ticker.history(start = datetime.date(datetime.now() - timedelta(days = 365)), end = datetime.date(datetime.now() - timedelta(days = 2)))
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['Std_50'] = data['Close'].rolling(window=50).std()
        data['Band_1'] = data['MA50'] - data['Std_50']
        data['Band_2'] = data['MA50'] + data['Std_50']
        data['ON_returns'] = data['Close'].shift(1) - data['Open'].shift()
        data['ON_returns_signal'] = np.where(data['ON_returns'] > 0, 1, 0)  # 1 is up, 0 is down
        data['dist_from_mean'] = data['Close'] - data['MA50']
        rsi = momentum.RSIIndicator(data['Close'], window=14)
        data['RSI'] = rsi.rsi()
        data['EMA20'] = data['Close'].ewm(span=20).mean()
        temp = pd.DataFrame()
        temp['P*V'] = data['Close'] * data['Volume']
        temp['V'] = data['Volume']
        temp['P*V_sum'] = temp['P*V'].rolling(window=50).sum()
        temp['V_sum'] = temp['V'].rolling(window=50).sum()
        data['VWAP'] = temp['P*V_sum'] / temp['V_sum']
        data['future_returns'] = data['Close'].shift(-1) - data['Close']  # tommorow's return

        data['future_returns'] = np.where(data['future_returns'] > 0, 1, 0)  # 1 = up, 0 = down
        data = data.tail(int(period[:-1]))

    return data


for ticker in tickers:
    data= get_data(ticker,'2y')
    x = data.drop(['future_returns'], axis=1)
    y = data['future_returns'].values

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .3,random_state=102)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)

    mean_acc = np.zeros(30)
    prediction_set = []
    for i in range(1,31):
        knn = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
        yhat= knn.predict(x_test)
        mean_acc[i-1] = metrics.accuracy_score(y_test, yhat)
        prediction_set.append(yhat)

    #k_maxes = np.argpartition(mean_acc, -5)[-5:]
    knn2 = KNeighborsClassifier(n_neighbors=mean_acc.argmax() + 1)
    knn2.fit(x_train, y_train)
    predictions2 = knn2.predict(x_test)
    cm1 = confusion_matrix(y_test, predictions2)
    cr1 = classification_report(y_test, predictions2)
    knn_accuracy = accuracy_score(y_test, predictions2)
    #ANN
    data = get_data(ticker, '14d')
    hist = get_data('AAPL', '14d')
    x = hist.drop(['future_returns'], axis=1).drop(hist.index[-1]).values
    y = hist['Close'].shift(-1).drop(hist.index[-1]).values
    #split data
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .1,random_state=102)
    #scale data
    scaler=MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    #mode;
    model=Sequential()
    model.add(Dense(5,input_shape=(18,),activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1)) #output
    model.compile(optimizer='rmsprop', loss='mse')

    scale = MinMaxScaler()
    model.fit(x =x_train, y=y_train, epochs=50)
    scale.min_, scale.scale_ = scaler.min_[0], scaler.scale_[0]

    #live predict
    data = get_data('AAPL', 't')
    today_features = hist.drop(['future_returns'], axis=1)
    prediction = knn2.predict(today_features)
    today_features = scaler.transform(today_features)
    price = model.predict(today_features)
    price = scale.inverse_transform(price)
    delta_price = abs((price - hist['Close'].values)[0][0])
    delta_price = round(delta_price,2)
    dir = 'up'
    if prediction[0]:
        dir = 'down'
    file_path = os.path.join(folder_path, ticker + '_prediction')
    with open(file_path, 'a') as file:
        file.write('Prediction: Stock will go ' + dir + 'by $' + str(delta_price))
        file.write('\nAccuracy: ' + str(knn_accuracy))
        file.write(str(cm1))
        file.write(str(cr1))


