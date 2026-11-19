import pandas;
from sklearn.model_selection import train_test_split;
import numpy;
import tensorflow;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Input, GRU, Dense;
from tensorflow.keras.optimizers import Adam;


New_Daily_Gold_Rate =  pandas.read_csv('Dates_Gold_Prices.csv');

features = ['Year', 'Month', "Day", "EUR"];

X = New_Daily_Gold_Rate[features];

y = New_Daily_Gold_Rate['USD'];

X_Train, X_Test, y_train, y_test = train_test_split(X, y, train_size = 0.8, shuffle=False);

# [Input(shape=(2,1)), GRU(50), Dense(100)]
GRU_model = Sequential();

GRU_model.add(GRU(50, input_shape=(2,1), return_sequences=True));
GRU_model.add(GRU(50));
GRU_model.add(Dense(1));

GRU_model.compile(optimizer=Adam(learning_rate=0.3), loss = 'mse');


GRU_model.fit(X_Train, y_train, epochs=10);

predicted_values = GRU_model.predict(X_Test);

print(predicted_values);
