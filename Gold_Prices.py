import pandas;
from sklearn.model_selection import train_test_split;
from sklearn.ensemble import RandomForestRegressor;

Daily_Gold_Rate = pandas.read_csv('daily_gold_rate.csv');

New_Daily_Gold_Rate = Daily_Gold_Rate.dropna();

X = New_Daily_Gold_Rate['Date'];

y = New_Daily_Gold_Rate['USD'];

X_Train, X_Test, y_train, y_test = train_test_split(X, y, train_size = 0.3);

Reggressor = RandomForestRegressor(n_estimators = 100);
Reggressor.fit(X_Train, y_train);

