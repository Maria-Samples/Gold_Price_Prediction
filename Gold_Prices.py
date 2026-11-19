import pandas;
from sklearn.model_selection import train_test_split;
from sklearn.ensemble import RandomForestRegressor;

New_Daily_Gold_Rate =  pandas.read_csv('Data_Price_Gold_Months.csv');

features = ['Year', 'Month'];

X = New_Daily_Gold_Rate[features];

y = New_Daily_Gold_Rate['#VALUE!'];

X_Train, X_Test, y_train, y_test = train_test_split(X, y, train_size = 0.8, shuffle=False, random_state=42);

Reggressor = RandomForestRegressor(n_estimators = 100, max_features='sqrt', random_state=42);
Reggressor.fit(X_Train, y_train);


y_pred = Reggressor.predict(X_Test);


import matplotlib.pyplot as plt
import matplotlib;
import numpy as np;
matplotlib.use('svg');
xpoints = X_Test['Year'] + X_Test['Month'];
ypoints = np.array(y_pred);

print(y_pred);
print(X_Test['Year']);

Graph = plt.figure();
Graph.set_figheight(10);
Graph.set_figwidth(10);

plt.plot(xpoints, ypoints, 'o');

plt.savefig('graph.svg');

xpoints = X_Test['Year'] + X_Test['Month'];
ypoints = np.array(y_test);

Graph = plt.figure();
Graph.set_figheight(10);
Graph.set_figwidth(10);

plt.plot(xpoints, ypoints, 'o');

plt.savefig('graph2.svg');
