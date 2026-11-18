import pandas;
from sklearn.model_selection import train_test_split;
from sklearn.ensemble import RandomForestRegressor;
import matplotlib.pyplot as plt;
import numpy;


New_Daily_Gold_Rate =  pandas.read_csv('Dates_Gold_Prices.csv');

features = ['Year', 'Month', 'Day'];

X = New_Daily_Gold_Rate[features];

y = New_Daily_Gold_Rate['USD'];

X_Train, X_Test, y_train, y_test = train_test_split(X, y, train_size = 0.3);

Reggressor = RandomForestRegressor(n_estimators = 100);
Reggressor.fit(X_Train, y_train);


x_plot = range(len(New_Daily_Gold_Rate['Date']));

plt.xticks(x_plot, New_Daily_Gold_Rate['Date']);

y_plot = numpy.array(X_Test);

plt.plot(x_plot, y_plot);

plt.show();

