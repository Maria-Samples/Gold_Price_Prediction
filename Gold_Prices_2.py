import pandas;
from sklearn.model_selection import train_test_split;


New_Daily_Gold_Rate =  pandas.read_csv('Data_Price_Gold_Months.csv');

features = ['Year', 'Month'];

X = New_Daily_Gold_Rate[features];

y = New_Daily_Gold_Rate['#VALUE!'];

X_Train, X_Test, y_train, y_test = train_test_split(X, y, train_size = 0.8, shuffle=False);
