import pandas;

Daily_Gold_Rate = pandas.read_csv('daily_gold_rate.csv');

New_Daily_Gold_Rate = Daily_Gold_Rate.dropna();

print(Daily_Gold_Rate['Date']);

X = New_Daily_Gold_Rate['Date'];

y = New_Daily_Gold_Rate['USD'];

print(X);
print(y);