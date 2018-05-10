import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
data = pd.read_csv("vectors.csv")
data_train = data[: int(data.shape[0] * 0.8)]
data_test = data[int(0.8 * data.shape[0]) : data.shape[0]]
y_test = data_test.ix[:,11]
x_test = data_test.ix[:,range(11)]
y_train = data_train.ix[:,11]
x_train = data_train.ix[:,range(11)]
model.fit(x_train,y_train)
y_test = np.array(data_test.ix[:,11])
data_test = data_test.ix[:,range(11)]
y_pre = np.array(model.predict(data_test))
count = 0
n = y_test.size
for i in range(n):
    if y_test[i] != y_pre[i]:
        count += 1
print(count/n)