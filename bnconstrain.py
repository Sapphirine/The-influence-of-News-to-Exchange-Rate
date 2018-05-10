import pandas as pd
import bayespy
import numpy as np
from pgmpy.models import BayesianModel
model = BayesianModel([("trump","inflation"),
                       ("bitcoin","trade"),
                       ("bitcoin","sentiment"),
                       ("federal","tax"),
                       ("federal","bank"),
                       ("federal","trade"),
                       ("federal","inflation"),
                       ("bank","tax"),
                       ("bank","inflation"),
                       ("stock","investor"),
                       ("cryptocurrency","bitcoin"),
                       ("cryptocurrency","investor"),
                       ("oil","tax"),
                       ("oil","bank"),
                       ("oil","inflation"),
                       ("sentiment","tax"),
                       ("sentiment","inflation")
                       ])
data = pd.read_csv("vectors.csv")
data_train = data[: int(data.shape[0] * 0.5)]
model.fit(data_train)
model.get_cpds()
data_test = data[int(0.5 * data.shape[0]) : data.shape[0]]
y_test = np.array(data_test.ix[:,11])
data_test = data_test.ix[:,range(11)]
y_pre = np.array(model.predict(data_test))
count = 0
n = y_test.size
for i in range(n):
    if y_test[i] != y_pre[i]:
        count += 1
print(count/n)
