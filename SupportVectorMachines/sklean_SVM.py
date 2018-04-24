from sklearn import svm
import pandas as pd

dt = pd.read_csv('../dataset-hw2/classification/train.csv')
dt.columns = ['x1', 'x2', 'x3', 'x4', 'y']
dt['b'] = 1
xpd = dt.loc[:, ['x1', 'x2', 'x3', 'x4']]
# xpd = dt.loc[:, ['b','x1', 'x2', 'x3', 'x4']]
dt.loc[dt['y'] == 0] = -1
ypd = dt.loc[:,['y']]


x = xpd.as_matrix()
y = ypd.T.values[0]

clf = svm.SVC()

clf.fit(x, y)

testD = pd.read_csv('../dataset-hw2/classification/test.csv')
testD.columns = ['x1', 'x2', 'x3', 'x4', 'y']
D = testD.as_matrix()
x_test = D[:, :4]
y_test = D[:, 4]


res = clf.predict(x_test)

count = 0
for i in range(len(y_test)):

    if y_test[i] == res[i]:
        count += 1

print(count/len(res))




