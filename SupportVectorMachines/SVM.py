import pandas as pd
import numpy as np
import sys
import os

def rt0(r0, t, d):
    denom = (r0*t)/d
    return (r0)/(1 + denom)

def rt1(r0, t, d):
    return (r0)/(1 + t)

class SVM:


    def fit(self, trainingData, max_epoch, r, rLearn, c, d):

        w = np.transpose(np.zeros(trainingData.shape[1]))
        _w = np.transpose(np.zeros(trainingData.shape[1]))
        N = trainingData.shape[0]

        for epoch in range (0, max_epoch):
            trainingData.sample(frac=1).reset_index(drop=True)
            xpd = trainingData.loc[:, ['x1', 'x2', 'x3', 'x4']]
            xpd['b'] = 1
            xpd = xpd.as_matrix()
            ypd = trainingData.loc[:,['y']].T.values[0]

            for t in range(trainingData.shape[0]):
                xi = xpd[t]

                yi = ypd[t] if ypd[t] == 1 else -1

                if yi * np.dot(np.transpose(w), np.array(xi)) <= 1:
                    _w = (1-rLearn(r, t, d))*(_w) + rLearn(r, t, d) * N * c * yi * np.array(xi)
                else:
                    _w = (1-rLearn(r, t, d))*_w
                w = w + _w
        return w

    def splitData(self, _dt):
        xpd = _dt.loc[:, ['x1', 'x2', 'x3', 'x4']]
        xpd['b'] = 1
        xpd = xpd.as_matrix()
        ypd = _dt.loc[:,['y']].T.values[0]

        return xpd, np.array([-1 if i == 0 else 1 for i in ypd])

    def predict(self, data, w):
        Xtest, Ytest = self.splitData(data)
        Yhat = np.sign(Xtest.dot(w))

        acc = sum(Yhat == Ytest)/float(np.size(Xtest,0))
        return acc

def readData(filename):
    _dt = pd.read_csv(filename)
    _dt.columns = ['x1', 'x2', 'x3', 'x4', 'y']
    return _dt

if __name__ == "__main__":

    print(os.getcwd())
    max_epoch = 50
    r = (1/2)**10
    C = [(10, 873), (100,873), (300,873), (500,873), (700,873)]
    d = (1/2)**5


    dt = readData(sys.argv[1])#('../dataset-hw2/classification/train.csv')
    test_dt = readData(sys.argv[2])#('../dataset-hw2/classification/test.csv')

    funcs = [rt0, rt1]
    for _func in funcs:
        for c in C:
            svm = SVM()
            w = svm.fit(dt, max_epoch, r, _func, c[0]/c[1], d)

            acc = svm.predict(test_dt, w)

            print("$C = \\frac{" + str(c[0]) + "}{" + str(c[1]) + "}$\\\ ")
            # print ("Weights: ${0}$ \\\ ".format(w))
            print ("Accuracy testing: ${0}$\\\ ".format(acc))
            print ("Test error = ${0}\\\ $".format(1-acc))
            print('\n')

    print('*'*10 + "Model Parameters" + '*'*10)
    for c in C:
        svm = SVM()
        w1 = svm.fit(dt, max_epoch, r, rt0, c[0]/c[1], d)
        acc1 = svm.predict(test_dt, w1)

        w2 = svm.fit(dt, max_epoch, r, rt1, c[0]/c[1], d)
        acc2 = svm.predict(test_dt, w2)

        print("$C = \\frac{" + str(c[0]) + "}{" + str(c[1]) + "}$\\\ ")

        print('Model Parameters: $\\frac{\gamma_0}{1+\\frac{\gamma_0}{d}t}$ and '),
        print('$\\frac{\gamma_0}{1+t}$')
        print()

        print('$[{0}]$\\\ '.format(', '.join([str(wi) for wi in w1]))),
        print('$[{0}]$\\\ '.format(', '.join([str(wi) for wi in w2]))),

        print()

        print('$\\frac{\gamma_0}{1+\\frac{\gamma_0}{d}t} = ' + str(acc1) + '$\\\ ')
        print('$\\frac{\gamma_0}{1+t} = ' + str(acc2) + '$\\\ ')
        print()