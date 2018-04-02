
import numpy as np
import pandas as pd

def ridge_regression(X, Y, lmbda=0.1):
    XtX_inv = np.linalg.inv(X.T.dot(X) + lmbda*np.eye(X.shape[1]))
    XtY = np.dot(X.T, Y)

    return np.dot(XtX_inv,XtY)


class RidgeRegression:
    def __init__(self, penalty=0.1, add_const=False, opt='stat', lr=0.1):
        self.beta = None
        self.penalty = penalty
        self.intercept = add_const
        self.opt = opt
        self.lr = lr

    # Calculate beta values that define model
    def fit(self, X, y):
        if self.opt == 'stat':
            XtX_inv = np.linalg.inv( X.T.dot(X) + self.penalty*np.eye(X.shape[1]) )
            XtY = np.dot(X.T, y)
            self.beta = np.dot(XtX_inv,XtY)

            if isinstance(X, pd.DataFrame):
                self.beta = np.ravel(self.beta)

        elif self.opt == 'sgd':
            self.beta = np.zeros(X.shape[1])



            # Try batch update
            # Try single data point update
            p1 = []
            p2 = []

            for i, xi in enumerate(X):
                # print(i, xi)

                pt1 = (1-2*self.penalty*self.lr)*self.beta

                err = y[i] - np.dot(xi, self.beta)
                # print( 'y = {}. err = {}'.format(y[i], err) )
                # print( 'Error shape = {}'.format(err.shape) )


                print(i)
                pt2 = 2*self.lr*xi*err

                self.beta = pt1 + pt2

                p1.append(pt1)
                p2.append(pt2)

                if self.beta[0] != self.beta[0]:
                    import pdb; pdb.set_trace()

    def predict(self, X):
        return X.dot(self.beta)
