
import numpy as np
import pandas as pd
from tqdm import tqdm

def ridge_regression(X, Y, lmbda=0.1):
    XtX_inv = np.linalg.inv(X.T.dot(X) + lmbda*np.eye(X.shape[1]))
    XtY = np.dot(X.T, Y)

    return np.dot(XtX_inv,XtY)

def calc_grad(X, y, beta,h):

    # import pdb; pdb.set_trace()

    grad = []

    for i in np.arange(0,len(beta)):
        h_vec = np.zeros(beta.shape)

        h_vec[i] = h

        f_x_plus_h = calc_cost(X, y, beta+h_vec)
        f_x_minus_h = calc_cost(X, y, beta-h_vec)
        grad.append( (f_x_plus_h-f_x_minus_h)/(2*h) )

    return np.array([np.squeeze(grad)]).T

def calc_cost(X, y, beta):
    return np.dot( (np.dot(X, beta) - y).T, (np.dot(X, beta) - y) )


class RidgeRegression:
    def __init__(self, penalty=0.1, add_const=False, opt='stat', lr=0.1, n_epochs=20):
        self.beta = None
        self.penalty = penalty
        self.intercept = add_const
        self.opt = opt
        self.lr = lr
        self.n_epochs = n_epochs
        self.J = np.array
        self.beta_history = np.array

    # Calculate beta values that define model
    def fit(self, X, y):

        if self.intercept:
            X = np.hstack((np.ones((X.shape[0],1)),X))

        if type(X) == pd.DataFrame:
            X = np.array(X)
        if type(y) == pd.DataFrame:
            y = np.array(y)


        if self.opt == 'stat':
            XtX_inv = np.linalg.inv( X.T.dot(X) + self.penalty*np.eye(X.shape[1]) )
            XtY = np.dot(X.T, y)
            self.beta = np.dot(XtX_inv,XtY)

            # if isinstance(X, pd.DataFrame):
            #     self.beta = np.ravel(self.beta)

        elif self.opt == 'gradient_descent':
            self.beta = np.array( np.zeros( (X.shape[1],1) ) )
            self.beta[0] = -2
            self.beta[1] = -3

            self.J = np.empty(self.n_epochs)
            self.beta_history = np.empty([self.beta.shape[0], self.n_epochs])

            # self.beta = np.array([[ 1.50192843e+02],
                                   # [ 6.55199614e-02],
                                   # [-1.86317709e+00],
                                   # [ 2.20902013e-02],
                                   # [ 8.14828026e-02],
                                   # [-2.47276537e-01],
                                   # [ 3.73276519e-03],
                                   # [-2.85747419e-04],
                                   # [-1.50284181e+02],
                                   # [ 6.86343742e-01],
                                   # [ 6.31476473e-01],
                                   # [ 1.93475697e-01]])*1.3

            g = []
            g_num = []

            h = 0.001

            for epoch in np.arange(0, self.n_epochs):

                # import pdb; pdb.set_trace()

                # Penalizing the size of the beta terms
                pt1 = -2*self.penalty*self.beta

                grad = -np.dot(np.dot(X.T, X), self.beta) - np.dot(X.T, y)
                grad_num = calc_grad(X, y, self.beta, h)
                g.append(grad)
                g_num.append(grad_num)

                # Penalizing the errors
                pt2 = - self.lr*grad_num


                self.beta = self.beta + pt1 + pt2/X.shape[0]

                # import pdb; pdb.set_trace()
                self.J[epoch] = calc_cost(X, y, self.beta).squeeze()
                self.beta_history[:,epoch] = self.beta.squeeze()

                # import pdb; pdb.set_trace()
                if self.beta[0] != self.beta[0]:
                    import pdb; pdb.set_trace()

        elif self.opt == 'sgd':
            self.beta = np.array( np.zeros( (X.shape[1],1) ) )
            self.beta[0] = -2
            self.beta[1] = -3

            self.J = np.empty(self.n_epochs*len(y))
            self.beta_history = np.empty([self.beta.shape[0], self.n_epochs*len(y)])

            g = []
            g_num = []

            h = 0.001

            for epoch in tqdm(np.arange(0, self.n_epochs)):

                #Shuffle datasets
                neworder = np.random.permutation(len(y))
                X = X[neworder]
                y = y[neworder]

                for i in np.arange(0, len(y)):

                    # Penalizing the size of the beta terms
                    pt1 = -2*self.penalty*self.beta

                    grad = -np.dot(np.dot(X[[i]].T, X[[i]]), self.beta) - np.dot(X[[i]].T, y[[i]])
                    grad_num = calc_grad(X[[i]], y[[i]], self.beta, h)
                    g.append(grad)
                    g_num.append(grad_num)

                    # Penalizing the errors
                    pt2 = - self.lr*grad_num


                    self.beta = self.beta + pt1 + pt2/X.shape[0]

                    # import pdb; pdb.set_trace()
                    iter = i + len(y)*epoch
                    self.J[iter] = calc_cost(X, y, self.beta).squeeze()
                    self.beta_history[:,iter] = self.beta.squeeze()

                    # import pdb; pdb.set_trace()
                    if self.beta[0] != self.beta[0]:
                        import pdb; pdb.set_trace()


    def predict(self, X):
        return X.dot(self.beta)




       #      self.beta = np.array([-4.81835161e-02, -1.94992120e+00, -2.92417298e-02,  2.52215689e-02,
       # -8.29971859e-01,  4.81834518e-03, -8.85493794e-04,  1.94321236e+00,
       #  1.86849322e-01,  4.15461148e-01,  3.66998785e-01])*1.3

           # err = y - np.dot(X, self.beta)
           # err2 = (X.T*err).T.sum(axis=0)
           # err3 = [xi*err[i]  for i, xi in enumerate(X)]
           # print(np.array(err).sum(axis=0))
           # print(np.array(err3).sum(axis=0))
