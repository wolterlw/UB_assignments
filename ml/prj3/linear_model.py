import numpy as np
#creating scikit-learn compatible predictors/transfromers to use Pipelines and GridSearch
from sklearn.base import RegressorMixin, ClassifierMixin, TransformerMixin 
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from tqdm import tqdm_notebook as tqdm #used to display a progressbar
import pickle #used to cache RBF functions

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

RMSE = lambda y_true,y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

class Subsampler():
    def __init__(self, num_batches, y_oh = True, normalize=True, neg_weight=1):
        self.num_batches = num_batches
        self.y_oh = y_oh
        self.normalize = normalize
        self.neg_weight = neg_weight
        self.fitted = False

    def fit(self, X, y):
        self.pos_idx = np.argwhere(y == 1)[:, 0]
        self.neg_idx = np.argwhere(y == 0)[:, 0]
        self.n_pos = len(self.pos_idx)
        max_ = X.max(axis=0)
        max_[max_ == 0] = 1
        self.norm_val = max_
        self.X_ = X / self.norm_val if self.normalize else X 
        self.y_ = np.c_[1-y, y] if self.y_oh else y.reshape(-1,1)
        self.fitted = True

    def transform(self, X, y, sample=False):
        X = X / self.norm_val if self.normalize else X 
        if sample:
            pos_idx = np.argwhere(y == 1)[:, 0]
            neg_idx = np.argwhere(y == 0)[:, 0]
            neg_subsample = np.random.choice(neg_idx, int(len(pos_idx)*self.neg_weight), replace=True)
            idx = np.r_[pos_idx, neg_subsample]
            np.random.shuffle(idx)
            X,y = X[idx], y[idx]
        y = np.c_[1-y, y] if self.y_oh else y.reshape(-1,1)
        return X, y

    def __iter__(self):
        assert self.fitted, "First fit the sampler to your data"
        for i in range(self.num_batches):
            neg_subsample = np.random.choice(self.neg_idx, int(self.n_pos*self.neg_weight), replace=True)
            idx = np.r_[self.pos_idx, neg_subsample]
            np.random.shuffle(idx)
            yield self.X_[idx], self.y_[idx]

    def get_generator(self):
        assert self.fitted, "First fit the sampler to your data"
        while True:
            neg_subsample = np.random.choice(self.neg_idx, int(self.n_pos * self.neg_weight), replace=True)
            idx = np.r_[self.pos_idx, neg_subsample]
            np.random.shuffle(idx)
            yield self.X_[idx], self.y_[idx]


class RBF_transformer(TransformerMixin):
    def __init__(self, num_clusters, cache=True):
        self.params = {'k': num_clusters}
        self._fi_j = None
        self.cache = cache

    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(params)

    def _get_mu_and_sigma(self, X):
        if self.cache:
            try:
                with open(f"./mu_sigma{self.params['k']}.pkl",'rb') as f:
                    mu_sigma = pickle.load(f)
                print("picked cached RBFs")
                return mu_sigma
            except:
                km = KMeans(n_clusters=self.params['k'], max_iter=500)
                labels = km.fit_predict(X)
                centroids = km.cluster_centers_
                mu_sigma = [{
                    'mu': centroids[i],
                    'sigma': np.diag(X[labels == i].var(axis=0) + np.random.uniform(0,0.1,size=X.shape[1]))
                } for i in range(self.params['k'])]
                
                print("writing to cache...")
                with open(f"./mu_sigma{self.params['k']}.pkl", 'wb') as f:
                    pickle.dump(mu_sigma, f)

                return mu_sigma
        else:
            km = KMeans(n_clusters=self.params['k'], max_iter=500)
            labels = km.fit_predict(X)
            centroids = km.cluster_centers_
            mu_sigma = [{
                'mu': centroids[i],
                'sigma': np.diag(X[labels == i].var(axis=0) + np.random.uniform(0,0.1,size=X.shape[1]))
            } for i in range(self.params['k'])]
            return mu_sigma

    def _fi(self, mu, sigma, vectorized=False):
        def f(x):
            return np.exp(
                -0.5 * (x-mu).T.dot(
                    np.dot(np.linalg.inv(sigma),(x-mu))
                )
            )

        def f_vec(x):
            return np.exp(
                -0.5* np.sum(
                    (x-mu).dot(
                    np.linalg.inv(sigma)) * (x-mu),
                    axis=1)
                )
        return f_vec if vectorized else f


    def fit(self, X, y=None, **fit_params):
        mu_sigma = self._get_mu_and_sigma(X)
        self._fi_j = [
            self._fi(ms['mu'], ms['sigma'], vectorized=True) for ms in mu_sigma
        ]
        return self

    def transform(self, X):
        Fi = np.matrix([f(X) for f in self._fi_j]).T
        Fi = np.c_[np.ones((Fi.shape[0],1)), Fi] # adding bias term
        return Fi

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y)
        return self.transform(X)

class MinibatchOptimized():
    """helper class to provide "fit" method"""
    def score(self, X, y, sample_weight=None):
        raise NotImplementedError

    def step(self, X_train, Y_train):
        """
        implements the gradient descent step and updates parameters
        """
        raise NotImplementedError

    def fit(self, X_train, Y_train, batch_generator, valid_set=None, n_epochs=1,**fit_params):
        batch_generator.fit(X_train, Y_train)
        Xtr, ytr = batch_generator.transform(X_train, Y_train, **fit_params)
        if valid_set:
            Xv, yv = batch_generator.transform(*valid_set, **fit_params)
        
        #initializing training history
        hist = []
        scores = [0,0,0] if valid_set else [0,0]
        scores[1] = self.score(Xtr, ytr)
        if valid_set:
            scores[2] = self.score(Xv, yv)
        hist.append(tuple(scores))
        
        #initializing iterators
        epochs = tqdm(range(1, n_epochs+1))

        for i in epochs:
            scores[0] = i
            for batch in batch_generator:
                self.step(*batch)

            scores[1] = self.score(Xtr, ytr)
            dsc = f"epoch: {i} train-sc={scores[1]:.3f}"
            
            if valid_set:
                scores[2] = self.score(Xv, yv)
                dsc += f" valid-sc={scores[2]:.3f}"

            hist.append(tuple(scores))
            epochs.set_description(dsc)
        self.fit_hist_ = hist
        return self
    
class LinRegression(RegressorMixin, MinibatchOptimized):
    def __init__(self, lr=0.01, lambda_=1, n_features=1, class_threshold=0.5, metric='RMSE'):
        self.params = {
            'lr': lr,
            'lambda_': lambda_,
            'thresh': class_threshold
        }
        self.metrics = {
            'RMSE': RMSE,
            'accuracy': accuracy_score
        } 
        assert metric in self.metrics.keys()
        self.metric = metric
        self.W = np.random.uniform(-1, 1 ,size=n_features).reshape(-1,1)
        
    def score(self, X , y, sample_weight=None):
        return self.metrics[self.metric](y, self.predict(X))
    
    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(params)
        
    def step(self, X_train, Y_train):
        grad = (X_train @ self.W - Y_train).T @ X_train
        update = 1 / len(Y_train) * (grad + self.params['lambda_'] * self.W.T)
        self.W -= self.params['lr'] * update.T
    
    def predict(self, X):
        if self.metric == 'RMSE':
            return np.asarray(X @ self.W).reshape(-1,) #not to return a matrix
        if self.metric == 'accuracy':
            return np.asarray(X @ self.W > self.params['thresh']).astype('uint8').reshape(-1,)

class LogRegression(ClassifierMixin, MinibatchOptimized):
    def __init__(self, lr=0.01, lambda_=1, n_features=1, class_threshold=0.5, metric='accuracy'):
        assert metric in {'RMSE','accuracy'}
        self.params = {
            'lr': lr,
            'lambda_': lambda_,
            'thresh': class_threshold
        }
        self.metrics = {
            'RMSE': RMSE,
            'accuracy': accuracy_score
        }
        self.metric = metric
        self.W = np.random.uniform(-1, 1 ,size=n_features).reshape(-1,1)

    def score(self, X , y, sample_weight=None):
        return self.metrics[self.metric](y, self.predict(X))

    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(params)

    def _sigm(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _loss(self, s, y):
        return (-y * np.log(s) - (1 - y) * np.log(1 - s)).mean()

    def step(self, X_train, Y_train):
        a = self._sigm(X_train @ self.W)
        update = 1/len(Y_train) * (X_train.T @ (a - Y_train) + self.params['lambda_'] * self.W)
        self.W -= self.params['lr'] * update

    def predict(self, X):
        if self.metric == 'RMSE':
            return self.predict_proba(X)
        if self.metric == 'accuracy':
            return (
                self.predict_proba(X) > self.params['thresh']
                ).astype('uint8')
    
    def predict_proba(self, X):
        return np.asarray(self._sigm(X @ self.W)).reshape(-1,1)

class Batcher():
    def __init__(self, batch_size, one_hot = True, fit_intercept = False, oh_order_func= sorted):
        self.batch_size = batch_size
        self.y_oh = one_hot
        self.fitted = False
        self.oh_of = oh_order_func

    def _one_hot(self, y):
        return np.c_[
            [y == i for i in self.oh_of(np.unique(y))]
        ].T.astype('uint8')
    
    def fit(self, X, y):
        self.X = X
        self.y =  self._one_hot(y) if self.y_oh else y.reshape(-1,1)
        self.fitted = True

    def transform(self, X, y, **kwargs):
        # X_ = X
        # y_ = self._one_hot(y) if self.y_oh else y.reshape(-1,1)
        return X, y

    def __iter__(self):
        assert self.fitted, "First fit the sampler to your data"
        for i in range(0, len(self.X), self.batch_size):
            yield self.X[i:i+self.batch_size], self.y[i:i+self.batch_size]

    def get_generator(self):
        while True:
            idx = np.random.randint(0, len(self.X), size=self.batch_size)
            yield self.X[idx], self.y[idx]
    
class SoftmaxRegression(MinibatchOptimized):
    def __init__(self, n_features, n_classes, lr=0.01, lambda_=1, metric='accuracy', fit_intercept=False):
        assert metric in {'accuracy'}
        self.params = {
            'lr': lr,
            'lambda_': lambda_,
            'fit_intercept': fit_intercept
        }
        self.n_classes = n_classes
        self.n_features = n_features
        self.metrics = {
            'accuracy': accuracy_score
        }
        self.metric = metric
        self.W = np.random.uniform(-1, 1 ,size=n_classes*(n_features+1)).reshape(n_classes, n_features+1)
        
    def add_bias(self, X):
        return np.c_[X, np.ones(len(X))]

    def score(self, X , y, sample_weight=None):
        y_ = y if y.ndim == 1 else np.argmax(y, axis=1)
        return self.metrics[self.metric](y_, self.predict(X))

    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(params)

    def _softmax(self, x):
        t = np.exp(x)
        return  t / np.sum(t, axis=1)[:,np.newaxis]
    
    def _loss(self, s, y):
        return (-np.log(s)*y).sum()

    def step(self, X_batch, Y_batch):
        X_ = self.add_bias(X_batch) if self.params['fit_intercept'] else X_batch
        z = self.predict_proba(X_batch)
        self.W += self.params['lr'] * (Y_batch - z).T @ X_ / len(X_)

    def predict(self, X):
        return np.argmax(
            self.predict_proba(X),
            axis=1)
    
    def predict_proba(self, X):
        X_ = self.add_bias(X) if self.params['fit_intercept'] else X
        return self._softmax(X_ @ self.W.T)