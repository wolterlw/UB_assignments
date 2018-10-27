import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin, TransformerMixin
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
import pickle

RMSE = lambda y_true,y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

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

class LinRegression(RegressorMixin):
    def __init__(self, lr=3e-4, lambda_=1, class_threshold=0.5, metric='RMSE'):
        assert metric in {'RMSE','accuracy'}
        self.params = {
            'lr': lr,
            'lambda_': lambda_,
            'thresh': class_threshold 
        }
        self.metric = metric
        self.W = None
        
    def score(self, X , y, sample_weight=None):

        if self.metric=='RMSE':
            score = RMSE(y, self.predict(X))
        if self.metric=='accuracy':
            score = accuracy_score(y, self.predict_class(X))
        return score
    
    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params.update(params)
        
    def step(self, W, X_train, Y_train):
        grad = (X_train @ W - Y_train).T @ X_train
        update = 1 / len(Y_train) * (grad + self.params['lambda_'] * W.T)
        return W - self.params['lr'] * update.T
    
    def predict(self, X):
        return np.asarray(X @ self.W).reshape(-1,) #not to return a matrix
    
    def predict_class(self, X, thresh = 0.5):
        return (X @ self.W > self.params['thresh']).astype('uint8')
        
    def fit(self, X_train, Y_train, valid_set=None, k=None, n_epochs=1000, batch_size=1,
            verbose=False, metric=RMSE, class_weights=[1,1], **fit_params):

        Y = Y_train.reshape(-1,1)
        np.random.seed(574)
        if k is None:
            X = X_train
        else:
            X = construct_Fi(X_train, k)

        self.W = np.random.uniform(-1, 1 ,size=X.shape[1]).reshape(-1,1)
        hist = []

        if verbose:
            epochs = tqdm(range(1,n_epochs))
        else:
            epochs = range(1,n_epochs)
        if valid_set:
            hist.append((0, metric(self.predict(X),Y), metric(self.predict(valid_set[0]), valid_set[1])))

        # dataset = ptDataset(X, Y)
        # sampler = WeightedRandomSampler(
        #     np.sum([(Y == i)*class_weights[i] for i in range(len(class_weights))], axis=1),
        #     num_samples=batch_size, replacement=False)

        # loader = DataLoader(dataset, batch_size, sampler=sampler)

        for i in epochs:
            self.W = self.step(self.W, X, Y)
            if valid_set:
                if verbose:
                    epochs.set_description(
                        f"""epoch {i}: 
                        train_error={metric(self.predict(X),Y):.4f}
                        valid_error={metric(self.predict(valid_set[0]),valid_set[1]):.4f}"""
                    )
                hist.append((i, metric(self.predict(X),Y), metric(self.predict(valid_set[0]), valid_set[1])))
            else:
                if verbose:
                    epochs.set_description(
                        f"epoch {i}: metric={metric(self.predict(X),Y):.4f}"
                    )
                hist.append((i,metric(self.predict(X),Y)))
        self.fit_hist_ = hist
        return self

class LogRegression(ClassifierMixin):
    def __init__(self, lr=0.01, lambda_=1, metric='accuracy', class_threshold=0.5, class_weights=[1, 1]):
        assert metric in {'RMSE','accuracy'}
        self.params = {
            'lr': lr,
            'lambda_': lambda_,
            'thresh': class_threshold,
            'class_weights': class_weights
        }
        self.metric = metric
        self.W = None

    def score(self, X , y, sample_weight=None):
        if self.metric=='RMSE':
            score = RMSE(y, self.predict_proba(X))
        if self.metric=='accuracy':
            score = accuracy_score(y, self.predict(X))
        return score

    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        self.params = params

    def _sigm(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _loss(self, s, y):
        return (-y * np.log(s) - (1 - y) * np.log(1 - s)).mean()
    

    def step(self, W, X_train, Y_train):
        a = self._sigm(X_train @ W)
        update = 1/len(Y_train) * (X_train.T @ (a - Y_train) + self.params['lambda_'] * W)
        return W - self.params['lr'] * update
    
    def predict_proba(self, X):
        return np.asarray(self._sigm(X @ self.W)).reshape(-1,1)
    
    def predict(self, X):
        return self.predict_proba(X) > self.params['thresh']

    def fit(self, X_train, Y_train, valid_set=None, n_epochs=1000, verbose=False, **fit_params):
        Y = Y_train.reshape(-1,1)
        np.random.seed(574)
        self.W = self.W = np.random.uniform(-1, 1 ,size=X_train.shape[1]).reshape(-1,1)
        hist = []
        
        if verbose:
            epochs = tqdm(range(1,n_epochs))
        else:
            epochs = range(1,n_epochs)
            
        for i in epochs:
            self.W = self.step(self.W, X_train, Y)
            train_loss = self._loss(self.predict_proba(X_train), Y)

            if valid_set:
                valid_loss = self._loss(self.predict_proba(valid_set[0]), valid_set[1].reshape(-1,1))
                if verbose:
                    epochs.set_description(
                        f"""epoch {i}:
                        train_loss={train_loss:.4f}
                        valid_loss={valid_loss:.4f}
                        """
                    )
                hist.append((i, train_loss, valid_loss))
            else:
                if verbose:
                    epochs.set_description(
                        f"epoch{i}: train_loss={train_loss:.4f}"
                        )
                hist.append((i,train_loss))
        self.fit_hist_ = hist
        return self


class ptDataset(Dataset):
    def __init__(self, X, y, normalize=False):
        if normalize:
            Xn = X / X.max(axis=0)
        else:
            Xn = X

        self.X = torch.tensor(Xn).float()
        self.y = torch.tensor(y).long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx]
        }

class Net(nn.Module):

    def __init__(self, in_size, out_size, hidden_size, lr=0.01):
        super(Net, self).__init__()
        self.params = {
            'in_size': in_size,
            'out_size': out_size,
            'hidden_size': hidden_size,
            'lr': lr
        }
        self._build_network()
        
    def _build_network(self):
        self.fc1 = nn.Linear(self.params['in_size'], self.params['hidden_size'])
        self.fc2 = nn.Linear(self.params['hidden_size'], self.params['out_size'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def fit(self, X_train, y_train, val_set=None, verbose=True,
            n_epochs=1, batch_size=1, class_weight=[1.,1.]):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.params['lr'])
        data_train = ptDataset(X_train, y_train)
        loader = DataLoader(data_train, batch_size, shuffle=True)

        if verbose:
            epochs = tqdm(range(n_epochs))
        else:
            epochs = range(n_epochs)
        
        for epoch in epochs:
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = self(batch['X'])
                loss = F.nll_loss(out, batch['y'], weight=torch.tensor(class_weight))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose:
                epochs.set_description(f"loss: {total_loss:.4f}")