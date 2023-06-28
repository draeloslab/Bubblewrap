import numpy as np

def rank_one_update_formula1(D, x1, x2=None):
    if x2 is None:
        x2 = x1
    return D - (D @ x1 @ x2.T @ D) / (1 + x2.T @ D @ x1)

class OnlineRegresion:
    def __init__(self):
        pass

    def initialize(self, use_stored, x, y):
        pass

    def lazy_observe(self, x, y):
        pass

    def update(self, x, y):
        pass

    def predict(self, x):
        pass


class SymmetricNoisy(OnlineRegresion):
    def __init__(self, d, forgetting_factor, noise_scale, n_perturbations=1, seed=24, init_min_ratio=3):
        super().__init__()

        if n_perturbations < 1:
            raise Exception("the number of perturbations has to be more than 1")

        if forgetting_factor > 1:
            raise Exception("the forgetting factor should be in (0,1]")

        # core stuff
        self.d = d
        self.forgetting_factor = forgetting_factor
        self.D = None
        self.F = np.zeros([d, d])
        self.c = np.zeros([d, 1])
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.n_perturbations = n_perturbations

        # initializations
        self.init_min_ratio = init_min_ratio
        self.n_observed = 0

    def initialize(self, use_stored=True, x=None, y=None):
        if not use_stored:
            for i in np.arange(x.shape[0]):
                self.update(x=x[i], y=y[i], update_D=False)
        self.D = np.linalg.pinv(self.F)

    def update(self, x, y, update_D=False):
        x = x.reshape([-1,1])
        y = np.squeeze(y)
        if update_D:
            self.D /= self.forgetting_factor
        else:
            self.F *= self.forgetting_factor

        self.c *= self.forgetting_factor

        for _ in range(self.n_perturbations):
            dx = self.rng.normal(scale=self.noise_scale, size=x.shape)
            for c in [-1, 1]:
                new_x = x + dx * c
                if update_D:
                    self.D = rank_one_update_formula1(self.D, new_x)
                else:
                    self.F = self.F + new_x @ new_x.T
                self.c = self.c + new_x * y

        self.n_observed += 1

    def lazy_observe(self, x, y):
        if self.n_observed >= self.init_min_ratio * self.d or self.D is not None:
            self.update(x, y, update_D=True)
        else:
            self.update(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.d:
                self.initialize()

    def predict(self, x):
        if self.D is None:
            return np.nan

        w = self.D @ self.c
        return np.squeeze(x.T @ w)


class WindowSlow(OnlineRegresion):
    def __init__(self, d,  window_size, forgetting_factor=1, init_min_ratio=3):
        super().__init__()

        if forgetting_factor > 1:
            raise Exception("the forgetting factor should be in (0,1]")

        # core stuff
        self.d = d
        self.forgetting_factor = forgetting_factor
        self.window_size = window_size

        self.x_window = []
        self.y_window = []
        self.D = None
        self.c = np.zeros([d, 1])


        # initializations
        self.init_min_ratio = init_min_ratio
        self.n_observed = 0

    def initialize(self, use_stored=True, x=None, y=None):
        if not use_stored:
            for i in np.arange(x.shape[0]):
                self.update(x=x[i], y=y[i], update_D=False)

        x_mat = np.array(self.x_window).reshape([-1,self.d])
        self.D = np.linalg.pinv(x_mat.T @ x_mat)

    def update(self, x, y, update_D=False):
        # x and y should not be multiple time-steps big
        x = x.reshape([-1,1])
        y = np.squeeze(y)

        # trim windows
        assert len(self.x_window) == len(self.y_window)
        while len(self.x_window) > self.window_size:
            self.x_window.pop(0)
            self.y_window.pop(0)

        # enforce forgetting factor
        if self.forgetting_factor < 1:
            for i in range(len(self.x_window)):
                # TODO: make all the forgetting factors uniform (with like square roots or whatever)
                self.x_window[i] *= self.forgetting_factor
                self.y_window[i] *= self.forgetting_factor

        self.x_window.append(x)
        self.y_window.append(y)

        # update c and d
        # todo: think about this D and update_D decision more
        x_mat = np.array(self.x_window).reshape([-1,self.d])
        y_mat = np.array(self.y_window).reshape([-1,1])
        if update_D:
            self.D = np.linalg.pinv(x_mat.T @ x_mat)
        self.c = x_mat.T @ y_mat

        self.n_observed += 1

    def lazy_observe(self, x, y):
        if self.n_observed >= self.init_min_ratio * self.d or self.D is not None:
            self.update(x, y, update_D=True)
        else:
            self.update(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.d:
                self.initialize()

    def predict(self, x):
        if self.D is None:
            return np.nan

        w = self.D @ self.c
        return np.squeeze(x.T @ w)


class WindowFast(OnlineRegresion):
    def __init__(self, d,  window_size, init_min_ratio=3):
        super().__init__()

        # core stuff
        self.d = d
        self.window_size = window_size

        self.x_window = []
        self.y_window = []
        self.D = None
        self.F = np.zeros([d, d])
        self.c = np.zeros([d, 1])

        # initializations
        self.init_min_ratio = init_min_ratio
        self.n_observed = 0

    def initialize(self, use_stored=True, x=None, y=None):
        if not use_stored:
            for i in np.arange(x.shape[0]):
                self.update(x=x[i], y=y[i], update_D=False)

        self.D = np.linalg.pinv(self.F)

    def update(self, x, y, update_D=False):
        # x and y should not be multiple time-steps big
        x = x.reshape([-1,1])
        y = np.squeeze(y)

        # trim windows
        assert len(self.x_window) == len(self.y_window)
        while len(self.x_window) > self.window_size:
            removed_x = self.x_window.pop(0)
            removed_y = self.y_window.pop(0)
            if update_D:
                self.D = rank_one_update_formula1(self.D, removed_x, -removed_x)
            else:
                self.F -= removed_x @ removed_x.T
            self.c -= removed_x @ removed_y.reshape([-1,1])

        self.x_window.append(x)
        self.y_window.append(y)

        # update c and d
        if update_D:
            self.D = rank_one_update_formula1(self.D, x)
        else:
            self.F += x @ x.T
        self.c += x @ y.reshape([-1,1])


        self.n_observed += 1

    def lazy_observe(self, x, y):
        if self.n_observed >= self.init_min_ratio * self.d or self.D is not None:
            self.update(x, y, update_D=True)
        else:
            self.update(x, y, update_D=False)
            if self.n_observed >= self.init_min_ratio * self.d:
                self.initialize()

    def predict(self, x):
        if self.D is None:
            return np.nan

        w = self.D @ self.c
        return np.squeeze(x.T @ w)


if __name__ == '__main__':

    rng = np.random.default_rng()
    m = 5000
    n = 5

    w = np.array([0, 1, 2, 3, 4]).reshape((-1,1))

    X = np.zeros(shape=(m,n))
    y = np.zeros(shape=(m,1))
    for i in range(m):
        X[i,:] = rng.normal(size=5)
    y = X @ w

    r1 = WindowFast(d=n, window_size=50)
    r2 = WindowSlow(d=n, window_size=50)
    for i in range(m):
        r1.lazy_observe(x=X[i], y=y[i])
        r2.lazy_observe(x=X[i], y=y[i])

    print(r1.D - r2.D)