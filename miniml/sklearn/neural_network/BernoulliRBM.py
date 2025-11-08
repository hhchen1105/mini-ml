import numpy as np

class BernoulliRBM:
    def __init__(self, n_components=128, learning_rate=0.01, n_epochs=10, 
                 batch_size=32, k=1, verbose=False, random_state=None):
        
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        
        self.rng = np.random.default_rng(random_state)
        
        self.W_ = None
        self.b_ = None
        self.c_ = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sample_prob(self, prob):
        return (self.rng.random(prob.shape) < prob).astype(np.float32)

    def _prop_up(self, v):
        pre_activation = v @ self.W_ + self.c_
        return self._sigmoid(pre_activation)

    def _prop_down(self, h):
        pre_activation = h @ self.W_.T + self.b_
        return self._sigmoid(pre_activation)

    def _contrastive_divergence(self, v0):
        ph0_prob = self._prop_up(v0)
        h0_state = self._sample_prob(ph0_prob)
        
        hk_state = h0_state
        vk_state = v0
        
        for _ in range(self.k):
            vk_prob = self._prop_down(hk_state)
            vk_state = self._sample_prob(vk_prob)
            
            phk_prob = self._prop_up(vk_state)
            hk_state = self._sample_prob(phk_prob)
        
        batch_size = v0.shape[0]
        
        grad_W = (v0.T @ ph0_prob - vk_state.T @ phk_prob) / batch_size        
        grad_b = np.mean(v0 - vk_state, axis=0, keepdims=True)        
        grad_c = np.mean(ph0_prob - phk_prob, axis=0, keepdims=True)
        
        reconstruction_error = np.mean((v0 - vk_prob)**2)
        
        return grad_W, grad_b, grad_c, reconstruction_error

    def fit(self, X):
        n_samples, n_features = X.shape
        
        if self.W_ is None:
            scale = np.sqrt(2.0 / (n_features + self.n_components))
            self.W_ = self.rng.normal(
                loc=0.0, 
                scale=scale, 
                size=(n_features, self.n_components)
            ).astype(np.float32)
            
            self.b_ = np.zeros((1, n_features), dtype=np.float32)
            self.c_ = np.zeros((1, self.n_components), dtype=np.float32)

        for epoch in range(self.n_epochs):
            epoch_error = 0.0
            
            shuffled_indices = self.rng.permutation(n_samples)
            X_shuffled = X[shuffled_indices]
            
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i : i + self.batch_size]
                
                grad_W, grad_b, grad_c, batch_error = self._contrastive_divergence(batch_X)
                
                self.W_ += self.learning_rate * grad_W
                self.b_ += self.learning_rate * grad_b
                self.c_ += self.learning_rate * grad_c
                
                epoch_error += batch_error
            
            if self.verbose:
                avg_error = epoch_error / (n_samples // self.batch_size)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Reconstruction Error: {avg_error:.6f}")
        
        return self

    def transform(self, X):
        if self.W_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        H = self._prop_up(X)
        return H

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def reconstruct(self, X, n_steps=1):
        if self.W_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        v = X
        for _ in range(n_steps):
            h_prob = self._prop_up(v)
            h_state = self._sample_prob(h_prob)
            v_prob = self._prop_down(h_state)
            v = v_prob
            
        return v_prob
    