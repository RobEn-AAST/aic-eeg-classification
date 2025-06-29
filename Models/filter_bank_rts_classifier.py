from scipy.signal import butter, sosfiltfilt
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#  for motor imagery
# [I 2025-06-28 19:47:51,512] Trial 30 finished with value: 0.6598790349742903 and parameters: {'tmin': 60, 'filter_order': 3, 'fs': 100, 'n_estimators': 300, 'min_samples_split': 7, 'min_samples_leaf': 3}. Best is trial 30 with value: 0.6598790349742903.

# for ssvep if we will use it

class FilterBankRTSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, fs=250, order=4, n_estimators=100, bands=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", n_jobs=-1):
        self.bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)] if bands is None else bands
        self.fs = fs
        self.order = order
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def compute_fb_covs(self, X):
        """X: (n_trials, C, T) → fb_covs: (n_trials, B, C, C)"""
        # Pre-compute SOS filters if not done
        if not hasattr(self, "sos_bands"):
            self.sos_bands = [butter(self.order, (l / (self.fs / 2), h / (self.fs / 2)), btype="bandpass", output="sos") for l, h in self.bands]

        n, C, _ = X.shape
        B = len(self.sos_bands)
        fb_covs = np.zeros((n, B, C, C))
        for i, sos in enumerate(self.sos_bands):
            Xf = sosfiltfilt(sos, X, axis=2)
            fb_covs[:, i] = Covariances(estimator="lwf").transform(Xf)
        return fb_covs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        fb_covs = self.compute_fb_covs(X)
        n, B, C, _ = fb_covs.shape

        # Flatten for tangent space
        covs_flat = fb_covs.reshape(n * B, C, C)
        labels_rep = np.repeat(y, B)

        # Fit tangent space
        self.ts = TangentSpace(metric="riemann").fit(covs_flat, labels_rep)
        Z = self.ts.transform(covs_flat)
        Z = Z.reshape(n, B, -1)

        # Compute mutual information weights
        self.w = mutual_info_classif(Z.reshape(n, -1), y, discrete_features=False).reshape(B, -1).mean(axis=1)
        self.w = self.w / self.w.sum()

        # Weight features
        Z_weighted = np.concatenate([np.sqrt(self.w[i]) * Z[:, i, :] for i in range(B)], axis=1)

        # Train classifier
        self.clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                random_state=42,
            ),
        )
        self.clf.fit(Z_weighted, y)
        return self

    def predict(self, X):
        fb_covs = self.compute_fb_covs(X)
        n, B, C, _ = fb_covs.shape

        covs_flat = fb_covs.reshape(n * B, C, C)
        Z = self.ts.transform(covs_flat).reshape(n, B, -1)
        Z_weighted = np.concatenate([np.sqrt(self.w[i]) * Z[:, i, :] for i in range(B)], axis=1)

        return self.clf.predict(Z_weighted)

    def predict_proba(self, X):
        fb_covs = self.compute_fb_covs(X)
        n, B, C, _ = fb_covs.shape

        covs_flat = fb_covs.reshape(n * B, C, C)
        Z = self.ts.transform(covs_flat).reshape(n, B, -1)
        Z_weighted = np.concatenate([np.sqrt(self.w[i]) * Z[:, i, :] for i in range(B)], axis=1)

        return self.clf.predict_proba(Z_weighted)

