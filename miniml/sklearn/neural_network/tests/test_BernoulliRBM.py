import pytest
import numpy as np
from numpy.testing import assert_allclose

from miniml.sklearn.neural_network import BernoulliRBM 


@pytest.fixture(scope="module")
def simple_rbm_data():
    """
    提供一個簡單可重現的數據集，用於所有測試。
    """
    pattern_A = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    pattern_B = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    
    X = np.vstack([pattern_A] * 100 + [pattern_B] * 100)
    
    # randomize with fixed seed for reproducibility
    rng = np.random.default_rng(42) 
    rng.shuffle(X)
    
    test_patterns = np.vstack([pattern_A, pattern_B])
    
    return {"X": X, "test_patterns": test_patterns}


def test_BernoulliRBM_fit_and_reconstruction(simple_rbm_data):
    """
    測試 fit() 是否能正確學習
    """
    X = simple_rbm_data["X"]
    test_patterns = simple_rbm_data["test_patterns"]
    
    n_features = X.shape[1]
    n_components = 2
    
    rbm = BernoulliRBM(
        n_components=n_components,
        learning_rate=0.1,
        n_epochs=300,
        batch_size=10,
        random_state=42
    )
    rbm.fit(X)

    # test shape of fitted parameters
    assert rbm.W_.shape == (n_features, n_components)
    assert rbm.b_.shape == (1, n_features)
    assert rbm.c_.shape == (1, n_components)
    
    # test reconstruction accuracy
    reconstruction = rbm.reconstruct(test_patterns, n_steps=1)
    assert_allclose(reconstruction, test_patterns, atol=0.05)

    print("BernoulliRBM fit/reconstruction test passed.")


def test_BernoulliRBM_transform(simple_rbm_data):
    """ 
    測試 transform() 
    """
    X = simple_rbm_data["X"]
    test_patterns = simple_rbm_data["test_patterns"]
    n_components = 2
    
    rbm = BernoulliRBM(
        n_components=n_components,
        n_epochs=300,
        random_state=42
    )
    rbm.fit(X)

    features = rbm.transform(test_patterns)
    
    # test output shape
    assert features.shape == (test_patterns.shape[0], n_components)

    # test value range (must be probabilities between 0 and 1)
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)

    # test functionality (features should be discriminative)
    features_A = features[0] # pattern A's features
    features_B = features[1] # pattern B's features

    # compute the distance between the two feature vectors (L2 distance squared)
    feature_distance = np.sum((features_A - features_B) ** 2)

    assert feature_distance > 0.5, "Features are not discriminative"
    print("BernoulliRBM transform test passed.")


def test_BernoulliRBM_invalid_input(simple_rbm_data):
    """
    測試在不當情況下呼叫應拋出錯誤：
    1. 在 'fit' 之前呼叫 'transform' 或 'reconstruct'。
    2. 'transform' 的輸入 features 數量不匹配。
    """
    rbm = BernoulliRBM(n_components=2)
    
    # test function before fit
    dummy_data = np.array([[1, 0, 1, 0, 0, 0]], dtype=np.float32)
    
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        rbm.transform(dummy_data)
        
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        rbm.reconstruct(dummy_data)
        
    # test with wrong input shape
    X = simple_rbm_data["X"] # length 6 features
    rbm.fit(X)
    
    X_wrong_shape = np.array([[1, 0, 1, 0]], dtype=np.float32) # wrong length 4 features
    
    with pytest.raises(ValueError):
        rbm.transform(X_wrong_shape)
        
    print("BernoulliRBM API error handling test passed.")


def test_BernoulliRBM_fit_transform_consistency(simple_rbm_data):
    """
    測試 rbm.fit_transform(X) 是否等同於 rbm.fit(X) 後再 rbm.transform(X)
    """
    X = simple_rbm_data["X"]
    
    # fit() + transform()
    rbm_fit = BernoulliRBM(n_components=2, n_epochs=50, random_state=42)
    rbm_fit.fit(X)
    features_from_fit = rbm_fit.transform(X)
    
    # fit_transform()
    rbm_fit_transform = BernoulliRBM(n_components=2, n_epochs=50, random_state=42)
    features_from_fit_transform = rbm_fit_transform.fit_transform(X)
    
    # assert two results should be the same
    assert features_from_fit_transform.shape == features_from_fit.shape
    assert_allclose(features_from_fit_transform, features_from_fit)
    print("BernoulliRBM fit_transform consistency test passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])