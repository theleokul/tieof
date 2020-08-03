# Spatio-temporal data recovery module

## Installation

1. Install dependencies in your environment
```bash
pip install -r requirements.txt  # Python 3 is required
```

## Usage

1. Prepare data  

All models are supervised. So you need to prepare **X** (lats, lons, times) and **Y** (values) datasets. Firstly, you can download your dataset as a 3-dimensional tensor **T** with shape (lats, lons, times) and pass it to `tensor_utils.tensor2supervised` to generate X and Y.

2. Choose the model of your preference and use it like this
```python
# Example
from models.dineof3 import DINEOF3

d = DINEOF3(R=15, tensor_shape=(32, 32, 16))
d.fit(X_train, Y_train)
d.predict(X_test)
d.score(X_test, Y_test)
```

---
## Notes

You can also use these models with scikit-learn GridSearchCV to handle hyperparameters.
