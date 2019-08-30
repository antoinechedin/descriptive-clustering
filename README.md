# Descriptive Clustering
Descriptive clustering or DesC is a parameter free clustering algorithm for
data description written in Python.

DesC is based on Scipy hierarchical clustering algortithm for cluster creation
and on knee detection algorithm for automatically choosing the number of
clusters to build.

## DesC usage
DesC class extends the default Scikit learn model, so it's fully compatible
with all the Scikit learn tools.
```python
from clustering import DesC
import numpy as np
X = np.array([[1, 3], [2, 3], [1, 0], [5, 2], [5, 3]])
desc = DesC().fit(X)

desc.K_
desc.eval_graph_
desc.labels_
```

Output
```python
Out[1]: 3
Out[2]: array([[1., 3.],
               [2., 3.],
               [3., 1.],
               [4., 1.],
               [5., 0.]])
Out[3]: array([2, 2, 3, 1, 1], dtype=int32)
```

## Knee detection usage
Knee detection module provide a list of functions to find knee point in a
curve. Just import the module to use them

```python
from clustering import kneedetector
import numpy as np
X = np.array([[0, 8], [1, 4], [2, 2], [4, 1], [8, 0]])

kneedetector.kneedle_scan(X)
kneedetector.l_method_scan(X)
kneedetector.max_amplitude_scan(X)
```

Output
```python
Out[1]: array([2, 2])
Out[2]: array([2, 2])
Out[3]: array([1, 4])
```
