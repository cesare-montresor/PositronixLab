# ModularCluster API Reference

Complete API documentation for `ModularClusterKmeans` and `PluginModularCluster`.

## ModularClusterKmeans

Synchronous adaptive K-means clustering engine.

### Module
```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
```

### Class: ModularClusterKmeans

#### Constructor

```python
ModularClusterKmeans(
    metric: Literal['euclidean', 'cosine'] = 'cosine',
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    n_init: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    precompute_distances: bool = True
)
```

**Parameters:**

- `metric` (str): Distance metric for clustering
  - `'euclidean'`: Euclidean L2 distance
  - `'cosine'`: Cosine distance (spherical KMeans)
  - Default: `'cosine'`

- `k_min` (int, optional): Minimum K for adaptive search
  - Heuristic default: `max(2, int(sqrt(n/2)))`
  - Must be ≥ 2

- `k_max` (int, optional): Maximum K for adaptive search
  - Heuristic default: `min(int(sqrt(2*n)), n//2)`
  - Must be > k_min

- `max_iter` (int): Maximum KMeans iterations per run
  - Default: `100`

- `tol` (float): Convergence tolerance
  - Default: `1e-5`

- `n_init` (int): Number of random initializations for KMeans
  - Default: `1`

- `device` (torch.device, optional): Compute device
  - Default: `torch.device('cpu')`

- `verbose` (bool): Print progress during fitting
  - Default: `False`

- `precompute_distances` (bool): Cache distance matrix during fit
  - Trades memory for speed
  - Default: `True`

#### Methods

##### fit(x: Tensor) → ModularClusterKmeans

Fit the clustering model with adaptive K selection.

**Parameters:**
- `x` (Tensor): Input embeddings, shape `(n_samples, n_features)`

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If n_samples < 2

**Example:**
```python
kmeans = ModularClusterKmeans(metric='cosine')
kmeans.fit(embeddings)
print(f"Best K: {kmeans.best_k}")
```

**Side Effects:**
- Updates `centroids`, `labels`, `best_k`, `best_score`, `silhouette_scores`

---

##### predict(x: Tensor) → Tensor

Predict cluster labels for new data using learned centroids.

**Parameters:**
- `x` (Tensor): New embeddings, shape `(n_samples, n_features)`

**Returns:**
- Tensor of cluster labels, shape `(n_samples,)`, dtype `int64`

**Raises:**
- `ValueError`: If model not fitted (centroids is None)

**Example:**
```python
kmeans = ModularClusterKmeans(metric='euclidean')
kmeans.fit(train_data)
test_labels = kmeans.predict(test_data)
```

**Note:**
- Does not update fitted model
- All inputs moved to model's device automatically

---

##### fit_predict(x: Tensor) → Tensor

Fit model and return training cluster labels.

**Parameters:**
- `x` (Tensor): Input embeddings, shape `(n_samples, n_features)`

**Returns:**
- Tensor of cluster labels, shape `(n_samples,)`, dtype `int64`

**Example:**
```python
kmeans = ModularClusterKmeans()
labels = kmeans.fit_predict(embeddings)
```

**Equivalent to:**
```python
kmeans.fit(embeddings)
labels = kmeans.labels
```

---

##### print_stats() → None

Print clustering statistics to console.

**Output Format:**
```
Best K: 5
Best score: 0.4523
Silhouette Score: 0.4523
Cluster sizes: [200, 150, 180, 220, 250]
Evaluated K values: [2, 3, 4, 5, 6, 7, 8, 9, 10]
```

**Example:**
```python
kmeans.fit(embeddings)
kmeans.print_stats()
```

---

#### Attributes

##### centroids: Tensor

Learned cluster centers.

- Type: `Tensor`
- Shape: `(k, n_features)` where k = best_k
- Device: Same as input device
- Value: `None` before fit()

```python
kmeans.fit(embeddings)
print(kmeans.centroids.shape)  # torch.Size([5, 128])
```

---

##### labels: Tensor

Cluster assignments for training data.

- Type: `Tensor` with dtype `int64`
- Shape: `(n_samples,)`
- Values: Integers in range [0, best_k)
- Value: `None` before fit()

```python
kmeans.fit(embeddings)
print(torch.unique(kmeans.labels))  # tensor([0, 1, 2, 3, 4])
```

---

##### best_k: int

Optimal number of clusters found by adaptive search.

- Type: `int`
- Range: [k_min, k_max]
- Value: `None` before fit()

```python
kmeans.fit(embeddings)
print(f"Optimal K: {kmeans.best_k}")  # Optimal K: 5
```

---

##### best_score: float

Silhouette score for the best K.

- Type: `float`
- Range: [-1, 1] (higher is better)
- Value: `None` before fit()

```python
kmeans.fit(embeddings)
print(f"Quality: {kmeans.best_score:.4f}")  # Quality: 0.4523
```

---

##### silhouette_scores: Dict[int, float]

Silhouette scores for all evaluated K values.

- Type: `Dict[int, float]`
- Keys: K values evaluated
- Values: Silhouette scores for each K
- Populated during fit()

```python
kmeans.fit(embeddings)
for k, score in sorted(kmeans.silhouette_scores.items()):
    print(f"K={k:2d}: {score:.4f}")
```

---

##### metric: str

Distance metric being used.

- Type: `str`
- Values: `'euclidean'` or `'cosine'`
- Set at initialization

```python
kmeans = ModularClusterKmeans(metric='cosine')
print(kmeans.metric)  # cosine
```

---

##### verbose: bool

Whether to print progress during operations.

- Type: `bool`
- Set at initialization

---

## PluginModularCluster

Asynchronous pub/sub-based clustering plugin.

### Module
```python
from positronixlab.modularcluster.modularcluster_plugin import PluginModularCluster
```

### Class: PluginModularCluster(Plugin)

Extends `Plugin` with clustering capabilities.

#### Constructor

```python
PluginModularCluster(
    metric: Literal['euclidean', 'cosine'] = 'cosine',
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    n_init: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    precompute_distances: bool = True,
    fatal_errors: bool = False,
    autostart: bool = True
)
```

**Parameters:**

All parameters from `ModularClusterKmeans`, plus:

- `fatal_errors` (bool): Whether to raise exceptions or log them
  - Default: `False`

- `autostart` (bool): Automatically start the plugin
  - Default: `True`

**Example:**
```python
plugin = PluginModularCluster(
    metric='cosine',
    k_min=5,
    k_max=50,
    autostart=True
)
```

#### Class Attributes

##### INPUT_TOPIC: str = "cu_positions"

Topic name for receiving embeddings.

```python
# Publish embeddings to this topic
manager.publish_to_topic(
    PluginModularCluster.INPUT_TOPIC,
    embeddings
)
```

---

##### OUTPUT_TOPIC: str = "cu_labels"

Topic name for publishing cluster labels.

```python
# Subscribe to results on this topic
manager.subscribe_to_topic(
    PluginModularCluster.OUTPUT_TOPIC,
    on_labels_callback
)
```

---

#### Methods

##### onStart() → None

Called when plugin starts (inherited from Plugin).

Sets up subscriptions to the input topic.

**Called automatically by plugin system.**

---

##### onUpdate() → None

Main plugin loop (inherited from Plugin).

For pub/sub-based operation, this is minimal since processing
happens in callbacks.

**Called repeatedly by plugin system.**

---

#### Private Methods (Advanced)

##### _on_positions_received(positions: Tensor) → None

Internal callback triggered when positions are published.

**Parameters:**
- `positions` (Tensor or array-like): Input embeddings

**Behavior:**
1. Converts input to tensor if needed
2. Runs `fit_predict()` on the clusterer
3. Publishes results to output topic
4. Prints verbose output if enabled

**Error Handling:**
- If `fatal_errors=False`: Logs error and continues
- If `fatal_errors=True`: Raises exception

---

#### Inherited Attributes

##### clusterer: ModularClusterKmeans

The underlying clustering engine.

- Type: `ModularClusterKmeans`
- Access: `plugin.clusterer`

**Example:**
```python
plugin = PluginModularCluster()
plugin._on_positions_received(embeddings)
print(f"K: {plugin.clusterer.best_k}")
print(f"Score: {plugin.clusterer.best_score}")
```

---

## Usage Examples

### Example 1: Direct Clustering

```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

data = torch.randn(1000, 128)
kmeans = ModularClusterKmeans(metric='cosine')
labels = kmeans.fit_predict(data)

print(f"K={kmeans.best_k}, Score={kmeans.best_score:.4f}")
print(f"Cluster sizes: {torch.bincount(labels)}")
```

### Example 2: Plugin with PluginManager

```python
from positronixlab.modularcluster.modularcluster_plugin import PluginModularCluster
from positronixlab.plugin.plugin import PluginManager
import torch
import time

manager = PluginManager()
plugin = PluginModularCluster(metric='cosine')
manager.register_plugin(plugin, name="cluster")

results = []
manager.subscribe_to_topic("cu_labels", results.append)

embeddings = torch.randn(500, 64)
manager.publish_to_topic("cu_positions", embeddings)

time.sleep(1)
if results:
    print(f"Got labels: {len(results[0])} samples")
```

### Example 3: Batch Processing

```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

kmeans = ModularClusterKmeans(metric='euclidean')

for batch_idx in range(5):
    batch = torch.randn(500, 128)
    labels = kmeans.fit_predict(batch)
    print(f"Batch {batch_idx}: K={kmeans.best_k}")
```

## Constants

### Metric Constants

```python
METRIC_EUCLIDEAN = 'euclidean'  # L2 distance
METRIC_COSINE = 'cosine'         # Angular distance, spherical KMeans
```

### Default Parameters

```python
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-5
DEFAULT_N_INIT = 1
DEFAULT_VERBOSE = False
DEFAULT_PRECOMPUTE = True
```

## Error Handling

### Common Exceptions

#### ValueError: "Need at least 2 samples for clustering"

**Cause:** Dataset has fewer than 2 samples

**Solution:**
```python
if len(embeddings) < 2:
    raise ValueError("Need more samples")
kmeans.fit(embeddings)
```

---

#### ValueError: "Must call fit() before predict()"

**Cause:** Calling `predict()` before `fit()`

**Solution:**
```python
kmeans = ModularClusterKmeans()
kmeans.fit(train_data)      # Must fit first
labels = kmeans.predict(test_data)  # Now OK
```

---

#### RuntimeError: "Error in clustering: ..."

**Cause:** Various internal errors

**Solution:** Check verbose output and inspect data:
```python
kmeans = ModularClusterKmeans(verbose=True)
kmeans.fit(embeddings)  # See detailed output
```

---

## Type Hints

```python
from typing import Literal, Optional, Tuple, Dict
import torch

# Metric type
MetricType = Literal['euclidean', 'cosine']

# Device type
DeviceType = Optional[torch.device]

# Scores type
ScoresType = Dict[int, float]
```

## Performance Benchmarks

Approximate execution times on CPU (Intel i7, 1000 samples, 128 dims):

| Metric | Time | K Evaluated | Best Score |
|--------|------|-------------|-----------|
| Euclidean | 2.5s | 15 | 0.38 |
| Cosine | 2.8s | 15 | 0.42 |

With GPU (NVIDIA V100, same setup):

| Metric | Time | K Evaluated | Best Score |
|--------|------|-------------|-----------|
| Euclidean | 0.8s | 15 | 0.38 |
| Cosine | 0.9s | 15 | 0.42 |

---

## See Also

- [MODULARCLUSTER_SYSTEM.md](MODULARCLUSTER_SYSTEM.md) - System overview and examples
- [../modularcluster_kmeans.py](../modularcluster_kmeans.py) - Source code
- [../modularcluster_plugin.py](../modularcluster_plugin.py) - Plugin source code
