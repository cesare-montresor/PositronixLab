# ModularCluster System

A flexible, adaptive K-means clustering system with support for multiple distance metrics and both synchronous and asynchronous (plugin-based) operation.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Usage Patterns](#usage-patterns)
- [Configuration](#configuration)
- [Examples](#examples)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Features

### Adaptive K Selection
- **Automatic K Discovery**: Uses silhouette score optimization with adaptive search
- **Heuristic K Range**: Automatically computes reasonable K bounds based on dataset size
- **Intelligent Sampling**: Three-phase search (coarse sampling → peak detection → refinement)

### Distance Metrics
- **Euclidean Distance**: Standard L2 distance for general-purpose clustering
- **Cosine Distance**: Spherical KMeans for normalized embeddings and angular similarity

### Dual Operation Modes
- **Synchronous**: `ModularClusterKmeans` for direct/blocking use
- **Asynchronous**: `PluginModularCluster` for non-blocking pub/sub communication

### Efficient Tensor Processing
- **Zero-Copy Communication**: Plugin system uses shared memory for tensors
- **GPU Support**: Optional CUDA device support for faster computation
- **Distance Caching**: Optional precomputation of distance matrices

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ModularClusterKmeans                       │
│        (Core synchronous clustering engine)             │
│  - Adaptive K selection via silhouette score            │
│  - Support for euclidean/cosine metrics                 │
│  - FAISS-based acceleration                             │
└─────────────────────────────────────────────────────────┘
                        ▲
                        │ inherits
                        │
┌─────────────────────────────────────────────────────────┐
│           PluginModularCluster(Plugin)                  │
│        (Asynchronous pub/sub wrapper)                   │
│  - Subscribes to "cu_positions" topic                   │
│  - Publishes to "cu_labels" topic                       │
│  - Event-driven clustering via callbacks                │
└─────────────────────────────────────────────────────────┘
```

### Data Flow (Plugin-Based)

```
External System
      │
      │ publish("cu_positions", embeddings)
      │
      ▼
┌─────────────────────────────────────────┐
│     PluginManager (pubsub broker)       │
└─────────────────────────────────────────┘
      │
      │ route to subscribers
      │
      ▼
┌─────────────────────────────────────────┐
│   PluginModularCluster._on_positions    │
│   (callback triggered)                   │
└─────────────────────────────────────────┘
      │
      │ fit_predict(embeddings)
      │
      ▼
┌─────────────────────────────────────────┐
│    ModularClusterKmeans                 │
│    (core algorithm)                     │
└─────────────────────────────────────────┘
      │
      │ returns labels
      │
      ▼
┌─────────────────────────────────────────┐
│    PluginModularCluster.publish()       │
│    ("cu_labels", labels)                │
└─────────────────────────────────────────┘
      │
      │ publish to topic
      │
      ▼
   External System
```

## Core Components

### ModularClusterKmeans

Synchronous clustering engine using adaptive K selection.

**Constructor Parameters:**

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

**Key Methods:**

- `fit(x: Tensor) → ModularClusterKmeans`: Fit model with adaptive K selection
- `predict(x: Tensor) → Tensor`: Predict cluster labels for new data
- `fit_predict(x: Tensor) → Tensor`: Fit and return labels in one call
- `print_stats()`: Print clustering statistics to console

**Key Attributes:**

- `centroids`: Cluster centers (shape: [k, n_features])
- `labels`: Cluster assignments for training data
- `best_k`: Optimal number of clusters found
- `best_score`: Silhouette score for best K
- `silhouette_scores`: Dict of {k: score} for all evaluated K values

### PluginModularCluster

Asynchronous pub/sub wrapper for clustering via the plugin system.

**Constructor Parameters:**

All parameters from `ModularClusterKmeans`, plus:

- `fatal_errors: bool = False`: Whether to raise or log errors
- `autostart: bool = True`: Automatically start the plugin

**Topics:**

- **Input**: `"cu_positions"` - Receives embeddings (Tensor)
- **Output**: `"cu_labels"` - Publishes cluster labels (Tensor)

**Example:**

```python
from positronixlab.plugin.plugin import PluginManager
from positronixlab.modularcluster.modularcluster_plugin import PluginModularCluster

manager = PluginManager()

# Create and register clustering plugin
cluster_plugin = PluginModularCluster(
    metric='cosine',
    k_min=5,
    k_max=50
)
manager.register_plugin(cluster_plugin, name="clustering")

# Subscribe to results
def on_labels(labels):
    print(f"Received {len(labels)} cluster labels")

manager.subscribe_to_topic("cu_labels", on_labels)

# Publish embeddings for clustering
embeddings = torch.randn(1000, 128)
manager.publish_to_topic("cu_positions", embeddings)

# Results arrive asynchronously via callback
```

## Usage Patterns

### Pattern 1: Simple Synchronous Clustering

```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

# Generate data
embeddings = torch.randn(1000, 128)

# Create clusterer
kmeans = ModularClusterKmeans(metric='cosine', verbose=True)

# Cluster
labels = kmeans.fit_predict(embeddings)

# Inspect results
print(f"K = {kmeans.best_k}")
print(f"Silhouette Score = {kmeans.best_score:.4f}")
print(f"Cluster sizes: {torch.bincount(labels)}")
```

### Pattern 2: Async Clustering with Plugin System

```python
from positronixlab.modularcluster.modularcluster_plugin import PluginModularCluster
from positronixlab.plugin.plugin import PluginManager
import torch
import time

manager = PluginManager()

# Create plugin
cluster_plugin = PluginModularCluster(metric='cosine')
manager.register_plugin(cluster_plugin, name="cluster")

# Collect results
results = []
manager.subscribe_to_topic("cu_labels", lambda labels: results.append(labels))

# Publish data for clustering
embeddings = torch.randn(500, 64)
manager.publish_to_topic("cu_positions", embeddings)

# Wait for result
time.sleep(1)

if results:
    print(f"Clustered into {len(torch.unique(results[0]))} clusters")
```

### Pattern 3: Multiple Metrics Comparison

```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

embeddings = torch.randn(1000, 128)

# Compare metrics
for metric in ['euclidean', 'cosine']:
    kmeans = ModularClusterKmeans(metric=metric, verbose=False)
    kmeans.fit(embeddings)
    
    print(f"\n{metric.upper()}")
    print(f"  K = {kmeans.best_k}")
    print(f"  Score = {kmeans.best_score:.4f}")
```

### Pattern 4: Constrained K Range

```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

embeddings = torch.randn(1000, 128)

# Force clustering into 10-20 clusters
kmeans = ModularClusterKmeans(
    metric='cosine',
    k_min=10,
    k_max=20,
    verbose=True
)
labels = kmeans.fit_predict(embeddings)

# best_k will be in range [10, 20]
print(f"Optimal K in [10, 20]: {kmeans.best_k}")
```

## Configuration

### Metric Selection

| Metric | Use Case | Properties |
|--------|----------|-----------|
| `'euclidean'` | General-purpose clustering | Works with any embeddings, sensitive to scale |
| `'cosine'` | Angular similarity | Requires normalized embeddings, spherical KMeans |

### K Range Heuristics

The system computes a default K range based on dataset size:

```
k_min = max(2, sqrt(n/2))      # Lower bound
k_max = min(sqrt(2n), n/2)     # Upper bound
```

For a 1000-sample dataset:
- k_min ≈ 22
- k_max ≈ 45

Override with explicit `k_min` and `k_max` parameters.

### Adaptive Search Phases

**Phase 1: Coarse Sampling** (~10 K values)
- Sample evenly across k_min to k_max
- Evaluate silhouette score for each

**Phase 2: Peak Detection**
- Find local maxima in the silhouette curve
- Identifies candidate optimal K regions

**Phase 3: Refinement**
- Fine-grained search around each peak
- 3 values on each side of peak

### Performance Tuning

```python
# Fast mode (fewer K evaluations)
kmeans = ModularClusterKmeans(
    k_min=10, k_max=30,  # Narrow range
    verbose=False,
    precompute_distances=False  # Skip distance matrix cache
)

# Accurate mode (more K evaluations)
kmeans = ModularClusterKmeans(
    k_min=2, k_max=100,  # Wide range
    verbose=True,
    precompute_distances=True  # Cache distances
)
```

## Examples

### Example 1: MNIST Embeddings

```python
import torch
from torchvision import datasets, transforms
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans

# Load MNIST and extract embeddings (simplified)
dataset = datasets.MNIST(root='/tmp', download=True)
images = dataset.data.float() / 255.0
embeddings = images.reshape(len(images), -1)  # Flatten

# Cluster
kmeans = ModularClusterKmeans(metric='euclidean', verbose=True)
labels = kmeans.fit_predict(embeddings[:5000])

print(f"Discovered {kmeans.best_k} digit clusters")
print(f"Quality: {kmeans.best_score:.4f}")
```

### Example 2: Real-time Clustering Pipeline

```python
from positronixlab.modularcluster.modularcluster_plugin import PluginModularCluster
from positronixlab.plugin.plugin import PluginManager
import torch
import time

manager = PluginManager()

# Clustering plugin
cluster_plugin = PluginModularCluster(metric='cosine')
manager.register_plugin(cluster_plugin, name="cluster")

# Accumulate statistics
stats = {"total_processed": 0, "total_clusters": 0}

def process_results(labels):
    stats["total_processed"] += len(labels)
    stats["total_clusters"] += len(torch.unique(labels))
    print(f"Processed: {stats['total_processed']}, "
          f"Avg clusters: {stats['total_clusters'] / (stats['total_processed']/100):.1f}")

manager.subscribe_to_topic("cu_labels", process_results)

# Simulate data stream
for batch_idx in range(10):
    batch = torch.randn(100, 64)
    manager.publish_to_topic("cu_positions", batch)
    time.sleep(0.5)
```

### Example 3: Batch Processing with Caching

```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

# Load all data
all_embeddings = torch.randn(10000, 128)

# Create persistent clusterer (maintains learned K)
kmeans = ModularClusterKmeans(metric='cosine', verbose=True)

# Fit on full dataset
kmeans.fit(all_embeddings)
print(f"Learned K = {kmeans.best_k}")

# Now use it to cluster batches
for i in range(0, len(all_embeddings), 1000):
    batch = all_embeddings[i:i+1000]
    batch_labels = kmeans.predict(batch)
    print(f"Batch {i//1000}: {torch.bincount(batch_labels)}")
```

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Single K evaluation | O(n * k * d * iter) | n=samples, k=clusters, d=dims, iter=KMeans iterations |
| Adaptive search | O(m * n * k * d * iter) | m ≈ 20 K values evaluated |
| Silhouette score | O(n²) | Quadratic in samples (can use distance cache) |

### Memory Usage

```python
# Without precomputation
memory = n * d * sizeof(float32)  # Just the data

# With precomputation
memory = n * d * sizeof(float32) + n² * sizeof(float32)  # Large for big n
```

For n=10000, d=128:
- Data: 5 MB
- Distance cache: 400 MB

### Optimization Tips

1. **Large Datasets**: Use `precompute_distances=False`
2. **Narrow K Range**: If you know approximate optimal K, set k_min/k_max to reduce search
3. **GPU Acceleration**: Pass `device=torch.device('cuda')`
4. **Batch Reuse**: Fit once, predict multiple times with learned centroids

## Troubleshooting

### Issue: Poor Clustering Quality (Low Silhouette Score)

**Symptoms**: Best silhouette score < -0.2

**Solutions**:
1. Check data normalization (for cosine metric, normalize before passing)
2. Widen K range to find better peaks
3. Try different metric (euclidean vs cosine)
4. Verify data is appropriate for clustering

```python
# Normalize for cosine
import torch.nn.functional as F
normalized = F.normalize(embeddings, p=2, dim=1)
kmeans = ModularClusterKmeans(metric='cosine')
labels = kmeans.fit_predict(normalized)
```

### Issue: K Selection Seems Wrong

**Symptoms**: best_k is too small/large

**Solutions**:
1. Check silhouette_scores dict - plot the curve
2. Manually override with k_min/k_max if you know better
3. Verify n_features > 0 and n_samples >= 2

```python
# Inspect all scores
kmeans.fit(embeddings)
for k, score in sorted(kmeans.silhouette_scores.items()):
    print(f"K={k:3d}: {score:.4f}")
```

### Issue: Plugin Not Receiving Results

**Symptoms**: Callback never called, no results

**Solutions**:
1. Verify plugin is registered before publishing
2. Topic names must match exactly: "cu_positions" and "cu_labels"
3. Add slight delay before checking results
4. Enable verbose mode to see debug output

```python
# Debug version
cluster_plugin = PluginModularCluster(verbose=True)
manager.register_plugin(cluster_plugin, "cluster")

# Give it time
manager.publish_to_topic("cu_positions", embeddings)
time.sleep(2)  # Wait for async processing
```

### Issue: Out of Memory

**Symptoms**: CUDA/CPU out of memory during precomputation

**Solutions**:
1. Disable distance precomputation:
```python
kmeans = ModularClusterKmeans(precompute_distances=False)
```
2. Reduce batch size
3. Use GPU with more VRAM
4. Process in smaller chunks, then merge results

## Testing

Run the comprehensive test suite:

```bash
# Synchronous tests
pytest positronixlab/modularcluster/tests/test_kmeans.py -v

# Plugin tests
pytest positronixlab/modularcluster/tests/test_kmeans_plugin.py -v

# Quick benchmarks
python -m pytest positronixlab/modularcluster/tests/test_kmeans.py::test_sync_euclidean -v
python -m pytest positronixlab/modularcluster/tests/test_kmeans.py::test_sync_cosine -v
```

## References

- **Silhouette Score**: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- **FAISS KMeans**: https://github.com/facebookresearch/faiss
- **KMeans Algorithm**: Lloyd, S. (1982). "Least squares quantization in PCM"
