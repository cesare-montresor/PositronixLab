# ModularCluster: Adaptive K-Means Clustering with Pub/Sub

A flexible, adaptive K-means clustering system supporting both synchronous and asynchronous (plugin-based) operation with multiple distance metrics.

## Quick Start

### Synchronous Clustering

```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

# Create data
data = torch.randn(1000, 128)

# Cluster with automatic K selection
kmeans = ModularClusterKmeans(metric='cosine')
labels = kmeans.fit_predict(data)

print(f"Optimal K: {kmeans.best_k}")
print(f"Quality: {kmeans.best_score:.4f}")
```

### Asynchronous Clustering with Pub/Sub

```python
from positronixlab.modularcluster.modularcluster_plugin import PluginModularCluster
from positronixlab.plugin.plugin import PluginManager

manager = PluginManager()
plugin = PluginModularCluster(metric='cosine')
manager.register_plugin(plugin, name="cluster")

# Publish embeddings
embeddings = torch.randn(500, 64)
manager.publish_to_topic("cu_positions", embeddings)

# Results arrive on "cu_labels" topic
```

## Features

- **Adaptive K Selection**: Automatically finds optimal number of clusters using silhouette score
- **Multiple Metrics**: Euclidean and cosine (spherical) distance metrics
- **Dual Operation Modes**: 
  - Synchronous (`ModularClusterKmeans`) for direct use
  - Asynchronous (`PluginModularCluster`) for plugin-based pipelines
- **Efficient**: FAISS-accelerated KMeans with optional GPU support
- **Zero-Copy Communication**: Uses shared memory for tensor transport

## Architecture

```
ModularClusterKmeans
    ↓ (async wrapper)
PluginModularCluster (Plugin)
    ↓ (subscribes to)
"cu_positions" topic
    ↓ (publishes to)
"cu_labels" topic
```

## Key Classes

### ModularClusterKmeans

Core synchronous clustering engine.

**Key Methods:**
- `fit(x)` - Fit model with adaptive K selection
- `predict(x)` - Predict labels for new data
- `fit_predict(x)` - Fit and return labels

**Key Attributes:**
- `best_k` - Optimal number of clusters
- `centroids` - Cluster centers
- `labels` - Cluster assignments
- `best_score` - Silhouette score

### PluginModularCluster

Asynchronous pub/sub wrapper.

**Topics:**
- Input: `"cu_positions"` (Tensor)
- Output: `"cu_labels"` (Tensor)

**Configuration:**
All parameters from `ModularClusterKmeans`, plus:
- `fatal_errors` - Whether to raise or log errors
- `name` - Plugin name

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest positronixlab/modularcluster/tests/ -v

# Just synchronous tests
pytest positronixlab/modularcluster/tests/test_kmeans.py -v

# Just plugin tests
pytest positronixlab/modularcluster/tests/test_kmeans_plugin.py -v

# Specific test
pytest positronixlab/modularcluster/tests/test_kmeans.py::TestModularClusterKmeans::test_euclidean_kmeans_basic -v
```

## Documentation

- [System Architecture](docs/MODULARCLUSTER_SYSTEM.md) - Overview, architecture, and detailed guide
- [API Reference](docs/MODULARCLUSTER_API.md) - Complete API documentation
- [Usage Guide](docs/MODULARCLUSTER_USAGE.md) - Practical examples and patterns

## Common Usage Patterns

### Pattern 1: Multi-Metric Comparison

```python
for metric in ['euclidean', 'cosine']:
    kmeans = ModularClusterKmeans(metric=metric)
    kmeans.fit(data)
    print(f"{metric}: K={kmeans.best_k}, Score={kmeans.best_score:.4f}")
```

### Pattern 2: Constrained K Range

```python
# Force clustering into 5-20 clusters
kmeans = ModularClusterKmeans(k_min=5, k_max=20)
labels = kmeans.fit_predict(data)
```

### Pattern 3: Reusable Model

```python
# Fit on training data
kmeans = ModularClusterKmeans()
kmeans.fit(train_data)

# Reuse for inference on batches
for batch in test_batches:
    labels = kmeans.predict(batch)  # Fast!
```

### Pattern 4: Quality-Based Decisions

```python
kmeans.fit(data)
if kmeans.best_score > 0.3:
    print("Good clustering quality")
else:
    print("Poor clustering, try different metric or data")
```

## Performance

Approximate times on CPU (Intel i7, 1000 samples, 128 dims):
- Euclidean: ~2.5s
- Cosine: ~2.8s

With GPU (NVIDIA V100):
- Euclidean: ~0.8s
- Cosine: ~0.9s

## Dependencies

- `torch` - Tensor operations
- `faiss` - Fast clustering (faiss-cpu or faiss-gpu)
- `scikit-learn` - Silhouette score calculation

Install:
```bash
pip install torch faiss-cpu scikit-learn
```

## Module Structure

```
positronixlab/modularcluster/
├── __init__.py
├── modularcluster_kmeans.py      # Core synchronous clustering
├── modularcluster_plugin.py       # Asynchronous plugin wrapper
├── tests/
│   ├── test_kmeans.py           # 12 tests for synchronous API
│   └── test_kmeans_plugin.py     # 14 tests for plugin API
└── docs/
    ├── MODULARCLUSTER_SYSTEM.md  # System overview and architecture
    ├── MODULARCLUSTER_API.md     # Complete API reference
    └── MODULARCLUSTER_USAGE.md   # Practical examples and recipes
```

## Troubleshooting

### Poor Clustering Quality

**Symptoms**: Silhouette score < -0.2

**Solutions**:
1. Normalize embeddings for cosine metric
2. Try different metric (euclidean vs cosine)
3. Verify data is appropriate for clustering
4. Widen K range to find better peaks

### K Selection Seems Wrong

**Solution**: Inspect `silhouette_scores` dict:
```python
for k, score in sorted(kmeans.silhouette_scores.items()):
    print(f"K={k}: {score:.4f}")
```

### Plugin Not Receiving Data

**Checklist**:
1. ✓ Plugin registered with manager
2. ✓ Topic names match: "cu_positions" and "cu_labels"
3. ✓ Plugin started before publishing
4. ✓ Data format is Tensor

## Examples

See [MODULARCLUSTER_USAGE.md](docs/MODULARCLUSTER_USAGE.md) for:
- Document embedding clustering
- Real-time image feature clustering
- Interactive model refinement
- Multi-metric comparison
- Save/load clustering models
- Dimensionality-aware K selection
- Confidence-based decision making
- Adaptive batching
- ML pipeline integration
- Streaming clustering

## License

Part of PositronixLab project.

## Related Documentation

- [Plugin System](../plugin/docs/PLUGIN_SYSTEM.md) - Pub/Sub architecture
- [Common Utils](../common/) - Utility functions
