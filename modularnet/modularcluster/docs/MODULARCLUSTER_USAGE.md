# ModularCluster Usage Guide

Practical guide with step-by-step examples for using the ModularCluster system.

## Table of Contents

- [Quick Start](#quick-start)
- [Synchronous vs Asynchronous](#synchronous-vs-asynchronous)
- [Practical Examples](#practical-examples)
- [Common Patterns](#common-patterns)
- [Integration Recipes](#integration-recipes)

## Quick Start

### Installation

ModularCluster is included in ModularNet. Ensure dependencies are installed:

```bash
pip install torch faiss-cpu scikit-learn
# Or for GPU:
pip install torch faiss-gpu scikit-learn
```

### Minimal Example

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

# Create sample data
embeddings = torch.randn(1000, 128)

# Cluster
kmeans = ModularClusterKmeans(metric='cosine')
labels = kmeans.fit_predict(embeddings)

# Results
print(f"Found {kmeans.best_k} clusters")
print(f"Quality score: {kmeans.best_score:.4f}")
```

## Synchronous vs Asynchronous

### Synchronous (ModularClusterKmeans)

Use when you need clustering results immediately.

```python
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import torch

kmeans = ModularClusterKmeans(metric='cosine')
labels = kmeans.fit_predict(embeddings)
# Results available immediately
print(labels)
```

**Advantages:**
- Simple API
- Results immediately
- Full control over execution
- Easy debugging

**Disadvantages:**
- Blocks execution
- No parallelism with other work
- Inefficient in event-driven systems

### Asynchronous (PluginModularCluster)

Use when clustering should run without blocking main thread.

```python
from modularnet.modularcluster.modularcluster_plugin import PluginModularCluster
from modularnet.plugin.plugin import PluginManager
import torch
import time

manager = PluginManager()
plugin = PluginModularCluster(metric='cosine')
manager.register_plugin(plugin, name="cluster")

# Set up callback
def on_labels(labels):
    print(f"Results ready: {labels.shape}")

manager.subscribe_to_topic("cu_labels", on_labels)

# Publish data
manager.publish_to_topic("cu_positions", embeddings)

# Main thread continues...
time.sleep(5)  # Results arrive asynchronously
```

**Advantages:**
- Non-blocking
- Can process multiple batches
- Integrates with plugin ecosystem
- Great for pipelines

**Disadvantages:**
- Requires plugin system setup
- Results arrive asynchronously
- Slightly more complex

## Practical Examples

### Example 1: Clustering Document Embeddings

Cluster text documents based on pre-computed embeddings.

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

# Assume we have 10,000 document embeddings (e.g., from BERT)
num_docs = 10000
embedding_dim = 768
doc_embeddings = torch.randn(num_docs, embedding_dim)

# Normalize for cosine similarity (important!)
import torch.nn.functional as F
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

# Cluster
kmeans = ModularClusterKmeans(metric='cosine', verbose=True)
doc_labels = kmeans.fit_predict(doc_embeddings)

# Results
print(f"\nFound {kmeans.best_k} document clusters")
print(f"Cluster quality: {kmeans.best_score:.4f}")

# Show cluster distribution
cluster_sizes = torch.bincount(doc_labels)
print(f"\nCluster sizes: {cluster_sizes}")

# Assign documents to topics
topic_assignments = {i: [] for i in range(kmeans.best_k)}
for doc_id, topic_id in enumerate(doc_labels.tolist()):
    topic_assignments[topic_id].append(doc_id)

print(f"\nTopic 0 contains: {topic_assignments[0][:5]}... ({len(topic_assignments[0])} docs)")
```

### Example 2: Real-Time Image Feature Clustering

Use plugin system to cluster CNN features in real-time.

```python
import torch
from torchvision import models, transforms
from modularnet.modularcluster.modularcluster_plugin import PluginModularCluster
from modularnet.plugin.plugin import PluginManager
import time

# Setup feature extractor
model = models.resnet50(pretrained=True)
model.eval()

# Setup clustering plugin
manager = PluginManager()
cluster_plugin = PluginModularCluster(
    metric='cosine',
    k_min=5,
    k_max=50,
    verbose=False
)
manager.register_plugin(cluster_plugin, name="feature_clusterer")

# Collect results
cluster_results = []

def on_cluster_results(labels):
    cluster_results.append({
        'timestamp': time.time(),
        'labels': labels,
        'k': len(torch.unique(labels))
    })
    print(f"Clustered into {cluster_results[-1]['k']} groups")

manager.subscribe_to_topic("cu_labels", on_cluster_results)

# Simulate image stream
print("Processing image stream...")
for batch_idx in range(5):
    # In real system: load images, extract features
    image_features = torch.randn(256, 2048)  # ResNet50 features
    
    # Publish for clustering
    manager.publish_to_topic("cu_positions", image_features)
    print(f"  Batch {batch_idx}: submitted")
    
    # Do other work while clustering happens
    time.sleep(0.5)

print(f"\nProcessed {len(cluster_results)} batches")
```

### Example 3: Interactive Model - Refine Clusters

Explore different K values interactively.

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

embeddings = torch.randn(1000, 128)

# First: find the adaptive K
kmeans_auto = ModularClusterKmeans(metric='cosine', verbose=False)
kmeans_auto.fit(embeddings)

print("Automatic K selection:")
print(f"  Best K: {kmeans_auto.best_k}")
print(f"  Best score: {kmeans_auto.best_score:.4f}")

# Show silhouette scores for all K
print("\nAll evaluated K values:")
for k in sorted(kmeans_auto.silhouette_scores.keys()):
    score = kmeans_auto.silhouette_scores[k]
    bar = "█" * int(score * 20)  # Visual bar
    print(f"  K={k:3d}: {score:.4f}  {bar}")

# Interactive refinement
user_k = kmeans_auto.best_k  # Or user could select

# Force clustering with specific K
kmeans_fixed = ModularClusterKmeans(
    metric='cosine',
    k_min=user_k,
    k_max=user_k,
    verbose=False
)
labels = kmeans_fixed.fit_predict(embeddings)

print(f"\nForced clustering with K={user_k}:")
print(f"  Cluster sizes: {torch.bincount(labels)}")
```

### Example 4: Multi-Metric Comparison

Compare clustering quality across different distance metrics.

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans
import time

embeddings = torch.randn(2000, 256)

results = {}

for metric in ['euclidean', 'cosine']:
    print(f"\n{'='*40}")
    print(f"Clustering with {metric.upper()} metric")
    print('='*40)
    
    start = time.time()
    kmeans = ModularClusterKmeans(metric=metric, verbose=False)
    labels = kmeans.fit_predict(embeddings)
    elapsed = time.time() - start
    
    results[metric] = {
        'k': kmeans.best_k,
        'score': kmeans.best_score,
        'time': elapsed,
        'sizes': torch.bincount(labels).tolist()
    }
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Best K: {kmeans.best_k}")
    print(f"Silhouette: {kmeans.best_score:.4f}")
    print(f"Sizes: {results[metric]['sizes']}")

# Summary
print(f"\n{'='*40}")
print("SUMMARY")
print('='*40)
best_metric = max(results, key=lambda m: results[m]['score'])
print(f"Best metric: {best_metric}")
print(f"  K: {results[best_metric]['k']}")
print(f"  Score: {results[best_metric]['score']:.4f}")
print(f"  Time: {results[best_metric]['time']:.2f}s")
```

### Example 5: Save/Load Fitted Model

Serialize and reuse a fitted clustering model.

```python
import torch
import pickle
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

# Fit model
embeddings = torch.randn(1000, 128)
kmeans = ModularClusterKmeans(metric='cosine')
kmeans.fit(embeddings)

# Save centroids and metadata
checkpoint = {
    'centroids': kmeans.centroids.cpu(),
    'best_k': kmeans.best_k,
    'best_score': kmeans.best_score,
    'metric': kmeans.metric,
    'silhouette_scores': kmeans.silhouette_scores
}

# Save to file
torch.save(checkpoint, 'clustering_model.pt')
print(f"Model saved: K={kmeans.best_k}, Score={kmeans.best_score:.4f}")

# Later: load and use
loaded = torch.load('clustering_model.pt')
kmeans_loaded = ModularClusterKmeans(
    metric=loaded['metric'],
    device=torch.device('cpu')
)
kmeans_loaded.centroids = loaded['centroids']
kmeans_loaded.best_k = loaded['best_k']
kmeans_loaded.best_score = loaded['best_score']

# Now predict on new data
new_data = torch.randn(500, 128)
predictions = kmeans_loaded.predict(new_data)
print(f"Predicted {len(predictions)} labels using loaded model")
```

## Common Patterns

### Pattern 1: Dimensionality-Aware K Selection

Choose K range based on data dimensionality.

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

def smart_cluster(embeddings, n_features_hint=None):
    """Cluster with intelligent K range based on dimensionality."""
    n_samples, n_dims = embeddings.shape
    
    if n_dims > 500:
        # High-dim: more clusters
        k_min, k_max = 10, 100
    elif n_dims > 100:
        # Medium-dim: moderate clusters
        k_min, k_max = 5, 50
    else:
        # Low-dim: fewer clusters
        k_min, k_max = 2, 20
    
    print(f"Data shape: {embeddings.shape}")
    print(f"K range: [{k_min}, {k_max}]")
    
    kmeans = ModularClusterKmeans(
        metric='cosine',
        k_min=k_min,
        k_max=k_max,
        verbose=True
    )
    return kmeans.fit_predict(embeddings), kmeans

# Try it
data = torch.randn(1000, 512)
labels, kmeans = smart_cluster(data)
```

### Pattern 2: Confidence-Based Decision

Make decisions based on clustering quality.

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

def cluster_with_confidence(embeddings, min_quality=0.3):
    """Cluster and decide if quality is acceptable."""
    kmeans = ModularClusterKmeans(metric='cosine', verbose=False)
    labels = kmeans.fit_predict(embeddings)
    
    confidence = kmeans.best_score
    
    if confidence >= min_quality:
        return labels, "HIGH_QUALITY"
    else:
        return labels, "LOW_QUALITY"

# Usage
embeddings = torch.randn(1000, 128)
labels, quality = cluster_with_confidence(embeddings, min_quality=0.3)

if quality == "HIGH_QUALITY":
    print("✓ Good clustering, using results")
else:
    print("✗ Poor clustering quality, consider:")
    print("  - Try different metric (euclidean vs cosine)")
    print("  - Check if data is appropriate for clustering")
    print("  - Normalize embeddings if needed")
```

### Pattern 3: Adaptive Batching

Cluster in batches but reuse model for consistency.

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

# Fit on representative sample
sample = torch.randn(500, 128)
kmeans = ModularClusterKmeans(metric='cosine')
kmeans.fit(sample)
print(f"Learned K={kmeans.best_k} from sample")

# Apply to large batches
large_data = torch.randn(10000, 128)
batch_size = 1000
all_labels = []

for i in range(0, len(large_data), batch_size):
    batch = large_data[i:i+batch_size]
    batch_labels = kmeans.predict(batch)
    all_labels.append(batch_labels)
    print(f"Batch {i//batch_size}: {len(batch)} samples")

# Combine
all_labels = torch.cat(all_labels)
print(f"Total: {len(all_labels)} labels assigned")
```

## Integration Recipes

### Recipe 1: Clustering in ML Pipeline

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

# 1. Feature extraction (your own model)
def extract_features(data):
    # Your feature extractor here
    return torch.randn(len(data), 128)

# 2. Load data
raw_data = torch.randn(5000, 64)  # Example: images or sequences

# 3. Extract features
features = extract_features(raw_data)

# 4. Cluster
kmeans = ModularClusterKmeans(metric='cosine')
cluster_labels = kmeans.fit_predict(features)

# 5. Use clusters for downstream task
def downstream_analysis(raw_data, cluster_labels, cluster_id):
    """Analyze specific cluster."""
    cluster_mask = cluster_labels == cluster_id
    cluster_data = raw_data[cluster_mask]
    print(f"Cluster {cluster_id}: {len(cluster_data)} samples")
    return cluster_data

for cid in range(kmeans.best_k):
    cluster_data = downstream_analysis(raw_data, cluster_labels, cid)
```

### Recipe 2: Streaming Clustering Pipeline

```python
import torch
from modularnet.modularcluster.modularcluster_plugin import PluginModularCluster
from modularnet.plugin.plugin import PluginManager
import time

# Setup
manager = PluginManager()

# Add cluster plugin
cluster_plugin = PluginModularCluster(metric='cosine')
manager.register_plugin(cluster_plugin, name="clustering")

# Statistics collector
stats = {
    'batches_processed': 0,
    'total_samples': 0,
    'cluster_history': []
}

def collect_stats(labels):
    stats['batches_processed'] += 1
    stats['total_samples'] += len(labels)
    stats['cluster_history'].append(len(torch.unique(labels)))

manager.subscribe_to_topic("cu_labels", collect_stats)

# Simulate data stream
print("Starting data stream...")
for batch_idx in range(10):
    batch = torch.randn(256, 64)  # Batch of 256 samples
    manager.publish_to_topic("cu_positions", batch)
    time.sleep(1)

print(f"\nStream finished:")
print(f"  Batches: {stats['batches_processed']}")
print(f"  Total samples: {stats['total_samples']}")
print(f"  Cluster k per batch: {stats['cluster_history']}")
```

### Recipe 3: Hyperparameter Tuning

```python
import torch
from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans

embeddings = torch.randn(1000, 128)

configs = [
    {'metric': 'euclidean', 'precompute': True},
    {'metric': 'euclidean', 'precompute': False},
    {'metric': 'cosine', 'precompute': True},
    {'metric': 'cosine', 'precompute': False},
]

best_config = None
best_score = -1

for config in configs:
    kmeans = ModularClusterKmeans(
        metric=config['metric'],
        precompute_distances=config['precompute'],
        verbose=False
    )
    kmeans.fit(embeddings)
    
    score = kmeans.best_score
    if score > best_score:
        best_score = score
        best_config = config

print(f"Best config: {best_config}")
print(f"Score: {best_score:.4f}")
```

## Tips & Tricks

### Tip 1: Always Normalize for Cosine
```python
import torch.nn.functional as F

# Good
normalized = F.normalize(embeddings, p=2, dim=1)
kmeans = ModularClusterKmeans(metric='cosine')
labels = kmeans.fit_predict(normalized)

# Better explicitly
embeddings = F.normalize(embeddings, p=2, dim=1)
```

### Tip 2: Use Verbose Mode While Debugging
```python
kmeans = ModularClusterKmeans(metric='cosine', verbose=True)
kmeans.fit(embeddings)
# Shows K evaluation progress
```

### Tip 3: Inspect Silhouette Scores
```python
kmeans.fit(embeddings)
scores = kmeans.silhouette_scores
avg_score = sum(scores.values()) / len(scores)
print(f"Average quality: {avg_score:.4f}")
```

### Tip 4: Use GPU for Large Data
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kmeans = ModularClusterKmeans(metric='cosine', device=device)
labels = kmeans.fit_predict(embeddings)
```

### Tip 5: Cache the Fitted Model
```python
# First time: fit
kmeans = ModularClusterKmeans()
kmeans.fit(train_data)
first_labels = kmeans.labels

# Reuse model: predict
test_labels = kmeans.predict(test_data)  # Much faster!
```
