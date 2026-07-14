# Temporal Cluster Identity Tracking

*(ModularCluster — matching subsequent clusterings of an evolving,
contrastively-trained embedding space)*

## 1. Problem

You have a fixed set of `N` items whose embeddings live in a `D`-dimensional
space that is *continuously changing* under contrastive training. At every
timestep `t` you re-run a clustering algorithm (spherical k-means, G-means,
etc. — external to this doc) on the current embeddings and obtain. At each timestep you have `i` clusters:

- `P_t` — for each cluster, the index of embedding.

- `C_t` — cosine-normalized centroids, one `D`-dim vector per cluster. (X,Y,...)`t``i`

- `O_t` — a new high-dimensional cosine space in which centroids are projected, decoupled from the original one. A separate contrastive process will act on this space. Humans can only sketch a 2D or 3D projection of `O_t`; the tracked space itself does not need to be low-dimensional.  (U,V,...)`t``i`

Because clustering is re-run from scratch each timestep, cluster *ids* are
arbitrary — cluster `3` at `t` has no inherent relationship to cluster `3` at
`t-1`. 

The problem this algorithm solves is: **given two independent
clusterings of (mostly) the same items, recover which new cluster continues
which old cluster's identity**. Both the embedding space and the outer space
are treated as cosine spaces.
A cluster identity is defined by it's elements, sure matching happen when more than 50% of the elements match ( best match ). Elements and clusters are "consumed" in the process, so they can't be picked twice.
Using ebedding indexes as identity, is to avoid complitely volume matching in high dimensionality.
The only moment when the C (X,Y,...) coords affects O (U,V,...) coords is upon "split" to compute a gap width for the new outer-space children. That gap is based on the distance between the embedding-space centroids, with both spaces cosine-normalized. In any other case, space O (U,V,...) is untouched by this process (it has its own optimizer).

Four situations can occur between `t-1` and `t`:

| Event    | Meaning                                                  |
|----------|-----------------------------------------------------------|
| `init`   | first clustering ever — no history to match against       |
| `match`  | one old cluster continues as one new cluster (identity carry-over) |
| `split`  | one old cluster's items are distributed into 2+ new clusters |
| `merge`  | 2+ old clusters' items consolidate into one new cluster    |
| `switch` | a minority of items jump from one lineage to another, without being large enough to count as a full split/merge |
| `add`    | a new cluster with no meaningful predecessor (spontaneous) |
| `remove` | an old cluster whose mass dissolved with no clear successor |

`switch` is not a primitive: it is **defined as a split (of the jumping
subgroup off its old cluster) followed by a merge (into the destination
cluster)** — i.e. it reuses the split and merge coordinate formulas rather
than needing its own.

---

## 2. Notation

| Symbol         | Shape          | Meaning                                            |
|----------------|----------------|-----------------------------------------------------|
| `N`            | scalar         | fixed total number of items (e.g. 512)              |
| `D`            | scalar         | embedding dimensionality                            |
| `d_out`        | scalar         | output space dimensionality (can be high-dimensional; human drawings use a 2D/3D projection)  |
| `K_t`          | scalar         | number of clusters at time `t` (variable)           |
| `assignment_t` | `[N]` (long)   | item → cluster id at time `t`, `-1` if unassigned    |
| `C_t`          | `[K_t, D]`     | cluster centroids in embedding space                |
| `sizes_t`      | `[K_t]` (long) | number of items per cluster                         |
| `O_t`          | `[K_t, d_out]` | stable outer-space coordinate per cluster            |

---

## 3. Pipeline (one timestep transition, `t-1 → t`)

```
1. Cluster the current embeddings  →  assignment_t, C_t          (external)
2. Compute the overlap matrix between P_{t-1} and P_t
3. For each new cluster, determine its primary lineage (arg-max overlap)
4. Aggregate on the old-cluster side to classify: match / split / merge / switch / add / remove
5. Update O_t according to the classified event for every cluster
```

### 3.1 Overlap matrix

For every pair `(new cluster j, old cluster i)`:

```
overlap[j, i] = | { item k : assignment_t[k] = j  AND  assignment_{t-1}[k] = i } |
```

Computed in one shot via one-hot matmul:

```
overlap = one_hot(assignment_t).T @ one_hot(assignment_{t-1})     # [K_t, K_{t-1}]
```

### 3.2 Grouping / sorting (as sketched bottom-left)

- Indices are grouped by their **new** cluster assignment (this *is* `P_t`).
- Within each group, indices are sorted (canonical ordering — this is what the
  hand-drawn table of sorted index columns shows).
- Groups (new clusters) are then **sorted by size, descending** — matching
  proceeds biggest cluster first, since large, well-populated clusters give
  the most reliable overlap signal and should claim their lineage before
  smaller/ambiguous ones are considered.

### 3.3 Matching rule ("> 50% of labels ⇒ same identity")

For new cluster `j`, let `frac(j, i) = overlap[j, i] / size_t[j]`.

- **Primary lineage** of `j` is `i* = argmax_i overlap[j, i]`.
- If `frac(j, i*) > 0.5` → `j` **is** `i*` (its identity is inherited from
  `i*`); this is the "consume labels" step in the sketch: `i*`'s identity is
  now consumed/claimed by `j`.
- Any *other* old cluster `i'` with `switch_threshold < frac(j, i') ≤ 0.5`
  is recorded as a **secondary (switch-in) contribution** to `j`.
- Contributions with `frac(j, i) ≤ switch_threshold` are treated as noise
  and ignored (mirrors the little wavy "noise" arrow in the bottom-right box).
- If **no** old cluster reaches `frac > 0.5`, `j` has no clean primary
  lineage. If the largest contributors together explain a majority of `j`
  (i.e. plausible **merge** target), those old clusters become `j`'s merge
  set. Otherwise `j` is a brand-new cluster (`add`).

### 3.4 Old-cluster side aggregation

After every new cluster has picked its primary (if any):

- an old cluster claimed as primary by **exactly one** new cluster → **match**
- an old cluster claimed as primary by **two or more** new clusters → **split**
  (all of those new clusters are its children)
- an old cluster claimed by **no one** (neither as primary nor as a merge
  contributor) → **remove**

---

## 4. Outer-space coordinate update rules

### 4.1 Init (`t0` box)

No history. `O_t0` is sampled uniformly at random as a collection of
embedding vectors (e.g. in `[-1, 1]^d_out`, or any other bounded region
used for initialization). There is no special geometric radius here; this
corresponds to the top-left box in the sketch.

### 4.2 Match

Direct carry-over, no computation:

```
O_t[j] = O_{t-1}[i*]
```

### 4.3 Split (`t1` "ADD" box)

Old cluster `i` is the parent of children `j_1 … j_m` (`m ≥ 2`).

1. **Magnitude** — computed from how far apart the children's *embedding-space*
   centroids actually are (this is the sketch's `DISTANCE`):

   ```
   distance = mean_pairwise_L2( C_t[j_1], …, C_t[j_m] )      # in D-dim space
   GAP       = distance / 2                                   # sketch: "GAP = DISTANCE / 2"
   gap       = gap_scale * GAP                                 # gap_scale is a tunable knob
   ```

2. **Direction** — placement around the parent's old position, so children
  separate while their size-weighted centroid stays anchored near
   `O_{t-1}[i]`. The tracked outer space is just a collection of vectors;
   any human-facing sketch is only a projection of that collection:
   - `d_out = 2`: children are placed evenly around a circle with offset
     magnitude `gap` centered at `O_{t-1}[i]` (angle `= 2π·k/m`).
   - `d_out > 2`: children are placed along deterministic (seeded-by-`i`)
     pseudo-random unit vectors with offset magnitude `gap`, since there is no
     canonical circle in higher dimensions.

   ```
   O_t[j_k] = O_{t-1}[i] + gap · direction_k
   ```

This matches the sketch's `MAX/MIN(x,y)` bookkeeping: those are the bounding
extremes of the children's centroids used to size the gap region that the new
`(u,v)` pair is placed within.

### 4.4 Merge (`t2` "REM" box)

Old clusters `i_1 … i_p` merge into new cluster `j`. Weighted by **size**
(the sketch's `n`, `m`):

```
O_t[j] = Σ_k ( size_{t-1}[i_k] · O_{t-1}[i_k] )  /  Σ_k size_{t-1}[i_k]
```

For the two-cluster case this is exactly the sketch's formula:

```
(u,v)_merged = ( (u,v)_1 · n + (u,v)_2 · m ) / (n + m)
```

### 4.5 Switch (bottom-right box, `SWITCH = SPLIT − MERGE` / composed)

A subgroup of items previously counted under old cluster `i₂` shows up as a
minority contribution inside new cluster `j` (whose primary lineage is some
other cluster `i₁`, handled by 4.2/4.3/4.4 already). The switch does not get
its own formula — it is literally:

1. **Split step** — treat the jumping subgroup as if it had split off `i₂`,
   giving it a temporary position `O_split = O_{t-1}[i₂] + gap · direction`
   (§4.3, with the subgroup as one "child" of size `overlap[j, i₂]`).
2. **Merge step** — fold `O_split` into `j`'s already-computed coordinate,
   weighted by the subgroup size vs. `j`'s otherwise-explained size (§4.4).

### 4.6 Add / Remove

- **Add**: no usable lineage → same as init (§4.1), random placement.
- **Remove**: identity is simply dropped; nothing carried forward.

---

## 5. Complexity

Let `K = max(K_{t-1}, K_t)`.

- Overlap matrix: one matmul, `O(N · K_{t-1} · K_t)` FLOPs, fully vectorized
  on GPU via one-hot encodings — no Python loop over `N`.
- Matching / classification: a Python-level loop, but **over clusters, not
  items** (`K` is small — tens, not hundreds), so this is negligible even
  though it isn't matrix-vectorized.
- Overall: dominated by the single matmul; `N` never appears in a Python loop.

---

## 6. Assumptions worth flagging

The sketch is a compressed personal shorthand, so a few points were resolved
with an explicit (and adjustable) convention:

1. **Matching is arg-max overlap per new cluster**, not a globally optimal
   assignment (e.g. Hungarian). This is simpler, `O(K)` after the matmul, and
   matches the ">50%, consume" description; it can produce theoretically
   sub-optimal matches in adversarial cases, but is the direct reading of the
   sketch's greedy "biggest first" procedure.
2. **`switch_threshold`** (minimum overlap fraction to count as a real
   secondary contribution rather than noise) is a new explicit hyperparameter
   not numerically specified in the sketch — default `0.05`.
3. **Split placement** uses seeded random unit vectors in `d_out > 2` rather
  than a true geometric radius or a literal higher-dimensional circle — flag
  if you need something more principled (e.g. children placed along the
  leading PCA axis of their embedding-space displacement).
4. **Merge weights = old cluster *sizes*** (matches the sketch's `n, m`),
   not the raw overlap counts — i.e. a cluster's whole prior mass votes on
   the merged position, not just the portion that happened to land in `j`.
