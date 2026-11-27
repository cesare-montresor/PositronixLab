# Open `xfail` Bugs Analysis

> **Scope:** All remaining `@pytest.mark.xfail(strict=True)` regression tests under `tests/`.
> **Totals:** **91** open `xfail` markers (**92** test cases) across **13** modules (15 test files).
> **Cleaned up:** 4 previously stale `XPASS(strict)` markers were removed (`modularcluster` B5; `modularobserver` `ModelMapper.map` device handling, `clear()` pre-forward hook cleanup, and `on_forward_layer` re-registration handle leak). `dataprovider` and `baselines` currently have 0 open `xfail` markers.
> **Important verification note (2026-11-26):** Running `--runxfail` shows that **4 of the P0 `xfail`s below are actually ALREADY FIXED in library code** and only continue to xfail due to test-code issues:
> - `modularcluster` B2 (`test_b2_predict_*`, `test_b2_stream_fit_*`): `_distance()` already guards `if x is y: dist.fill_diagonal_(0.0)`; tests fail because K=3 FAISS splits cluster 0 into two labels `[1, 2]` (no diagonal corruption).
> - `modularcluster` B3 (`test_b3_tracker_reorders_centroids_on_relabel`): `_relabel_to_match_previous()` already returns `(reordered_centroids, remapped_labels)`; test fails with `AttributeError: 'bool'` because it does not unpack the tuple (and return type annotation still says `-> torch.Tensor`).
> - `modularobserver` (`test_modular_observer_on_grad_weights_closure_bug`): `modularobserver.py:145` already binds `m=module_ref, p=param`; test fails with `TypeError: expected Variable, but hook returned 'tuple'` from the test callback.

---

## 1. Summary by Module

| Module | Open `xfail`s | Test File(s) | Detailed Module Audit |
| :--- | :---: | :--- | :--- |
| **`stats`** | 14 | [test_stats.py](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py) | [stats_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/stats_analysis.md) |
| **`metamodel`** | 11 | [test_metamodel.py](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py) | [metamodel_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/metamodel_analysis.md) |
| **`spacemath`** | 9 | [test_spacemath_extended.py](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py) | [spacemath_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/spacemath_analysis.md) |
| **`tasks`** | 9 | [test_tasks_compositionality.py](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py) | [tasks_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/tasks_analysis.md) |
| **`modularui`** | 8 | [test_modularui.py](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py) | [modularui_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularui_analysis.md) |
| **`deeplearning`** | 6 | [test_deeplearning.py](file:///home/cesare/Projects/PositronixLab/tests/deeplearning/test_deeplearning.py) | [deeplearning_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/deeplearning_analysis.md) |
| **`modularobserver`** | 6 | [test_modularobserver.py](file:///home/cesare/Projects/PositronixLab/tests/modularobserver/test_modularobserver.py) | [modularobserver_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularobserver_analysis.md) |
| **`template`** | 6 | [test_template.py](file:///home/cesare/Projects/PositronixLab/tests/template/test_template.py) | [template_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/template_analysis.md) |
| **`utils`** | 6 | [test_utils.py](file:///home/cesare/Projects/PositronixLab/tests/utils/test_utils.py) | [utils_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/utils_analysis.md) |
| **`nas`** | 5 | [test_nas.py](file:///home/cesare/Projects/PositronixLab/tests/nas/test_nas.py) | [nas_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/nas_analysis.md) |
| **`modularcluster`** | 4 | [test_streaming_kmeans.py](file:///home/cesare/Projects/PositronixLab/tests/modularcluster/test_streaming_kmeans.py) | [modularcluster_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularcluster_analysis.md) |
| **`modularspace`** | 4 | [test_clustertracker.py](file:///home/cesare/Projects/PositronixLab/tests/modularspace/test_clustertracker.py), [test_modularspace.py](file:///home/cesare/Projects/PositronixLab/tests/modularspace/test_modularspace.py), [test_space_step.py](file:///home/cesare/Projects/PositronixLab/tests/modularspace/test_space_step.py) | [modularspace_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularspace_analysis.md) |
| **`pipeline`** | 3 | [test_pipeline_config.py](file:///home/cesare/Projects/PositronixLab/tests/pipeline/test_pipeline_config.py) | [pipeline_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/pipeline_analysis.md) |
| **Total** | **91** | **15 test files** | |

---

## 2. Priority Triage

### P0 — Silent Correctness & Data Corruption (Fix First)
- **`deeplearning` (1):** `DeeplearningBase.process_batch()` runs `self.model(x)` outside `torch.set_grad_enabled(self.is_training)` during eval (B4).
- **`metamodel` (4):** `MetaMetric` shadows `def custom_metric()` with `self.custom_metric = {}`; `output_num_layers=1` outputs `input_hidden_dim`; `output_hidden_dim` ignored; `dropout_every=0` enables dropout on layer 0.
- **`spacemath` (3):** `weighting.py` `mask_cluster_membership` uses `max()+1` instead of unique non-negative cluster count; `losses.py` `loss_batch_repulsion` ignores `mixed` for 2D inputs; `common.py` `spow` produces `inf` for negative powers near zero.
- **`template` (2):** `TemplateFMnistFC/CNN` use `'fmnist'` instead of `'fashion_mnist'` (Bug #3); `TemplateVariant.matches_trial_parameters()` returns `True` on missing fixed keys (Bug #2).
- **`utils` (1):** `seed_everything()` sets `cudnn.benchmark = True`, contradicting `cudnn.deterministic = True` (B3).
- **Test-only fixes (4 — library bugs already solved, archived in `reports/solved/`):** `modularcluster` B2 ×2 (`test_b2_predict_*`, `test_b2_stream_fit_*`), `modularcluster` B3 (`test_b3_tracker_reorders_centroids_on_relabel`), and `modularobserver` (`test_modular_observer_on_grad_weights_closure_bug`).

### P1 — Crashes, Leaks & Broken Features
- **`modularspace` (4):** `ModularSpace.get_space()` crashes with `stats_enabled=False` (B1); `LayerSpace.process_scores()` calls `.backward()` on detached zero tensor (B3); `ClusterTracker` `k_old == 0` crash (B4) and single-child split offset (B5).
- **`nas` (5):** Solo OOM retry off-by-one; `TrialConfig` `device=None` not overridden by `gpu_id`; `extract_metrics` treats `bool` as numeric and crashes on `(mean, None)`; `export_best()` leaves stale YAML; `warm_start_from()` leaks global `SESSION_FACTORY` on error.
- **`pipeline` (3):** `BaseConfig.from_dict()` fails to coerce `int | None` and `Optional[str]` without future annotations (B2 ×2); `BaseConfig.load()` raises `AttributeError` on non-dict YAML root (B7).
- **`deeplearning` (4):** Empty DataLoader `UnboundLocalError` (B5); draft mode runs 21 instead of 20 batches (B7); `on_batch_start` signature mismatch (B8); multiple `backward=True` metrics fail without `retain_graph` (B9).
- **`modularobserver` (5):** `GroupNorm` metrics; `ModelNode.to_dict()` callable without `__name__`; float `trace_samples` decremented; `F.leaky_relu` omitted from `_MapperInterceptor`; `ModularObserver.map()` returns `None`.
- **`modularui` (8):** Empty `visible_layers` shows all; scalar epoch crash; broken relative CSS paths; `add_point(None)` crash; non-contiguous position indices misalign clusters; `VarsDisplay` mixed key types; `get_model_graph()` mutable ref; `DashboardLayout` drops >3 top panels.
- **`stats` (14):** `_as_tensor` non-scalar list crash; `DisplacementFilter` invalid method & empty tensor; `WindowReducer` shape change & `maxlen=0`; `DisplacementReducer` empty tensor; `UmapProcessor` `last_index` ambiguity, `n_neighbors`, warm-start neuron count change, empty `centroids_2d`; `OverallProcessor` empty items; `Stream` empty generator & displacement-gated `_count`; `merge_streams(reducer_args=None)`.
- **`template` (4):** Ax 1.3 `ChoiceParameterConfig` `log_scale` crash (#1); YAML tuple round-trip (#4); unknown param kind silently ignored (#6); `TemplateFMnistCNN` `nas_min_acc` (#5).
- **`utils` (5):** `process_start_spawn(force=True)` ignores `'fork'` (B2); `check_tensor`/`info_tensor` `debug` forwarding & early-binding (B4 ×2); `info_tensor` crashes on int/empty tensors (B5 ×2).
- **`metamodel` (6):** `MetaModel.__init__` ignores `device="cpu"`; CNN classification `BatchNorm2d`/missing `Flatten`; CNN autoencoder `ConvTranspose2d` `layer_size()`; `nmum_fop()` on `Conv1d`; `layer_size(resnet_block)`; hardcoded `padding=1` breaks `cnn_kernel_size != 3`.

### P2 — Refactor Gaps & Legacy / Prototype Code
- **Config / feature gaps (2):**
  - `deeplearning`: `DeeplearningBase.__init__` requires `config.task`, incompatible with standalone `TrainConfig`.
  - `metamodel`: `'rnn'` listed in `LAYER_TYPES` but unimplemented in `block_layer()`.
- **Legacy `SpaceMath` & minor primitives (6):**
  - `spacemath`: `infer_shape` stray `self`, Euclidean `distance_elementwise(x, x)` `sqrt(eps)` floor, `loss_layer_alignment([])`, and 3 legacy `spacemath.py` bugs.
- **Earlier `tasks/compositionality` prototype (9):**
  - `atlas_check.py`, `weightmask_check.py`, `compositionality.py` — superseded for new benchmarks by `positronixlab.tasks` (`SubnetworkMask`, `MaskApplier`, `AtlasDiscoverer`, `ModularityBenchmark`), though still tested and exposed.

---

## 3. Complete Open Bug Inventory (91 Markers)

### `deeplearning` (6)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_b4_process_batch_eval_disables_grad_during_forward](file:///home/cesare/Projects/PositronixLab/tests/deeplearning/test_deeplearning.py#L492) | Bug B4 | `DeeplearningBase.process_batch()` runs `self.model(x)` outside `torch.set_grad_enabled(self.is_training)`. |
| [test_b5_process_epoch_empty_dataloader](file:///home/cesare/Projects/PositronixLab/tests/deeplearning/test_deeplearning.py#L507) | Bug B5 | `DeeplearningBase.process_epoch()` raises `UnboundLocalError` on an empty `DataLoader`. |
| [test_b7_draft_mode_processes_20_batches](file:///home/cesare/Projects/PositronixLab/tests/deeplearning/test_deeplearning.py#L523) | Bug B7 | Draft mode decrements `draft_batches=20` and breaks on `< 0`, executing 21 batches instead of 20. |
| [test_b8_train_on_batch_start_accepts_with_data](file:///home/cesare/Projects/PositronixLab/tests/deeplearning/test_deeplearning.py#L544) | Bug B8 | `DeeplearningTrain.on_batch_start()` omits the `with_data` parameter supported by `DeeplearningBase`. |
| [test_b9_multiple_backward_metrics](file:///home/cesare/Projects/PositronixLab/tests/deeplearning/test_deeplearning.py#L552) | Bug B9 | Multiple `backward=True` metrics call `loss.backward()` without `retain_graph` or summing losses. |
| [test_standalone_train_config_accepted_by_deeplearning_base](file:///home/cesare/Projects/PositronixLab/tests/deeplearning/test_deeplearning.py#L569) | Config gap | `DeeplearningBase.__init__` requires `config.task`, which standalone `TrainConfig` does not define. |

### `metamodel` (11)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_custom_metric_registration_bug](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L191) | `metacriterion.py:45,57` | `self.custom_metric = {}` shadows `def custom_metric()`. |
| [test_init_respects_cpu_device_arg](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L292) | `metamodel.py:61-72` | `MetaModel.__init__` ignores `device` arg and unconditionally calls `to_gpu()`. |
| [test_single_output_layer_outputs_output_data_shape](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L303) | `metamodel.py:250-255` | `output_num_layers=1` hits `is_first_layer` before `is_last_layer`, outputting `input_hidden_dim`. |
| [test_output_hidden_dim_is_used](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L322) | `metamodel.py:252` | `build_model_output_block` uses `input_hidden_dim` instead of `output_hidden_dim`. |
| [test_cnn_classification_forward](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L344) | `metamodel.py:264-276` | CNN classification attaches `BatchNorm2d` after `Linear` and lacks `Flatten`. |
| [test_cnn_autoencoder_multilayer_builds](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L365) | `metamodel.py:302,390-403` | CNN autoencoder with ≥2 layers calls `layer_size()` on `ConvTranspose2d` and crashes. |
| [test_dropout_every_zero_disables_dropout](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L385) | `metamodel.py:222,279` | `dropout_every=0` enables dropout on layer 0 instead of disabling dropout. |
| [test_nmum_fop_handles_conv1d](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L402) | `metamodel.py:420` | `nmum_fop()` indexes `kernel_size[1]` on `nn.Conv1d`, causing `IndexError`. |
| [test_layer_size_accepts_resnet_block](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L412) | `metamodel.py:392-403` | `layer_size(resnet_block)` unwraps to `nn.Sequential` instead of inner layer. |
| [test_cnn_kernel_size_5_with_residual](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L429) | `metamodel.py:164` | Hardcoded `padding=1` shrinks spatial dims and breaks ResNet skip when `cnn_kernel_size != 3`. |
| [test_rnn_listed_in_layer_types_builds](file:///home/cesare/Projects/PositronixLab/tests/metamodel/test_metamodel.py#L453) | `metamodel.py:58,158-172` | `'rnn'` is listed in `LAYER_TYPES` but unimplemented in `block_layer()`. |

### `modularcluster` (4)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_b2_predict_does_not_force_diagonal_assignment](file:///home/cesare/Projects/PositronixLab/tests/modularcluster/test_streaming_kmeans.py#L190) | Bug B2 | `_distance()` calls `fill_diagonal_(0.0)` on `(n, k)` point-to-centroid matrix, corrupting `predict()[:k]`. |
| [test_b2_stream_fit_does_not_corrupt_first_k_labels](file:///home/cesare/Projects/PositronixLab/tests/modularcluster/test_streaming_kmeans.py#L203) | Bug B2 | `ModularClusterKmeansStream._kmeans()` uses `_distance(x, centroids)`, zeroing diagonal and corrupting `labels[:k]`. |
| [test_b3_tracker_reorders_centroids_on_relabel](file:///home/cesare/Projects/PositronixLab/tests/modularcluster/test_streaming_kmeans.py#L214) | Bug B3 | `StreamingClusterTracker._relabel_to_match_previous()` remaps labels without reordering centroids. |
| [test_b6_warm_start_respects_configured_k_bounds](file:///home/cesare/Projects/PositronixLab/tests/modularcluster/test_streaming_kmeans.py#L245) | Bug B6 | `ModularClusterKmeansStream.fit(warm_start_k=..., k_search_radius=...)` ignores `self.k_min`/`self.k_max`. |

### `modularobserver` (6)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_extract_layer_io_metrics_groupnorm_bug](file:///home/cesare/Projects/PositronixLab/tests/modularobserver/test_modularobserver.py#L144) | `modelmapper.py:43-48` | `nn.GroupNorm` exposes `num_channels`, not `num_features`/`normalized_shape`. |
| [test_model_node_to_dict_callable_without_dunder_name_bug](file:///home/cesare/Projects/PositronixLab/tests/modularobserver/test_modularobserver.py#L235) | `modelmapper.py:115` | `ModelNode.to_dict()` assumes `activation_fn` has `__name__`. |
| [test_model_mapper_float_trace_budget_standing_rate_bug](file:///home/cesare/Projects/PositronixLab/tests/modularobserver/test_modularobserver.py#L341) | `modelmapper.py:465-474,523-524` | Float `trace_samples` is never sampled and is decremented to 0. |
| [test_model_mapper_discovers_leaky_relu_bug](file:///home/cesare/Projects/PositronixLab/tests/modularobserver/test_modularobserver.py#L353) | `modelmapper.py:244` | `_MapperInterceptor.target_ops` omits `F.leaky_relu`. |
| [test_modular_observer_map_return_value_bug](file:///home/cesare/Projects/PositronixLab/tests/modularobserver/test_modularobserver.py#L432) | `modularobserver.py:94-103` | `ModularObserver.map()` omits `return` statement and returns `None`. |
| [test_modular_observer_on_grad_weights_closure_bug](file:///home/cesare/Projects/PositronixLab/tests/modularobserver/test_modularobserver.py#L443) | `modularobserver.py:125` | Late-binding closure binds `module_ref` and `param` to the last loop iteration. |

### `modularspace` (4)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_clustertracker_recovers_from_empty_previous_step](file:///home/cesare/Projects/PositronixLab/tests/modularspace/test_clustertracker.py#L127) | `custertracker.py:175` (B4) | `ClusterTracker._match_and_project` calls `torch.argmax` on empty row when `k_old == 0`. |
| [test_clustertracker_split_positions_single_child_offsets_from_parent](file:///home/cesare/Projects/PositronixLab/tests/modularspace/test_clustertracker.py#L139) | `custertracker.py:239,286-287` (B5) | `switch` calls `_split_positions` with 1 centroid (`m==1`), returning `parent_outer` unchanged. |
| [test_modularspace_works_with_stats_disabled](file:///home/cesare/Projects/PositronixLab/tests/modularspace/test_modularspace.py#L51) | `modularspace.py:59` (B1) | `ModularSpace.get_space()` checks `None in (..., stats)` and raises `ValueError` when `stats_enabled=False`. |
| [test_layerspace_all_zero_losses_does_not_crash](file:///home/cesare/Projects/PositronixLab/tests/modularspace/test_space_step.py#L79) | `layerspace.py:210,338` (B3) | `LayerSpace.process_scores` calls `loss.backward()` on a detached zero tensor when all active losses are `0.0`. |

### `modularui` (8)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_layer_board_empty_visible_layers_hides_all](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L437) | `plots.py:237-242` | `_filtered_layer_ids` treats empty list `[]` as unfiltered and returns all layers. |
| [test_format_epoch_handles_scalar_int](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L446) | `layouts.py:98-104` | `_format_epoch` crashes with `TypeError` when given a scalar epoch `int`. |
| [test_external_stylesheets_has_no_broken_relative_theme_paths](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L454) | `layouts.py:34`, `modularui.py:123` | `external_stylesheets` contains nonexistent relative `themes/*.css` paths. |
| [test_add_point_ignores_none_value](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L462) | `modularui.py:257-266` | `add_point` raises `TypeError` on `None` instead of ignoring non-numeric scalar. |
| [test_non_contiguous_position_indices_align_clusters](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L470) | `modularui.py:441-445` | `_position_offset_locked` assumes contiguous `+1` indices; gaps misalign cluster lookup. |
| [test_vars_display_handles_mixed_key_types](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L481) | `components.py:70` | `VarsDisplay.render` crashes with `TypeError` on nested dict with mixed key types. |
| [test_get_model_graph_returns_defensive_copy](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L490) | `modularui.py:232-234` | `get_model_graph` returns internal mutable dict reference instead of a copy. |
| [test_dashboard_layout_renders_more_than_three_top_panels](file:///home/cesare/Projects/PositronixLab/tests/modularui/test_modularui.py#L500) | `docks.py:95,113` | `DashboardLayout` silently drops `top_panels` beyond `TOP_AREAS` (3). |

### `nas` (5)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_bug_solo_oom_retry_off_by_one](file:///home/cesare/Projects/PositronixLab/tests/nas/test_nas.py#L679) | `experiment.py:458` | `_retry_oom_solo` runs `2 + oom_retries` attempts on solo trials instead of `1 + oom_retries`. |
| [test_bug_trial_config_gpu_id_with_none_device](file:///home/cesare/Projects/PositronixLab/tests/nas/test_nas.py#L711) | `trial_config.py:57` | Uses `setdefault('device', ...)` which does not overwrite an explicit `None`. |
| [test_bug_extract_metrics_bool_and_none_sem](file:///home/cesare/Projects/PositronixLab/tests/nas/test_nas.py#L720) | `trial_pipeline.py:114-115` | Treats `bool` as `int`/`float` and crashes on `(mean, None)` tuple. |
| [test_bug_export_best_leaves_stale_yaml](file:///home/cesare/Projects/PositronixLab/tests/nas/test_nas.py#L734) | `experiment.py:551` | `export_best()` does not remove stale `best/trial_*.yaml` files on subsequent exports. |
| [test_bug_warm_start_error_leaks_global_db_session](file:///home/cesare/Projects/PositronixLab/tests/nas/test_nas.py#L761) | `experiment.py:254-286` | `warm_start_from()` does not restore global `SESSION_FACTORY` in a `finally` block on error. |

### `pipeline` (3)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_from_dict_pep604_optional_coercion_bug](file:///home/cesare/Projects/PositronixLab/tests/pipeline/test_pipeline_config.py#L184) | Bug B2 | `BaseConfig.from_dict()` fails to coerce PEP 604 union annotations (`int \| None`). |
| [test_load_non_dict_yaml_root_bug](file:///home/cesare/Projects/PositronixLab/tests/pipeline/test_pipeline_config.py#L197) | Bug B7 | `BaseConfig.load()` raises `AttributeError` instead of `ValueError` on non-dict YAML root. |
| [test_pipeline_config_v2_optional_str_coercion_bug](file:///home/cesare/Projects/PositronixLab/tests/pipeline/test_pipeline_config.py#L267) | Bug B2 | `pipeline_config_v2.py` omits `from __future__ import annotations`, so `Optional[str]` is not coerced. |

### `spacemath` (9)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_bug_infer_shape_stray_self](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L149) | `common.py:49` | `infer_shape` has stray `self` parameter. |
| [test_bug_spow_negative_power_near_zero](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L161) | `common.py:30` | `spow` defaults to `force_zero=True`, producing `inf` for negative powers near zero. |
| [test_bug_distance_elementwise_euclidean_zero_floor](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L202) | `space_type.py:70` | Euclidean `distance_elementwise(x, x)` returns `sqrt(eps)=1e-4` instead of `0.0`. |
| [test_bug_mask_cluster_membership_single_nonzero_cluster_id](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L300) | `weighting.py:74` | `mask_cluster_membership` uses `max()+1` instead of unique non-negative cluster count. |
| [test_bug_loss_layer_alignment_empty_list](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L346) | `losses.py:120` | `loss_layer_alignment` indexes `[0]` on empty sequence. |
| [test_bug_loss_batch_repulsion_2d_ignores_mixed](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L352) | `losses.py:45-54` | `loss_batch_repulsion` ignores `mixed` for 2D inputs. |
| [test_bug_legacy_spacemath_conv_activations_to_scores](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L446) | `spacemath.py:112` (legacy) | Unpacks `(B, 1, k)` tensor into `q, scores`. |
| [test_bug_legacy_spacemath_get_quantile_k_kernels_mismatch](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L453) | `spacemath.py:262` (legacy) | `get_quantile_k_kernels_per_sample` indexes `sorted_indices` with unsorted mask positions. |
| [test_bug_legacy_spacemath_euclidean_squeeze](file:///home/cesare/Projects/PositronixLab/tests/spacemath/test_spacemath_extended.py#L468) | `spacemath.py:85` (legacy) | `SpaceMath.distance` Euclidean pairwise calls `.squeeze()`, collapsing `(1, N)` to `(N,)`. |

### `stats` (14)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_as_tensor_list_of_1d_tensors_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L152) | `common.py:17` | `_as_tensor` calls `.reshape(())` on each tensor in a list/tuple, crashing on non-scalar tensors. |
| [test_invalid_method_should_raise](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L226) | `filter.py:9-11` | `DisplacementFilter` does not validate `method`; invalid values silently fall through to `'rms'`. |
| [test_empty_tensor_max_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L234) | `filter.py:26` | `DisplacementFilter(method='max')` crashes on empty tensors via `.max()`. |
| [test_window_reducer_shape_change_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L329) | `reducer.py:110-115` | `WindowReducer` does not reset deque on shape change, crashing `torch.stack` in `drain()`. |
| [test_window_reducer_maxlen_zero_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L340) | `reducer.py:105-115` | `WindowReducer(maxlen=0)` marks key dirty with empty deque, crashing `torch.stack([])`. |
| [test_displacement_reducer_empty_tensor_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L380) | `reducer.py:153` | `DisplacementReducer(method='max')` crashes on empty tensors via `.max()`. |
| [test_umap_last_index_before_emission_ambiguous](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L466) | `processor.py:168-170` | `last_index()` returns `0` before any emission, indistinguishable from first emission (`index=0`). |
| [test_umap_n_neighbors_exceeds_small_layer_size](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L475) | `processor.py:98` | `_create_mapper` clamps `n_neighbors` up to `min_neighbors` even when `num_neurons <= min_neighbors`. |
| [test_umap_warmstart_neuron_count_change_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L485) | `processor.py:119-124` | Warm-start `fit_transform` does not guard against neuron count change after fit. |
| [test_umap_centroids_2d_empty_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L497) | `processor.py:190` | `centroids_2d` crashes with `ValueError` on empty labels/points. |
| [test_overall_processor_empty_items_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L523) | `processor.py:213` | `OverallProcessor.process({})` crashes with `RuntimeError` when `_latest` is empty. |
| [test_stream_empty_generator_extractor_increments_count](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L642) | `stream.py:50` | `if not items:` is always `False` for lazy generators, incrementing `_count` on 0-item yields. |
| [test_stream_displacement_gated_should_not_be_due](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L661) | `stream.py:55` | `Stream` increments `_count` even when `DisplacementReducer` suppresses an update, yielding empty blocks. |
| [test_merge_streams_none_args_crashes](file:///home/cesare/Projects/PositronixLab/tests/stats/test_stats.py#L685) | `stream.py:89` | `merge_streams` crashes with `TypeError` when `reducer_args` is explicitly `None`. |

### `tasks` — `compositionality` prototype (9)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_modularity_score_before_run_handles_empty_stats](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L211) | `atlas_check.py:310-315` | `modularity_score()` divides by `len(stats) == 0` when empty. |
| [test_atlas_detaches_hooks_after_run](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L226) | `atlas_check.py:117` | `AtlasCheck` never detaches forward hooks after `run()`. |
| [test_weightmask_respects_num_epochs](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L293) | `weightmask_check.py:18,33` | `num_epochs` is ignored; only 1 epoch executes. |
| [test_weightmask_handles_batch_missing_a_class](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L318) | `weightmask_check.py:41-54` | Empty class slice produces `NaN` `task_loss`/`total_loss`. |
| [test_weightmask_restores_requires_grad](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L341) | `weightmask_check.py:95-96,144-159` | `remove_hooks()` does not restore `requires_grad`. |
| [test_weightmask_manager_ignores_weightless_layers](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L352) | `weightmask_check.py:98-117` | Target layer without `weight` raises `KeyError` in `pre_hook`. |
| [test_compositionality_check_empty_dataloader](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L424) | `compositionality.py:151` | `run()` divides by zero on empty dataloader. |
| [test_static_mask_applier_accepts_raw_atlas_masks](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L435) | `atlas_check.py:235-240`, `compositionality.py:28-36` | Atlas 1D mask `(out_features,)` fails to broadcast against non-square weight `(out, in)` and uses dotted keys (`'net.0'`) ignored by `StaticMaskApplier` (`'net_0'`). |
| [test_compositionality_pipeline_handles_missing_test_split](file:///home/cesare/Projects/PositronixLab/tests/tasks/test_tasks_compositionality.py#L453) | `compositionality.py:229` | `CompositionalityPipeline` crashes when `dataprovider.test` is `None`. |

### `template` (6)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_add_choice_with_log_scale_converts_to_ax](file:///home/cesare/Projects/PositronixLab/tests/template/test_template.py#L92) | Bug #1 | `ChoiceParameterConfig` in Ax 1.3 does not accept `log_scale` kwarg. |
| [test_yaml_round_trip_with_tuple_fixed_value](file:///home/cesare/Projects/PositronixLab/tests/template/test_template.py#L187) | Bug #4 | `save()` uses `yaml.dump` (emits `!!python/tuple`) while `load()` uses `yaml.safe_load`. |
| [test_unknown_param_kind_raises](file:///home/cesare/Projects/PositronixLab/tests/template/test_template.py#L198) | Bug #6 | `to_parameter_configs()` silently ignores unknown parameter types instead of raising `ValueError`. |
| [test_fmnist_dataset_name_matches_dataprovider_factory](file:///home/cesare/Projects/PositronixLab/tests/template/test_template.py#L280) | Bug #3 | `TemplateFMnistFC/CNN` set `dataset_name='fmnist'` instead of `DataProviderFactory`'s `'fashion_mnist'`. |
| [test_fmnist_cnn_nas_min_acc_matches_fmnist_fc](file:///home/cesare/Projects/PositronixLab/tests/template/test_template.py#L289) | Bug #5 | `TemplateFMnistCNN` forgets to override `nas_min_acc=0.85` (inherits `0.95` from `TemplateDL`). |
| [test_variant_matches_trial_parameters_missing_keys_returns_false](file:///home/cesare/Projects/PositronixLab/tests/template/test_template.py#L362) | Bug #2 | `matches_trial_parameters()` returns `True` when fixed keys are missing from parameters. |

### `utils` (6)
| Test | Source / Bug | Summary |
| :--- | :--- | :--- |
| [test_process_start_spawn_force_overrides_fork](file:///home/cesare/Projects/PositronixLab/tests/utils/test_utils.py#L71) | Bug B2 | `process_start_spawn(force=True)` only warns when start method is `'fork'` and never calls `mp.set_start_method('spawn', force=True)`. |
| [test_seed_everything_disables_cudnn_benchmark](file:///home/cesare/Projects/PositronixLab/tests/utils/test_utils.py#L266) | Bug B3 | `seed_everything()` sets `torch.backends.cudnn.benchmark = True`, contradicting `cudnn.deterministic = True`. |
| [test_check_tensor_forwards_debug_to_info_tensor](file:///home/cesare/Projects/PositronixLab/tests/utils/test_utils.py#L300) | Bug B4 | `check_tensor(t, debug=True)` calls `info_tensor(tensor, name)` without forwarding `debug=debug`. |
| [test_info_tensor_respects_module_debug_flag](file:///home/cesare/Projects/PositronixLab/tests/utils/test_utils.py#L311) | Bug B4 | `info_tensor` default argument `debug=debug` is early-bound to `False` at import time, ignoring runtime `utils.debug = True`. |
| [test_info_tensor_handles_integer_tensor](file:///home/cesare/Projects/PositronixLab/tests/utils/test_utils.py#L322) | Bug B5 | `info_tensor()` calls `tensor.mean()` directly, raising `RuntimeError` on integer tensors. |
| [test_info_tensor_handles_empty_tensor](file:///home/cesare/Projects/PositronixLab/tests/utils/test_utils.py#L330) | Bug B5 | `info_tensor()` calls `tensor.min()` without checking `numel() == 0`, raising `RuntimeError` on empty tensors. |
