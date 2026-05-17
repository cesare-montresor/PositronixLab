# Plugin System Tests

## Overview

The plugin system test suite is comprehensive and covers all major features at a high level:

- **PubSub Feature**: Full pub/sub messaging system
- **Lifecycle Feature**: Load/start/stop/unload plugin lifecycle  
- **Lifecycle Events Feature**: onStart, onStop, onUpdate, onError hooks
- **Env Feature**: Environment variables and onChange subscriptions
- **Integration Tests**: Multi-feature workflows

## Running the Tests

### Quick Start

```bash
# From project root
source .venv/bin/activate
pytest modularnet/plugin/tests/test_plugin_system.py -v
```

### Important: Ray Environment

The tests use Ray for process isolation. Ray requires a clean environment:

```bash
# Kill any existing Ray instances
pkill -9 ray 2>/dev/null

# Clean Ray temp files
rm -rf /tmp/ray 2>/dev/null

# Then run tests
pytest modularnet/plugin/tests/test_plugin_system.py -v
```

### Known Issues

**Python 3.14 + Ray 2.55.1 Dashboard Issue**: 
- Ray's dashboard has a pydantic v1 compatibility issue on Python 3.14
- The dashboard process may fail to start, but Ray core still initializes
- This causes ~30s delay on first test run
- Workaround: Dashboard is disabled by default in `conftest.py`

If tests hang for >60s on Ray initialization:
```bash
# Kill and retry
pkill -9 ray
rm -rf /tmp/ray
pytest modularnet/plugin/tests/test_plugin_system.py::TestLifecycleFeature -v
```

## Test Organization

### TestPubSubFeature
Tests the pub/sub messaging system:
- `test_pubsub_basic_publish_receive`: Publish → receive message flow
- `test_pubsub_multiple_topics`: Multiple plugins on same topic
- `test_pubsub_republish`: Plugins republishing processed messages

### TestLifecycleFeature
Tests plugin lifecycle management:
- `test_lifecycle_load_start_stop_unload`: Full lifecycle sequence
- `test_lifecycle_multiple_plugins`: Managing 3+ plugins
- `test_lifecycle_start_before_load_fails`: Error handling
- `test_lifecycle_double_start_idempotent`: Graceful handling of repeated starts

### TestLifecycleEventsFeature
Tests event callbacks:
- `test_event_on_start_called`: Verify onStart hook
- `test_event_on_stop_called`: Verify onStop hook
- `test_event_on_update_called`: Verify onUpdate hook on config change
- `test_event_on_error_handling`: Verify onError hook and error tracking

### TestEnvFeature
Tests environment variables:
- `test_env_set_and_get`: Basic get/set
- `test_env_get_with_default`: Default values
- `test_env_get_all`: Get all variables
- `test_env_on_change_single_subscription`: Single plugin receives onChange
- `test_env_on_change_tracking`: Track multiple changes
- `test_env_multiple_subscribers`: Multiple plugins subscribe to same var

### TestIntegration
Tests multi-feature workflows:
- `test_full_system_workflow`: Combined pub/sub + env + lifecycle

## Test Plugins

The test suite includes 4 dummy plugins for comprehensive testing:

- **DummyBasicPlugin**: Tests lifecycle events
- **DummyPubSubPlugin**: Tests pub/sub messaging
- **DummyEnvPlugin**: Tests environment subscriptions
- **DummyErrorPlugin**: Tests error handling

## Running Individual Tests

```bash
# Run single test class
pytest modularnet/plugin/tests/test_plugin_system.py::TestPubSubFeature -v

# Run single test
pytest modularnet/plugin/tests/test_plugin_system.py::TestPubSubFeature::test_pubsub_basic_publish_receive -xvs

# Run with high verbosity and full traceback
pytest modularnet/plugin/tests/test_plugin_system.py -xvs --tb=long
```

## Coverage

Each major feature has dedicated tests covering:
- ✅ Happy path (normal operation)
- ✅ Multiple instances (concurrency)
- ✅ Error cases (graceful degradation)
- ✅ Integration scenarios (feature interaction)

## Performance Notes

- First test run: ~30-40s (Ray initialization)
- Subsequent tests: ~2-5s per test
- Full suite: ~2-3 minutes

## Troubleshooting

### Tests hang on setup
```bash
pkill -9 ray
rm -rf /tmp/ray
sleep 2
pytest modularnet/plugin/tests/test_plugin_system.py -v
```

### "Ray not initialized" errors
```bash
# Ray needs clean startup
pkill -9 python
pkill -9 ray
rm -rf /tmp/ray
sleep 3
pytest modularnet/plugin/tests/test_plugin_system.py -v
```

### TimeoutError during test
This is usually due to Ray raylet startup being slow. Either:
- Wait for it to complete (takes ~30-60s on first run)
- Or kill and retry: `pkill -9 ray && sleep 2 && pytest ...`

## Notes

- Tests use `PluginManager` high-level API
- Each test is isolated via pytest fixtures
- Ray cluster is created per test fixture
- Tests verify both functionality and error handling
- Dummy plugins demonstrate proper plugin structure
