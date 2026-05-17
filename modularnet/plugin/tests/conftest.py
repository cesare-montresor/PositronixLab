# Configure Ray before ANY imports
import os

# Completely disable Ray dashboard and related services to avoid pydantic v1 issues on Python 3.14
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_disable_memory_monitor"] = "1"
os.environ["RAY_disable_metrics_exporter"] = "1"
os.environ["RAY_disable_error_monitor"] = "1"
os.environ["RAY_head_node_manager_port"] = "0"
os.environ["RAY_metrics_export_period"] = "0"

# Increase timeouts significantly to handle slow startup
os.environ["RAY_raylet_start_wait_time_s"] = "180"
os.environ["RAY_redis_db_connect_retries"] = "30"
os.environ["RAY_gcs_server_request_timeout_seconds"] = "60"

# Now pytest can proceed



