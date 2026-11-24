from positronixlab.pipeline.config import BaseConfig

_LAZY_EXPORTS = {
    "Pipeline": ("positronixlab.pipeline.pipeline", "Pipeline"),
    "PipelineConfig": ("positronixlab.pipeline.pipeline_config", "PipelineConfig"),
    "PipelineOptions": ("positronixlab.pipeline.pipeline_config", "PipelineOptions"),
    "PipelineConfigV1": ("positronixlab.pipeline.pipeline_config_v1", "PipelineConfigV1"),
    "PipelineConfigV2": ("positronixlab.pipeline.pipeline_config_v2", "PipelineConfigV2"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseConfig",
    "Pipeline",
    "PipelineConfig",
    "PipelineOptions",
    "PipelineConfigV1",
    "PipelineConfigV2",
]
