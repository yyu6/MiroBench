"""Lazy model factory used by the portable SynthPAI baseline.

Upstream imports every optional provider at module import time.  The matched
baseline only needs the OpenAI-compatible provider, so lazy imports keep the
fresh-machine environment small without changing provider behavior.
"""
from src.configs import ModelConfig

from .model import BaseModel
from .open_ai import OpenAIGPT


def get_model(config: ModelConfig) -> BaseModel:
    if config.provider in {"openai", "azure"}:
        return OpenAIGPT(config)
    if config.provider == "hf":
        from .hf_model import HFModel

        return HFModel(config)
    if config.provider == "together":
        from .together_model import TogetherModel

        return TogetherModel(config)
    if config.provider == "anthropic":
        from .anthropic_model import AnthropicModel

        return AnthropicModel(config)
    if config.provider == "gcp":
        from .gcp.gcp_model import GCPModel

        return GCPModel(config)
    if config.provider == "loc":
        if config.name == "multi":
            from .multi_model import MultiModel

            return MultiModel(config, [get_model(item) for item in config.submodels])
        if config.name == "chain":
            from .chain_model import ChainModel

            return ChainModel(config, [get_model(item) for item in config.submodels])
    raise NotImplementedError(f"Unsupported SynthPAI provider: {config.provider}")
