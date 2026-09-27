from dataclasses import dataclass, field
from typing import Dict, Type, Any

try:
    from vlm_benchmark.vlm_client_dashscope_image import DashscopeImageGenerationClient  # type: ignore
    from vlm_benchmark.vlm_client_dashscope_mm import DashscopeMultiModalClient  # type: ignore
    from vlm_benchmark.vlm_client_openai import OpenAIVideoClient  # type: ignore
    from vlm_benchmark.vlm_client_seed20_ark import Seed20ArkClient  # type: ignore
except ModuleNotFoundError:
    from vlm_client_dashscope_image import DashscopeImageGenerationClient  # type: ignore
    from vlm_client_dashscope_mm import DashscopeMultiModalClient  # type: ignore
    from vlm_client_openai import OpenAIVideoClient  # type: ignore
    from vlm_client_seed20_ark import Seed20ArkClient  # type: ignore


@dataclass
class VLMConfig:
    """Connection settings and client options for one supported VLM."""

    client_cls: Type[Any]
    model: str
    base_url: str
    api_key_env: str
    client_kwargs: Dict[str, Any] = field(default_factory=dict)


VLM_REGISTRY: Dict[str, VLMConfig] = {
    "qwen-image-2.0-pro": VLMConfig(
        client_cls=DashscopeImageGenerationClient,
        model="qwen-image-2.0-pro",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={
            "max_frames": 64,
            "n": 1,
            "size": "512*512",
            "watermark": False,
            "prompt_extend": False,
            "retry_wait_s": 20.0,
        },
    ),
    "qwen3.5-plus": VLMConfig(
        client_cls=DashscopeMultiModalClient,
        model="qwen3.5-plus",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"fps": 5, "max_frames": 64, "prefer_image_frames": True},
    ),
    "qwen3.5-plus-2026-02-15": VLMConfig(
        client_cls=DashscopeMultiModalClient,
        model="qwen3.5-plus-2026-02-15",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"fps": 5, "max_frames": 64, "prefer_image_frames": True},
    ),
    "qwen3.6-flash-2026-04-16": VLMConfig(
        client_cls=DashscopeMultiModalClient,
        model="qwen3.6-flash-2026-04-16",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"fps": 5, "max_frames": 64, "prefer_image_frames": True},
    ),
    "qwen3.5-122b-a10b": VLMConfig(
        client_cls=DashscopeMultiModalClient,
        model="qwen3.5-122b-a10b",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"fps": 5, "max_frames": 64, "prefer_image_frames": True},
    ),
    "qwen3.5-27b": VLMConfig(
        client_cls=DashscopeMultiModalClient,
        model="qwen3.5-27b",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"fps": 5, "max_frames": 64, "prefer_image_frames": True},
    ),
    "qwen3.6-plus": VLMConfig(
        client_cls=DashscopeMultiModalClient,
        model="qwen3.6-plus",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"fps": 5, "max_frames": 64, "prefer_image_frames": True},
    ),
    "seed2.0-vl": VLMConfig(
        client_cls=Seed20ArkClient,
        model="doubao-seed-2-0-lite-260215",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="ARK_API_KEY",
        client_kwargs={"fps": 5},
    ),
    "gpt5.4": VLMConfig(
        client_cls=OpenAIVideoClient,
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        client_kwargs={
            "fps": 5,
            "max_frames": 64,
            "use_responses_api": True,
            "use_reasoning_summary": True,
        },
    ),
}


def create_vlm_client(tag: str):
    """Create a configured VLM client by registry tag."""
    if tag not in VLM_REGISTRY:
        raise KeyError(f"unknown VLM tag: {tag}; choices: {list(VLM_REGISTRY)}")
    cfg = VLM_REGISTRY[tag]
    return cfg.client_cls(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        **cfg.client_kwargs,
    )


__all__ = ["VLMConfig", "VLM_REGISTRY", "create_vlm_client"]
