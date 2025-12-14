from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    max_tokens: int = 2400
    temperature: float = 0.5
    max_model_len: int = 5000
    gpu_memory_util: float = 0.91
    tensor_parallel: int = 1
    dtype: str = "auto"