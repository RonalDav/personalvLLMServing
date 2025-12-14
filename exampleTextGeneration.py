# import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from model_config import ModelConfig
from standardTextServing import VLLMChatClient


# Global logger instance
logger = None

@dataclass
class PipelineConfig:
    """Pipeline configuration settings"""
    history_file: Path
    model_name: str
    model_config: ModelConfig
    log_level: str = "INFO"
    output_dir: Path = Path("output")


def initialize_model(config: PipelineConfig) -> Optional[VLLMChatClient]:
    """Initialize the VLLM model with error handling"""
    try:
        return VLLMChatClient(
            config.model_name,
            config=config.model_config,
            history_file=config.history_file
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        # raise PipelineError("Model initialization failed") from e

def run_demo(model: VLLMChatClient) -> str:
    pass


def main() -> None:
    """Main entry point with configuration and error handling"""
    config = PipelineConfig(
        history_file=Path("Generative_history.json"),
        model_name="microsoft/Fara-7B",
        model_config=ModelConfig(
            max_tokens=3000,
            temperature=0.0,
            max_model_len=20000,
            gpu_memory_util=0.9
        ),
        log_level="INFO",
        output_dir=Path("pipeline_output")
    )
        
    model = initialize_model(config)
    # result = run_pipeline(model)
    result = run_demo(model)

    print()



if __name__ == "__main__":
    main()