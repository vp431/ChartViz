"""
Modern configuration management for ChartViz - Chart QA Explainability Tool.
"""
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ModelType(Enum):
    """Supported model types for chart QA."""
    UNICHART = "unichart"
    VISION_TRANSFORMER = "vit"
    MULTIMODAL_TRANSFORMER = "multimodal"


class VisualizationType(Enum):
    """Types of attention visualizations available."""
    CROSS_ATTENTION = "cross_attention"
    SELF_ATTENTION = "self_attention"
    HEATMAP_OVERLAY = "heatmap_overlay"
    PATCH_ATTENTION = "patch_attention"


@dataclass
class PathConfig:
    """Path configuration for the application."""
    base_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path(__file__).parent.absolute())
    local_models_dir: pathlib.Path = field(init=False)
    local_datasets_dir: pathlib.Path = field(init=False)
    cache_dir: pathlib.Path = field(init=False)
    outputs_dir: pathlib.Path = field(init=False)
    
    def __post_init__(self):
        self.local_models_dir = self.base_dir / "LocalModels"
        self.local_datasets_dir = self.base_dir / "LocalDatasets"
        self.cache_dir = self.base_dir / "cache"
        self.outputs_dir = self.base_dir / "outputs"
        
        # Ensure all directories exist
        for directory in [self.local_models_dir, self.local_datasets_dir, 
                         self.cache_dir, self.outputs_dir]:
            directory.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_type: ModelType
    local_dir: pathlib.Path
    hf_repo: Optional[str] = None
    requires_vision: bool = True
    max_image_size: Tuple[int, int] = (512, 512)
    max_text_length: int = 128
    patch_size: Tuple[int, int] = (16, 16)
    supports_attention_extraction: bool = True


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    local_dir: pathlib.Path
    hf_repo: Optional[str] = None
    default_samples: int = 100
    has_images: bool = True
    has_questions: bool = True
    has_answers: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    colorscale: List[List] = field(default_factory=lambda: [[0, "rgb(0,0,80)"], [1, "rgb(0,220,220)"]])
    attention_threshold: float = 0.1
    overlay_alpha: float = 0.6
    heatmap_resolution: Tuple[int, int] = (256, 256)
    patch_grid_size: Tuple[int, int] = (32, 32)
    max_attention_heads: int = 12
    default_layer_idx: int = -1  # Last layer
    default_head_idx: int = 0


@dataclass
class UIConfig:
    """UI configuration."""
    theme: str = "bootstrap"
    bootstrap_theme: str = "FLATLY"
    brand_name: str = "ChartViz - Chart QA Explainability"
    navbar_color: str = "primary"
    sidebar_width: str = "300px"
    chart_display_height: str = "500px"


@dataclass
class AppConfig:
    """Main application configuration."""
    debug_mode: bool = True
    port: int = 8051
    host: str = "0.0.0.0"
    auto_reload: bool = True
    log_level: str = "INFO"
    # GPU Configuration
    allow_multi_gpu: bool = False  # Set to True to enable multi-GPU model spreading
    force_gpu_id: Optional[int] = 0  # Force specific GPU (None for auto)


class ChartVizConfig:
    """Main configuration class for ChartViz."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.models = self._setup_models()
        self.datasets = self._setup_datasets()
        self.visualization = VisualizationConfig()
        self.ui = UIConfig()
        self.app = AppConfig()
        self.features = self._setup_features()
    
    def _setup_models(self) -> Dict[str, ModelConfig]:
        """Setup model configurations."""
        return {
            "unichart": ModelConfig(
                name="UniChart",
                model_type=ModelType.UNICHART,
                local_dir=self.paths.local_models_dir / "UniChart",
                hf_repo="ahmed-masry/unichart-base-960",
                requires_vision=True,
                max_image_size=(960, 960),
                patch_size=(32, 32),
                supports_attention_extraction=True
            ),
            "unichart-chartqa": ModelConfig(
                name="UniChart ChartQA-960",
                model_type=ModelType.UNICHART,
                local_dir=self.paths.local_models_dir / "unichart-chartqa",
                hf_repo="ahmed-masry/unichart-chartqa-960",
                requires_vision=True,
                max_image_size=(960, 960),
                patch_size=(32, 32),
                supports_attention_extraction=True,
                max_text_length=128
            ),
            "qwen2.5-vl-7b": ModelConfig(
                name="Qwen2.5-VL-7B-Instruct",
                model_type=ModelType.VISION_TRANSFORMER,
                local_dir=self.paths.local_models_dir / "qwen2.5-vl-7b",
                hf_repo="Qwen/Qwen2.5-VL-7B-Instruct",
                requires_vision=True,
                max_image_size=(1024, 1024),  # Supports dynamic resolution
                patch_size=(28, 28),
                supports_attention_extraction=True,
                max_text_length=32768  # 32K context length
            ),
            "llava_v1_5_7b": ModelConfig(
                name="LLaVA-v1.5-7B",
                model_type=ModelType.VISION_TRANSFORMER,
                local_dir=self.paths.local_models_dir / "LLaVA_v1_5_7B",
                hf_repo="llava-hf/llava-1.5-7b-hf",
                requires_vision=True,
                max_image_size=(336, 336),
                patch_size=(14, 14),
                supports_attention_extraction=True,
                max_text_length=512
            ),
            "llava_next_mistral_7b": ModelConfig(
                name="LLaVA-NeXT-Mistral-7B",
                model_type=ModelType.VISION_TRANSFORMER,
                local_dir=self.paths.local_models_dir / "LLaVA_NeXT_Mistral_7B",
                hf_repo="llava-hf/llava-v1.6-mistral-7b-hf",
                requires_vision=True,
                max_image_size=(672, 672),  # LLaVA-NeXT supports higher resolution
                patch_size=(14, 14),
                supports_attention_extraction=True,
                max_text_length=2048  # Longer context with Mistral backbone
            )
        }
    
    def _setup_datasets(self) -> Dict[str, DatasetConfig]:
        """Setup dataset configurations."""
        return {
            "chartqa": DatasetConfig(
                name="ChartQA",
                local_dir=self.paths.local_datasets_dir / "ChartQA",
                hf_repo="HuggingFaceM4/ChartQA",
                default_samples=200,
                has_images=True,
                has_questions=True,
                has_answers=True
            ),
            "plotqa": DatasetConfig(
                name="PlotQA",
                local_dir=self.paths.local_datasets_dir / "PlotQA",
                hf_repo="plotqa/PlotQA",
                default_samples=100,
                has_images=True,
                has_questions=True,
                has_answers=True
            ),
            "figureqa": DatasetConfig(
                name="FigureQA",
                local_dir=self.paths.local_datasets_dir / "FigureQA",
                hf_repo="ibm/figureqa",
                default_samples=100,
                has_images=True,
                has_questions=True,
                has_answers=True
            )
        }
    
    def _setup_features(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Setup feature configurations for different analysis types."""
        return {
            "chart_qa": {
                "instance": [
                    {"id": "attention_maps", "label": "Attention Maps", "color": "primary"},
                    {"id": "cross_attention", "label": "Cross-Modal Attention", "color": "info"},
                    {"id": "patch_analysis", "label": "Patch-level Analysis", "color": "success"},
                    {"id": "token_importance", "label": "Token Importance", "color": "warning"}
                ],
                "model": [
                    {"id": "error_analysis", "label": "Error Analysis", "color": "danger"},
                    {"id": "attention_patterns", "label": "Attention Patterns", "color": "info"},
                    {"id": "bias_detection", "label": "Bias Detection", "color": "warning"},
                    {"id": "performance_analysis", "label": "Performance Analysis", "color": "success"}
                ]
            }
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration for a specific dataset."""
        return self.datasets.get(dataset_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.datasets.keys())


# Global configuration instance
config = ChartVizConfig()

# Export commonly used configurations
MODEL_CONFIG = config.models
DATASET_CONFIG = config.datasets
VISUALIZATION_CONFIG = config.visualization
UI_CONFIG = config.ui
APP_CONFIG = config.app
FEATURE_CONFIG = config.features
PATHS = config.paths

# Backward compatibility constants
DEFAULT_UNICHART_MODEL = "unichart"
DEFAULT_CHARTQA_DATASET = "chartqa"
LOCAL_MODELS_DIR = str(PATHS.local_models_dir)
LOCAL_DATASETS_DIR = str(PATHS.local_datasets_dir)
DEBUG_MODE = APP_CONFIG.debug_mode
PORT = APP_CONFIG.port
HOST = APP_CONFIG.host

