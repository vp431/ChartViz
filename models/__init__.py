"""
ChartViz Models Package - Modern model management for chart QA explainability.
"""

from .model_manager import ModelManager
from .base_model import BaseChartQAModel

# Optional imports for specific models (only if available)
try:
    from .unichart_model import UniChartModel
    __all__ = ["ModelManager", "BaseChartQAModel", "UniChartModel"]
except ImportError:
    __all__ = ["ModelManager", "BaseChartQAModel"]

try:
    from .unichart_chartqa_model import UniChartChartQAModel
    __all__.append("UniChartChartQAModel")
except ImportError:
    pass

try:
    from .llava_model import LLaVAModel
    __all__.append("LLaVAModel")
except ImportError:
    pass

try:
    from .qwen25_vl_model import Qwen25VLModel
    __all__.append("Qwen25VLModel")
except ImportError:
    pass

try:
    from .llava_next_model import LLaVANextModel
    __all__.append("LLaVANextModel")
except ImportError:
    pass

