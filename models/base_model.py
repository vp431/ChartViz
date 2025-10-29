"""
Base model class for chart QA models with attention extraction capabilities.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoProcessor

from config import ModelConfig


@dataclass
class AttentionOutput:
    """Container for attention analysis outputs."""
    # Cross-modal attention (text-to-image)
    cross_attention: Optional[torch.Tensor] = None  # [layers, heads, text_tokens, image_patches]
    
    # Self-attention within modalities
    text_self_attention: Optional[torch.Tensor] = None  # [layers, heads, text_tokens, text_tokens]
    image_self_attention: Optional[torch.Tensor] = None  # [layers, heads, image_patches, image_patches]
    
    # Token information
    text_tokens: Optional[List[str]] = None
    image_patch_coords: Optional[List[Tuple[int, int]]] = None  # (x, y) coordinates
    
    # Model predictions
    predicted_answer: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Additional metadata
    layer_count: Optional[int] = None
    head_count: Optional[int] = None
    image_size: Optional[Tuple[int, int]] = None
    patch_size: Optional[Tuple[int, int]] = None


@dataclass
class ModelPrediction:
    """Container for model prediction outputs."""
    answer: str
    confidence: float
    logits: Optional[torch.Tensor] = None
    attention: Optional[AttentionOutput] = None
    processing_time: Optional[float] = None


class BaseChartQAModel(ABC):
    """Abstract base class for chart QA models with explainability features."""
    
    def __init__(self, model_config: ModelConfig, device: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_config: Configuration for the model
            device: Device to run the model on (auto-detected if None)
        """
        self.config = model_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components (to be initialized by subclasses)
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[AutoProcessor] = None
        
        # Attention hooks for extraction
        self._attention_hooks = []
        self._attention_outputs = {}
        
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model, tokenizer, and processor from local or remote storage."""
        pass
    
    @abstractmethod
    def predict(self, image: Union[Image.Image, str], question: str) -> ModelPrediction:
        """
        Make a prediction for a chart image and question.
        
        Args:
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            ModelPrediction with answer and confidence
        """
        pass
    
    @abstractmethod
    def extract_attention(self, image: Union[Image.Image, str], question: str) -> AttentionOutput:
        """
        Extract attention weights for explainability analysis.
        
        Args:
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            AttentionOutput with attention weights and metadata
        """
        pass
    
    def preprocess_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if necessary
        if image.size != self.config.max_image_size:
            image = image.resize(self.config.max_image_size, Image.Resampling.LANCZOS)
        
        return image
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text preprocessing
        text = text.strip()
        
        # Truncate if too long
        if len(text.split()) > self.config.max_text_length:
            words = text.split()[:self.config.max_text_length]
            text = " ".join(words)
        
        return text
    
    def _setup_attention_hooks(self) -> None:
        """Set up hooks to capture attention weights during forward pass."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self._clear_attention_hooks()
        self._attention_outputs = {}
        
        # Register hooks for attention extraction
        for name, module in self.model.named_modules():
            if self._is_attention_module(name, module):
                hook = module.register_forward_hook(self._attention_hook_fn(name))
                self._attention_hooks.append(hook)
    
    def _clear_attention_hooks(self) -> None:
        """Remove all attention hooks."""
        for hook in self._attention_hooks:
            hook.remove()
        self._attention_hooks = []
        self._attention_outputs = {}
    
    def _attention_hook_fn(self, module_name: str):
        """Create a hook function for capturing attention weights."""
        def hook(module, input, output):
            # Store attention weights from the module output
            if hasattr(output, 'attentions') and output.attentions is not None:
                self._attention_outputs[module_name] = output.attentions
            elif isinstance(output, tuple) and len(output) > 1:
                # Try to find attention in tuple output
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor) and out.dim() >= 3:
                        # Likely attention tensor [batch, heads, seq, seq]
                        self._attention_outputs[f"{module_name}_output_{i}"] = out
        return hook
    
    @abstractmethod
    def _is_attention_module(self, name: str, module: torch.nn.Module) -> bool:
        """
        Determine if a module is an attention module that should be hooked.
        
        Args:
            name: Module name
            module: PyTorch module
            
        Returns:
            True if this module should be hooked for attention extraction
        """
        pass
    
    def get_image_patches(self, image: Image.Image) -> List[Tuple[int, int]]:
        """
        Get the coordinates of image patches for attention visualization.
        
        Args:
            image: Input image
            
        Returns:
            List of (x, y) coordinates for each patch
        """
        patch_h, patch_w = self.config.patch_size
        img_h, img_w = image.size
        
        patches = []
        for y in range(0, img_h, patch_h):
            for x in range(0, img_w, patch_w):
                patches.append((x, y))
        
        return patches
    
    def create_attention_heatmap(self, attention_weights: torch.Tensor, 
                               image_size: Tuple[int, int]) -> np.ndarray:
        """
        Create a heatmap from attention weights.
        
        Args:
            attention_weights: Attention weights [num_patches] or [height, width]
            image_size: Target image size (width, height)
            
        Returns:
            Attention heatmap as numpy array
        """
        if attention_weights.dim() == 1:
            # Convert 1D patch attention to 2D grid
            patch_h, patch_w = self.config.patch_size
            grid_h = image_size[1] // patch_h
            grid_w = image_size[0] // patch_w
            
            # Reshape to grid
            attention_grid = attention_weights.view(grid_h, grid_w)
        else:
            attention_grid = attention_weights
        
        # Convert to numpy and resize to image dimensions
        attention_np = attention_grid.detach().cpu().numpy()
        
        # Normalize to 0-1 range
        attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
        
        return attention_np
    
    def compute_token_importance(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute importance scores for text tokens based on attention.
        
        Args:
            attention_weights: Cross-attention weights [text_tokens, image_patches]
            
        Returns:
            Dictionary mapping token indices to importance scores
        """
        if attention_weights.dim() != 2:
            raise ValueError("Expected 2D attention weights [text_tokens, image_patches]")
        
        # Sum attention across image patches for each token
        token_importance = attention_weights.sum(dim=1)
        
        # Normalize
        token_importance = token_importance / token_importance.sum()
        
        return {i: float(importance) for i, importance in enumerate(token_importance)}
    
    def validate_inputs(self, image: Union[Image.Image, str], question: str) -> Tuple[Image.Image, str]:
        """
        Validate and preprocess inputs.
        
        Args:
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            Tuple of (preprocessed_image, preprocessed_question)
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            processed_image = self.preprocess_image(image)
        except Exception as e:
            raise ValueError(f"Invalid image: {e}")
        
        processed_question = self.preprocess_text(question)
        
        return processed_image, processed_question
    
    def _safe_dtype_conversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely convert tensor dtype for model compatibility."""
        if not torch.is_tensor(tensor):
            return tensor
            
        # Only convert float32 to half if on CUDA 
        # Most models can handle half precision on GPU
        if tensor.dtype == torch.float32 and self.device == "cuda":
            try:
                return tensor.half()
            except:
                # If conversion fails, return original tensor
                return tensor
        
        return tensor
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.config.name,
            "type": self.config.model_type.value,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "max_image_size": self.config.max_image_size,
            "patch_size": self.config.patch_size,
            "max_text_length": self.config.max_text_length,
            "supports_attention": self.config.supports_attention_extraction,
            "local_dir": str(self.config.local_dir),
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._clear_attention_hooks()
        # Optionally unload model to free memory
        # self.unload_model()
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        self._clear_attention_hooks()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

