"""
Model manager for handling multiple chart QA models with dynamic loading.
"""
import importlib
import os
import time
import threading
from typing import Dict, List, Optional, Type, Union
from pathlib import Path
import torch
from rich.console import Console

from .base_model import BaseChartQAModel, ModelPrediction, AttentionOutput
from config import config, ModelConfig, ModelType


console = Console()


class ModelManager:
    """Manages multiple chart QA models with dynamic loading and caching."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.config = config
        self._loaded_models: Dict[str, BaseChartQAModel] = {}
        self._model_classes: Dict[str, Type[BaseChartQAModel]] = {}
        self._model_last_used: Dict[str, float] = {}  # Track last usage time
        self._cleanup_thread = None
        self._cleanup_interval = 60  # Check every minute
        self._inactivity_timeout = 600  # 10 minutes in seconds
        self._start_cleanup_thread()
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """Register default model implementations."""
        # Dynamic import to avoid circular dependencies
        try:
            from .unichart_model import UniChartModel
            self._model_classes["unichart"] = UniChartModel
        except ImportError:
            console.print("[yellow]Warning: UniChart model not available[/yellow]")

        try:
            from .unichart_chartqa_model import UniChartChartQAModel
            self._model_classes["unichart-chartqa"] = UniChartChartQAModel
        except ImportError:
            console.print("[yellow]Warning: UniChart ChartQA model not available[/yellow]")

        try:
            from .llava_model import LLaVAModel
            self._model_classes["llava_v1_5_7b"] = LLaVAModel
        except ImportError:
            console.print("[yellow]Warning: LLaVA model not available[/yellow]")

        try:
            from .qwen25_vl_model import Qwen25VLModel
            self._model_classes["qwen2.5-vl-7b"] = Qwen25VLModel
        except ImportError:
            console.print("[yellow]Warning: Qwen2.5-VL model not available[/yellow]")

        try:
            from .llava_next_model import LLaVANextModel
            self._model_classes["llava_next_mistral_7b"] = LLaVANextModel
        except ImportError:
            console.print("[yellow]Warning: LLaVA-NeXT model not available[/yellow]")
        
        # Note: Other Chart QA models (Donut) will use base implementations initially
        # Specific model classes can be created later for better optimization
        
        # Add more models as they become available
    
    def _start_cleanup_thread(self) -> None:
        """Start the background thread for cleaning up inactive models."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self._cleanup_interval)
                    self._cleanup_inactive_models()
                except Exception as e:
                    console.print(f"[yellow]Cleanup thread error: {e}[/yellow]")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        console.print("✓ Model cleanup thread started")

    def _free_memory_for_new_model(self, keep_models: Optional[List[str]] = None) -> None:
        """Proactively free GPU/CPU memory before loading a new model.
        Unloads all loaded models not in keep_models and clears CUDA cache.
        """
        if keep_models is None:
            keep_models = []
        # Unload all other models
        to_unload = [mid for mid in list(self._loaded_models.keys()) if mid not in keep_models]
        for mid in to_unload:
            console.print(f"Unloading {mid} to free memory for new model...")
            self.unload_model(mid)
        # Clear GPU cache
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    def _cleanup_inactive_models(self) -> None:
        """Unload models that have been inactive for too long."""
        current_time = time.time()
        inactive_models = []
        
        for model_id, last_used in self._model_last_used.items():
            if current_time - last_used > self._inactivity_timeout:
                inactive_models.append(model_id)
        
        for model_id in inactive_models:
            if model_id in self._loaded_models:
                console.print(f"🔄 Auto-unloading inactive model: {model_id}")
                self.unload_model(model_id)
    
    def clear_and_load_model(self, model_id: str) -> BaseChartQAModel:
        """
        Clear all loaded models and load the specified model.
        This ensures maximum memory availability for the new model.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model_id is not available
            RuntimeError: If model loading fails
        """
        console.print(f"Clearing all models and loading {model_id}")
        
        # Clear all loaded models first
        self.unload_all_models()
        
        # Force garbage collection and CUDA cache clear
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load the requested model
        return self.load_model(model_id)
    
    def unload_all_models(self) -> None:
        """Unload all currently loaded models."""
        model_ids_to_unload = list(self._loaded_models.keys())
        for model_id in model_ids_to_unload:
            self.unload_model(model_id)
        console.print("✓ All models unloaded")
    
    def _clear_cuda_memory(self) -> None:
        """Clear CUDA memory cache."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                console.print("✓ CUDA memory cleared")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not clear CUDA memory: {e}[/yellow]")
    
    def _update_model_usage(self, model_id: str) -> None:
        """Update the last usage time for a model."""
        self._model_last_used[model_id] = time.time()
    
    def register_model_class(self, model_id: str, model_class: Type[BaseChartQAModel]) -> None:
        """
        Register a new model class.
        
        Args:
            model_id: Unique identifier for the model
            model_class: Model class inheriting from BaseChartQAModel
        """
        if not issubclass(model_class, BaseChartQAModel):
            raise ValueError(f"Model class must inherit from BaseChartQAModel")
        
        self._model_classes[model_id] = model_class
        console.print(f"✓ Registered model class: {model_id}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model IDs.
        
        Returns:
            List of model IDs that can be loaded
        """
        available = []
        for model_id in self.config.get_available_models():
            model_config = self.config.get_model_config(model_id)
            if model_config and self._is_model_available(model_id, model_config):
                available.append(model_id)
        return available
    
    def _is_model_available(self, model_id: str, model_config: ModelConfig) -> bool:
        """
        Check if a model is available (downloaded and class registered).
        
        Args:
            model_id: Model identifier
            model_config: Model configuration
            
        Returns:
            True if model can be loaded
        """
        # Check if model files exist
        if not model_config.local_dir.exists():
            return False
        
        # Check for essential files
        config_file = model_config.local_dir / "config.json"
        weight_files = (
            list(model_config.local_dir.glob("*.bin")) + 
            list(model_config.local_dir.glob("*.safetensors"))
        )
        
        if not config_file.exists() or not weight_files:
            return False
        
        # Check if model class is registered
        # Look up by model_id first, then fall back to model_type
        if model_id in self._model_classes:
            return True
        
        model_type = model_config.model_type.value
        if model_type in self._model_classes:
            return True
        
        return False
    
    def is_model_implemented(self, model_id: str) -> bool:
        """
        Check if a model has an implementation class registered.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model has an implementation
        """
        if model_id not in self.config.get_available_models():
            return False
        
        model_config = self.config.get_model_config(model_id)
        if not model_config:
            return False
        
        # Look up by model_id first, then fall back to model_type
        if model_id in self._model_classes:
            return True
        
        model_type = model_config.model_type.value
        return model_type in self._model_classes
    
    def get_model_status(self, model_id: str) -> Dict[str, bool]:
        """
        Get detailed status of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with status information
        """
        if model_id not in self.config.get_available_models():
            return {
                "configured": False,
                "implemented": False,
                "downloaded": False,
                "available": False
            }
        
        model_config = self.config.get_model_config(model_id)
        implemented = self.is_model_implemented(model_id)
        downloaded = model_config.local_dir.exists() and (
            (model_config.local_dir / "config.json").exists() and 
            (list(model_config.local_dir.glob("*.bin")) + 
             list(model_config.local_dir.glob("*.safetensors")))
        )
        
        return {
            "configured": True,
            "implemented": implemented,
            "downloaded": downloaded,
            "available": implemented and downloaded
        }
    
    def load_model(self, model_id: str, device: Optional[str] = None, 
                  force_reload: bool = False) -> BaseChartQAModel:
        """
        Load a model by ID.
        
        Args:
            model_id: Model identifier
            device: Device to load model on (auto-detected if None)
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model ID is not available
            RuntimeError: If model loading fails
        """
        # Check model status first
        status = self.get_model_status(model_id)
        
        if not status["configured"]:
            available_models = [mid for mid in self.config.get_available_models() 
                              if self.get_model_status(mid)["available"]]
            raise ValueError(f"Model '{model_id}' is not configured. "
                           f"Available models: {available_models}")
        
        if not status["implemented"]:
            raise ValueError(f"Model '{model_id}' is configured but not implemented. "
                           f"Please ensure the model class is properly registered.")
        
        if not status["downloaded"]:
            raise ValueError(f"Model '{model_id}' is not downloaded. "
                           f"Please download the model first using the download script.")
        
        if model_id not in self.get_available_models():
            raise ValueError(f"Model '{model_id}' is not available. "
                           f"Available models: {self.get_available_models()}")
        
        # Return cached model if already loaded
        if model_id in self._loaded_models and not force_reload:
            console.print(f"Using cached model: {model_id}")
            self._update_model_usage(model_id)  # Update usage time
            return self._loaded_models[model_id]
        
        # Proactively free memory before loading a new model
        self._free_memory_for_new_model(keep_models=[model_id])
        
        # Get model configuration
        model_config = self.config.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"No configuration found for model: {model_id}")
        
        # Get model class - look up by model_id first, then model_type
        model_class = None
        if model_id in self._model_classes:
            model_class = self._model_classes[model_id]
        else:
            model_type = model_config.model_type.value
            if model_type in self._model_classes:
                model_class = self._model_classes[model_type]
        
        if model_class is None:
            raise ValueError(f"No implementation found for model: {model_id}")
        
        try:
            console.print(f"Loading model: {model_config.name}")
            console.print(f"Device: {device or 'auto'}")
            
            # Create and load model
            model = model_class(model_config, device)
            model.load_model()
            
            # Cache the loaded model
            self._loaded_models[model_id] = model
            self._update_model_usage(model_id)  # Track initial load time
            
            console.print(f"✓ Successfully loaded {model_config.name}")
            return model
            
        except Exception as e:
            # If we hit OOM or similar, aggressively free memory and retry once
            error_msg = str(e).lower()
            if ("out of memory" in error_msg or "cuda" in error_msg or "cudnn" in error_msg):
                console.print("[yellow]Memory error detected while loading model. Freeing memory and retrying once...[/yellow]")
                self._free_memory_for_new_model(keep_models=[])
                try:
                    model = model_class(model_config, device)
                    model.load_model()
                    self._loaded_models[model_id] = model
                    self._update_model_usage(model_id)
                    console.print(f"✓ Successfully loaded {model_config.name} after memory cleanup")
                    return model
                except Exception as e2:
                    console.print(f"[red]Retry failed for {model_config.name}: {e2}[/red]")
                    raise RuntimeError(f"Model loading failed after memory cleanup: {e2}")
            else:
                console.print(f"[red]✗ Failed to load {model_config.name}: {e}[/red]")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model was unloaded, False if not loaded
        """
        if model_id not in self._loaded_models:
            console.print(f"Model {model_id} is not loaded")
            return False
        
        model = self._loaded_models[model_id]
        model.unload_model()
        del self._loaded_models[model_id]
        
        # Clean up usage tracking
        if model_id in self._model_last_used:
            del self._model_last_used[model_id]
        
        console.print(f"✓ Unloaded model: {model_id}")
        return True
    
    def unload_all_models(self) -> None:
        """Unload all loaded models."""
        model_ids = list(self._loaded_models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model IDs.
        
        Returns:
            List of loaded model IDs
        """
        return list(self._loaded_models.keys())
    
    def predict(self, model_id: str, image, question: str) -> ModelPrediction:
        """
        Make a prediction using a specific model.
        
        Args:
            model_id: Model identifier
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            Model prediction
        """
        model = self.load_model(model_id)
        self._update_model_usage(model_id)  # Track usage on prediction
        return model.predict(image, question)
    
    def extract_attention(self, model_id: str, image, question: str) -> AttentionOutput:
        """
        Extract attention weights using a specific model.
        
        Args:
            model_id: Model identifier
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            Attention analysis output
        """
        model = self.load_model(model_id)
        if not model.config.supports_attention_extraction:
            raise ValueError(f"Model {model_id} does not support attention extraction")
        
        self._update_model_usage(model_id)  # Track usage on attention extraction
        return model.extract_attention(image, question)
    
    def compare_models(self, model_ids: List[str], image, question: str) -> Dict[str, ModelPrediction]:
        """
        Compare predictions from multiple models.
        
        Args:
            model_ids: List of model identifiers
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            Dictionary mapping model IDs to predictions
        """
        results = {}
        
        for model_id in model_ids:
            try:
                prediction = self.predict(model_id, image, question)
                results[model_id] = prediction
            except Exception as e:
                console.print(f"[red]Error with model {model_id}: {e}[/red]")
                results[model_id] = ModelPrediction(
                    answer="Error",
                    confidence=0.0,
                    processing_time=None
                )
        
        return results
    
    def get_model_info(self, model_id: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Get information about models.
        
        Args:
            model_id: Specific model ID, or None for all available models
            
        Returns:
            Model information dictionary or list of dictionaries
        """
        if model_id:
            if model_id not in self.get_available_models():
                raise ValueError(f"Model {model_id} is not available")
            
            model_config = self.config.get_model_config(model_id)
            info = {
                "id": model_id,
                "name": model_config.name,
                "type": model_config.model_type.value,
                "local_dir": str(model_config.local_dir),
                "max_image_size": model_config.max_image_size,
                "patch_size": model_config.patch_size,
                "supports_attention": model_config.supports_attention_extraction,
                "is_loaded": model_id in self._loaded_models,
                "hf_repo": model_config.hf_repo
            }
            
            # Add runtime info if loaded
            if model_id in self._loaded_models:
                model = self._loaded_models[model_id]
                runtime_info = model.get_model_info()
                info.update({
                    "device": runtime_info["device"],
                    "parameters": runtime_info["parameters"]
                })
            
            return info
        else:
            # Return info for all available models
            return [self.get_model_info(mid) for mid in self.get_available_models()]
    
    def optimize_memory(self, keep_models: Optional[List[str]] = None) -> None:
        """
        Optimize memory usage by unloading unused models.
        
        Args:
            keep_models: List of model IDs to keep loaded
        """
        if keep_models is None:
            keep_models = []
        
        to_unload = [
            model_id for model_id in self._loaded_models.keys()
            if model_id not in keep_models
        ]
        
        for model_id in to_unload:
            self.unload_model(model_id)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        console.print(f"✓ Memory optimization complete. Kept {len(keep_models)} models loaded.")
    
    def auto_discover_models(self, models_dir: Optional[Path] = None) -> None:
        """
        Automatically discover and register models from a directory.
        
        Args:
            models_dir: Directory to search for models (uses config default if None)
        """
        if models_dir is None:
            models_dir = self.config.paths.local_models_dir
        
        if not models_dir.exists():
            console.print(f"Models directory does not exist: {models_dir}")
            return
        
        discovered_count = 0
        
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Check if it looks like a model directory
            config_file = model_dir / "config.json"
            if not config_file.exists():
                continue
            
            # Try to determine model type and register
            try:
                import json
                with open(config_file) as f:
                    model_config = json.load(f)
                
                # Basic model type detection (can be improved)
                model_name = model_dir.name.lower()
                if "unichart" in model_name:
                    model_type = ModelType.UNICHART
                elif "pix2struct" in model_name:
                    model_type = ModelType.VISION_TRANSFORMER
                else:
                    # Skip unknown models
                    continue
                
                # Create configuration
                discovered_config = ModelConfig(
                    name=model_dir.name,
                    model_type=model_type,
                    local_dir=model_dir,
                    supports_attention_extraction=True
                )
                
                # Register in global config
                self.config.models[model_name] = discovered_config
                discovered_count += 1
                console.print(f"✓ Discovered model: {model_name}")
                
            except Exception as e:
                console.print(f"[yellow]Could not process {model_dir.name}: {e}[/yellow]")
        
        console.print(f"Auto-discovery complete. Found {discovered_count} models.")
    
    def preload_model(self, model_id: str, device: Optional[str] = None) -> bool:
        """
        Preload a model into memory (for auto-loading on selection).
        
        Args:
            model_id: Model identifier
            device: Device to load model on
            
        Returns:
            True if successful, False otherwise
        """
        try:
            console.print(f"🚀 Preloading model: {model_id}")
            model = self.load_model(model_id, device)
            console.print(f"✓ Model {model_id} preloaded successfully")
            return True
        except Exception as e:
            console.print(f"[red]✗ Failed to preload {model_id}: {e}[/red]")
            return False
    
    def get_model_usage_info(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Get usage information for loaded models.
        
        Returns:
            Dictionary with model usage info
        """
        current_time = time.time()
        usage_info = {}
        
        for model_id in self._loaded_models.keys():
            last_used = self._model_last_used.get(model_id, current_time)
            inactive_time = current_time - last_used
            
            usage_info[model_id] = {
                "last_used": time.strftime("%H:%M:%S", time.localtime(last_used)),
                "inactive_minutes": round(inactive_time / 60, 1),
                "auto_unload_in": max(0, round((self._inactivity_timeout - inactive_time) / 60, 1))
            }
        
        return usage_info
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_all_models()
        except:
            pass

