"""
Model scanner for detecting and listing available local models.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a detected model."""
    id: str
    name: str
    path: Path
    model_type: str
    size_mb: float
    config_available: bool
    weights_available: bool
    tokenizer_available: bool


class ModelScanner:
    """Scanner for local models in the LocalModels directory."""
    
    def __init__(self, models_dir: str = "LocalModels"):
        """
        Initialize the model scanner.
        
        Args:
            models_dir: Directory containing local models
        """
        self.models_dir = Path(models_dir)
        self.supported_weight_formats = ['.bin', '.safetensors', '.pt', '.pth']
        self.required_files = ['config.json']
    
    def scan_models(self) -> List[ModelInfo]:
        """
        Scan the models directory and return information about available models.
        
        Returns:
            List of ModelInfo objects for detected models
        """
        models = []
        
        if not self.models_dir.exists():
            return models
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_info = self._analyze_model_directory(model_dir)
                if model_info:
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x.name)
    
    def _analyze_model_directory(self, model_dir: Path) -> Optional[ModelInfo]:
        """
        Analyze a model directory to extract model information.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            ModelInfo object if valid model found, None otherwise
        """
        # Check for required config file
        config_file = model_dir / "config.json"
        if not config_file.exists():
            return None
        
        try:
            # Read model config
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Extract model type
            model_type = self._determine_model_type(config, model_dir)
            
            # Check for weight files
            weight_files = self._find_weight_files(model_dir)
            weights_available = len(weight_files) > 0
            
            # Check for tokenizer files
            tokenizer_available = self._check_tokenizer_files(model_dir)
            
            # Calculate directory size
            size_mb = self._calculate_directory_size(model_dir)
            
            # Generate model ID and name
            model_id = model_dir.name.lower().replace(' ', '_')
            model_name = config.get('name', model_dir.name)
            
            return ModelInfo(
                id=model_id,
                name=model_name,
                path=model_dir,
                model_type=model_type,
                size_mb=size_mb,
                config_available=True,
                weights_available=weights_available,
                tokenizer_available=tokenizer_available
            )
            
        except Exception as e:
            print(f"Error analyzing model directory {model_dir}: {e}")
            return None
    
    def _determine_model_type(self, config: Dict, model_dir: Path) -> str:
        """
        Determine the model type from config and directory structure.
        
        Args:
            config: Model configuration dictionary
            model_dir: Path to model directory
            
        Returns:
            Model type string
        """
        # Check config for model type hints
        if 'model_type' in config:
            return config['model_type']
        
        if 'architectures' in config:
            arch = config['architectures'][0] if config['architectures'] else ''
            if 'pix2struct' in arch.lower():
                return 'pix2struct'
            elif 'unichart' in arch.lower():
                return 'unichart'
            elif 'donut' in arch.lower():
                return 'donut'
        
        # Check directory name for hints
        dir_name = model_dir.name.lower()
        if 'unichart' in dir_name:
            return 'unichart'
        elif 'pix2struct' in dir_name:
            return 'pix2struct'
        elif 'donut' in dir_name:
            return 'donut'
        
        return 'unknown'
    
    def _find_weight_files(self, model_dir: Path) -> List[Path]:
        """Find weight files in the model directory."""
        weight_files = []
        
        for file_path in model_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_weight_formats:
                weight_files.append(file_path)
        
        return weight_files
    
    def _check_tokenizer_files(self, model_dir: Path) -> bool:
        """Check if tokenizer files are available."""
        tokenizer_files = [
            'tokenizer.json',
            'tokenizer_config.json', 
            'vocab.json',
            'merges.txt',
            'sentencepiece.bpe.model'
        ]
        
        for tokenizer_file in tokenizer_files:
            if (model_dir / tokenizer_file).exists():
                return True
        
        return False
    
    def _calculate_directory_size(self, model_dir: Path) -> float:
        """Calculate the total size of a directory in MB."""
        total_size = 0
        
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_available_model_ids(self) -> List[str]:
        """Get list of available model IDs."""
        models = self.scan_models()
        return [model.id for model in models if model.weights_available]
    
    def get_model_info_dict(self) -> Dict[str, Dict]:
        """Get model information as a dictionary."""
        models = self.scan_models()
        info_dict = {}
        
        for model in models:
            info_dict[model.id] = {
                'name': model.name,
                'type': model.model_type,
                'path': str(model.path),
                'size_mb': round(model.size_mb, 1),
                'config_available': model.config_available,
                'weights_available': model.weights_available,
                'tokenizer_available': model.tokenizer_available,
                'status': 'ready' if model.weights_available else 'incomplete'
            }
        
        return info_dict


def scan_local_models() -> List[str]:
    """
    Quick function to scan and return available model IDs.
    
    Returns:
        List of model IDs that are ready to use
    """
    scanner = ModelScanner()
    return scanner.get_available_model_ids()


def get_model_details() -> Dict[str, Dict]:
    """
    Quick function to get detailed model information.
    
    Returns:
        Dictionary with model details
    """
    scanner = ModelScanner()
    return scanner.get_model_info_dict()


if __name__ == "__main__":
    # Test the scanner
    scanner = ModelScanner()
    models = scanner.scan_models()
    
    print("Available Models:")
    for model in models:
        print(f"- {model.name} ({model.id})")
        print(f"  Type: {model.model_type}")
        print(f"  Size: {model.size_mb:.1f} MB")
        print(f"  Status: {'Ready' if model.weights_available else 'Incomplete'}")
        print()


