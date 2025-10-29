"""
Modern model download utility for ChartViz.
Supports downloading and caching of chart QA models with proper error handling.
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor, 
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, AutoConfig,
    Pix2StructForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)

from config import config, ModelConfig


console = Console()


class ModelDownloader:
    """Handles downloading and verification of chart QA models."""
    
    def __init__(self):
        self.config = config
        self.available_models = self.config.get_available_models()
    
    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        console.print("[bold blue]Checking system requirements...[/bold blue]")
        
        # Check PyTorch installation
        try:
            torch_version = torch.__version__
            console.print(f"✓ PyTorch {torch_version} detected")
        except ImportError:
            console.print("[red]✗ PyTorch not found. Please install PyTorch first.[/red]")
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"✓ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
        else:
            console.print("[yellow]⚠ CUDA not available. Models will run on CPU.[/yellow]")
        
        # Check available disk space (cross-platform)
        model_dir = self.config.paths.local_models_dir
        try:
            # Use shutil.disk_usage which works on both Windows and Unix
            total, used, free = shutil.disk_usage(model_dir)
            free_space_gb = free / (1024**3)
            console.print(f"✓ Available disk space: {free_space_gb:.1f} GB")
            
            if free_space_gb < 5:
                console.print("[yellow]⚠ Low disk space. Models may require 2-10 GB each.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not check disk space: {e}[/yellow]")
            console.print("[yellow]⚠ Please ensure you have at least 10 GB free space.[/yellow]")
        
        return True
    
    def list_available_models(self) -> None:
        """Display table of available models."""
        table = Table(title="Available Chart QA Models")
        table.add_column("ID", style="cyan", no_wrap=True, width=8)
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Size (GB)", style="red", justify="right")
        table.add_column("HuggingFace Repo", style="blue")
        table.add_column("Status", style="magenta")
        
        for i, (model_id, model_config) in enumerate(self.config.models.items(), 1):
            status = "✓ Downloaded" if self._is_model_downloaded(model_config) else "○ Not downloaded"
            
            # Estimate model size based on type and parameters
            estimated_size = self._estimate_model_size(model_config)
            
            table.add_row(
                str(i),
                f"{model_config.name} ({model_id})",
                model_config.model_type.value,
                f"{estimated_size:.1f}",
                model_config.hf_repo or "N/A",
                status
            )
        
        console.print(table)
    
    def _estimate_model_size(self, model_config) -> float:
        """Estimate model size in GB based on model type and configuration."""
        # These are rough estimates based on typical model sizes
        size_estimates = {
            "unichart": 3.2,  # UniChart is typically around 3.2GB
            "pix2struct": 2.8,  # Pix2Struct base is around 2.8GB
            "donut": 2.1,  # Donut is typically around 2.1GB
            "llava_next": 14.0,  # LLaVA-NeXT is around 14GB
            "llava": 13.0,  # LLaVA v1.5 is around 13GB
        }
        
        model_name = model_config.name.lower()
        for key, size in size_estimates.items():
            if key in model_name:
                return size
        
        # Default estimate for unknown models
        return 2.5
    
    def _load_model_with_auto_detection(self, hf_repo: str, cache_dir: str, model_id: str):
        """Load model using architecture-specific detection."""
        try:
            # Get model-specific loading strategy
            loading_strategy = self._get_model_loading_strategy(model_id, hf_repo)
            
            console.print(f"Using loading strategy for {model_id}: {loading_strategy['class'].__name__}")
            
            # Apply model-specific arguments
            kwargs = {
                "local_files_only": False,
                **loading_strategy.get('kwargs', {})
            }
            
            # Load the model with the appropriate class and arguments
            model = loading_strategy['class'].from_pretrained(hf_repo, **kwargs)
            console.print(f"✓ Successfully loaded {model_id} with {loading_strategy['class'].__name__}")
            return model
            
        except Exception as e:
            console.print(f"[red]Error loading {model_id}: {e}[/red]")
            console.print(f"[yellow]Falling back to auto-detection...[/yellow]")
            return self._fallback_auto_detection(hf_repo, cache_dir)
    
    def _get_model_loading_strategy(self, model_id: str, hf_repo: str) -> Dict:
        """Get model-specific loading strategy."""
        # Import here to avoid circular imports
        from transformers import (
            Pix2StructForConditionalGeneration,
            VisionEncoderDecoderModel,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM
        )
        
        # Import model-specific classes
        from transformers import (
            Blip2ForConditionalGeneration,
            GitForCausalLM,
            LlavaForConditionalGeneration
        )
        
        strategies = {
            "unichart": {
                "class": VisionEncoderDecoderModel,
                "kwargs": {}
            },
            "llava_v1_5_7b": {
                "class": LlavaForConditionalGeneration,
                "kwargs": {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
            },
            "llava_next_mistral_7b": {
                "class": LlavaNextForConditionalGeneration,
                "kwargs": {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
            }
        }
        
        # Return specific strategy or fallback
        return strategies.get(model_id, {
            "class": VisionEncoderDecoderModel,
            "kwargs": {}
        })
    
    def _fallback_auto_detection(self, hf_repo: str, cache_dir: str):
        """Fallback auto-detection method."""
        from transformers import (
            VisionEncoderDecoderModel,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
            AutoModel,
            Pix2StructForConditionalGeneration
        )
        
        # First, try to get the config to understand the model architecture
        try:
            config = AutoConfig.from_pretrained(hf_repo)
            if hasattr(config, 'model_type'):
                model_type = config.model_type
                console.print(f"Detected model type: {model_type}")
        except Exception as e:
            console.print(f"Could not load config: {e}")
        
        # Try different model classes based on common patterns
        model_classes_to_try = [
            VisionEncoderDecoderModel,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
            AutoModel,
        ]
        
        for model_class in model_classes_to_try:
            try:
                console.print(f"Trying {model_class.__name__}...")
                model = model_class.from_pretrained(
                    hf_repo,
                    local_files_only=False
                )
                console.print(f"✓ Successfully loaded with {model_class.__name__}")
                return model
            except Exception as e:
                console.print(f"⚠ {model_class.__name__} failed: {str(e)[:100]}...")
                continue
        
        # If all fail, raise the last exception
        raise RuntimeError("Could not load model with any of the available model classes")
    
    def _is_model_downloaded(self, model_config: ModelConfig) -> bool:
        """Check if model is already downloaded."""
        model_path = model_config.local_dir
        if not model_path.exists():
            return False
        
        # Check for essential files
        required_files = ["config.json"]
        pytorch_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
        
        has_config = any((model_path / f).exists() for f in required_files)
        has_weights = len(pytorch_files) > 0
        
        return has_config and has_weights
    
    def download_model(self, model_id: str, force_redownload: bool = False) -> bool:
        """Download a specific model."""
        if model_id not in self.available_models:
            console.print(f"[red]Error: Model '{model_id}' not found in configuration.[/red]")
            console.print("Available models:")
            self.list_available_models()
            return False
        
        model_config = self.config.get_model_config(model_id)
        
        if not model_config.hf_repo:
            console.print(f"[red]Error: No HuggingFace repository specified for {model_id}[/red]")
            return False
        
        # Check if already downloaded
        if self._is_model_downloaded(model_config) and not force_redownload:
            console.print(f"[green]✓ Model {model_id} already downloaded. Use --force to redownload.[/green]")
            return True
        
        # Show special info panel for LLaVA-NeXT
        if model_id == "llava_next_mistral_7b":
            self._show_llava_next_info()
        
        console.print(f"[bold blue]Downloading {model_config.name}...[/bold blue]")
        console.print(f"Repository: {model_config.hf_repo}")
        console.print(f"Local path: {model_config.local_dir}")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                # Create local directory
                model_config.local_dir.mkdir(parents=True, exist_ok=True)
                
                # Download model components
                download_task = progress.add_task("Downloading model...", total=3)
                
                # Download model
                progress.update(download_task, description="Downloading model weights...")
                model = self._load_model_with_auto_detection(
                    model_config.hf_repo,
                    str(model_config.local_dir),
                    model_id
                )
                model.save_pretrained(str(model_config.local_dir))
                progress.advance(download_task)
                
                # Download tokenizer
                progress.update(download_task, description="Downloading tokenizer...")
                try:
                    loading_strategy = self._get_model_loading_strategy(model_id, model_config.hf_repo)
                    tokenizer_kwargs = loading_strategy.get('kwargs', {})
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_config.hf_repo,
                        **tokenizer_kwargs
                    )
                    tokenizer.save_pretrained(str(model_config.local_dir))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not download tokenizer: {e}[/yellow]")
                progress.advance(download_task)
                
                # Download processor (for vision models)
                progress.update(download_task, description="Downloading processor...")
                try:
                    loading_strategy = self._get_model_loading_strategy(model_id, model_config.hf_repo)
                    processor_kwargs = loading_strategy.get('kwargs', {})
                    
                    # For Pix2Struct models, use the specific processor
                    if model_id in ["pix2struct", "unichart"]:
                        from transformers import Pix2StructProcessor
                        processor = Pix2StructProcessor.from_pretrained(
                            model_config.hf_repo,
                            **processor_kwargs
                        )
                    # For LLaVA-NeXT models, use the specific processor
                    elif model_id == "llava_next_mistral_7b":
                        processor = LlavaNextProcessor.from_pretrained(
                            model_config.hf_repo,
                            cache_dir=str(model_config.local_dir)
                        )
                    else:
                        processor = AutoProcessor.from_pretrained(
                            model_config.hf_repo,
                            **processor_kwargs
                        )
                    processor.save_pretrained(str(model_config.local_dir))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not download processor: {e}[/yellow]")
                progress.advance(download_task)
            
            # Verify download
            if self._is_model_downloaded(model_config):
                console.print(f"[green]✓ Successfully downloaded {model_config.name}[/green]")
                self._print_model_info(model_config)
                return True
            else:
                console.print(f"[red]✗ Download verification failed for {model_config.name}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error downloading {model_config.name}: {str(e)}[/red]")
            return False
    
    def _show_llava_next_info(self) -> None:
        """Display detailed information about LLaVA-NeXT model."""
        info_panel = Panel(
            "[bold cyan]LLaVA-NeXT (v1.6) - Mistral-7B[/bold cyan]\n\n"
            "[white]Model:[/white] llava-hf/llava-v1.6-mistral-7b-hf\n"
            "[white]Size:[/white] ~14 GB\n"
            "[white]Architecture:[/white] Vision-Language Model\n"
            "[white]Features:[/white]\n"
            "  • High-resolution image support (up to 672x672)\n"
            "  • Improved visual reasoning\n"
            "  • Mistral-7B language backbone\n"
            "  • Cross-attention extraction support\n"
            "  • Enhanced chart understanding\n",
            title="Model Information",
            border_style="blue"
        )
        console.print(info_panel)
    
    def _print_model_info(self, model_config: ModelConfig) -> None:
        """Print information about the downloaded model."""
        info_panel = Panel.fit(
            f"""[bold]{model_config.name}[/bold]
Type: {model_config.model_type.value}
Location: {model_config.local_dir}
Vision Support: {'Yes' if model_config.requires_vision else 'No'}
Max Image Size: {model_config.max_image_size}
Patch Size: {model_config.patch_size}
Attention Extraction: {'Supported' if model_config.supports_attention_extraction else 'Not Supported'}""",
            title="Model Information",
            border_style="green"
        )
        console.print(info_panel)
    
    def download_all_models(self, force_redownload: bool = False) -> None:
        """Download all available models."""
        console.print("[bold blue]Downloading all available models...[/bold blue]")
        
        success_count = 0
        total_count = len(self.available_models)
        
        for model_id in self.available_models:
            console.print(f"\n[bold]Processing {model_id} ({success_count + 1}/{total_count})[/bold]")
            if self.download_model(model_id, force_redownload):
                success_count += 1
        
        console.print(f"\n[bold green]Downloaded {success_count}/{total_count} models successfully.[/bold green]")
    
    def cleanup_models(self, model_ids: Optional[List[str]] = None) -> None:
        """Remove downloaded models to free up space."""
        if model_ids is None:
            model_ids = self.available_models
        
        console.print("[bold yellow]Cleaning up models...[/bold yellow]")
        
        for model_id in model_ids:
            if model_id not in self.available_models:
                console.print(f"[red]Warning: Unknown model {model_id}[/red]")
                continue
            
            model_config = self.config.get_model_config(model_id)
            if model_config.local_dir.exists():
                import shutil
                shutil.rmtree(model_config.local_dir)
                console.print(f"✓ Removed {model_config.name}")
            else:
                console.print(f"○ {model_config.name} was not downloaded")


def main():
    """Main interactive interface."""
    console.print(Panel.fit(
        "[bold blue]ChartViz Model Downloader[/bold blue]\n"
        "Download and manage Chart QA models for attention visualization",
        title="Welcome",
        border_style="blue"
    ))
    
    downloader = ModelDownloader()
    
    while True:
        console.print("\n[bold cyan]Available Actions:[/bold cyan]")
        console.print("1. Check system requirements")
        console.print("2. List available models")
        console.print("3. Download specific models")
        console.print("4. Download all models")
        console.print("5. Cleanup downloaded models")
        console.print("6. Exit")
        
        try:
            choice = console.input("\n[bold yellow]Select an action (1-6): [/bold yellow]")
            
            if choice == "1":
                console.print("\n[bold blue]Checking system requirements...[/bold blue]")
                if downloader.check_system_requirements():
                    console.print("[green]✓ System requirements check passed[/green]")
                else:
                    console.print("[red]✗ System requirements check failed[/red]")
            
            elif choice == "2":
                console.print("\n[bold blue]Available models:[/bold blue]")
                downloader.list_available_models()
            
            elif choice == "3":
                console.print("\n[bold blue]Available models to download:[/bold blue]")
                downloader.list_available_models()
                
                model_input = console.input("\n[bold yellow]Enter model numbers (e.g., 1,2) or model IDs (e.g., unichart,llava_v1_5_7b): [/bold yellow]").strip()
                if not model_input:
                    console.print("[red]No models specified.[/red]")
                    continue
                
                # Parse input - could be numbers or model IDs
                model_ids = []
                available_models = list(downloader.config.models.keys())
                
                for item in model_input.split(','):
                    item = item.strip()
                    if item.isdigit():
                        # It's a number
                        idx = int(item) - 1
                        if 0 <= idx < len(available_models):
                            model_ids.append(available_models[idx])
                        else:
                            console.print(f"[red]Invalid model number: {item}[/red]")
                    else:
                        # It's a model ID
                        if item in available_models:
                            model_ids.append(item)
                        else:
                            console.print(f"[red]Unknown model ID: {item}[/red]")
                
                if not model_ids:
                    console.print("[red]No valid models specified.[/red]")
                    continue
                
                console.print(f"[blue]Selected models: {', '.join(model_ids)}[/blue]")
                
                force_response = console.input("[yellow]Force redownload if exists? (y/N): [/yellow]").strip().lower()
                force_redownload = force_response in ['y', 'yes']
                
                if not downloader.check_system_requirements():
                    console.print("[red]System requirements check failed. Cannot proceed.[/red]")
                    continue
                
                # Download each selected model
                success_count = 0
                for model_id in model_ids:
                    console.print(f"\n[bold]Downloading {model_id}...[/bold]")
                    success = downloader.download_model(model_id, force_redownload)
                    if success:
                        console.print(f"[green]✓ Successfully downloaded {model_id}[/green]")
                        success_count += 1
                    else:
                        console.print(f"[red]✗ Failed to download {model_id}[/red]")
                
                console.print(f"\n[bold]Downloaded {success_count}/{len(model_ids)} models successfully.[/bold]")
            
            elif choice == "4":
                console.print("\n[bold blue]This will download all available models.[/bold blue]")
                confirm = console.input("[yellow]Are you sure? This may take a while and use significant disk space (y/N): [/yellow]").strip().lower()
                
                if confirm not in ['y', 'yes']:
                    console.print("[yellow]Download cancelled.[/yellow]")
                    continue
                
                force_response = console.input("[yellow]Force redownload existing models? (y/N): [/yellow]").strip().lower()
                force_redownload = force_response in ['y', 'yes']
                
                if not downloader.check_system_requirements():
                    console.print("[red]System requirements check failed. Cannot proceed.[/red]")
                    continue
                
                downloader.download_all_models(force_redownload)
                console.print("[green]✓ Batch download completed[/green]")
            
            elif choice == "5":
                console.print("\n[bold blue]Downloaded models:[/bold blue]")
                downloader.list_available_models()
                
                cleanup_choice = console.input("\n[yellow]Cleanup all models or specific ones? (all/specific/cancel): [/yellow]").strip().lower()
                
                if cleanup_choice == "cancel":
                    console.print("[yellow]Cleanup cancelled.[/yellow]")
                    continue
                elif cleanup_choice == "all":
                    confirm = console.input("[red]Are you sure you want to delete ALL downloaded models? (yes/N): [/red]").strip()
                    if confirm == "yes":
                        downloader.cleanup_models()
                        console.print("[green]✓ All models cleaned up[/green]")
                    else:
                        console.print("[yellow]Cleanup cancelled.[/yellow]")
                elif cleanup_choice == "specific":
                    model_id = console.input("[yellow]Enter model ID to remove: [/yellow]").strip()
                    if model_id:
                        confirm = console.input(f"[red]Delete {model_id}? (y/N): [/red]").strip().lower()
                        if confirm in ['y', 'yes']:
                            downloader.cleanup_models([model_id])
                            console.print(f"[green]✓ {model_id} cleaned up[/green]")
                        else:
                            console.print("[yellow]Cleanup cancelled.[/yellow]")
                else:
                    console.print("[red]Invalid choice.[/red]")
            
            elif choice == "6":
                console.print("[green]Goodbye! 👋[/green]")
                break
            
            else:
                console.print("[red]Invalid choice. Please select 1-6.[/red]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            continue


def main_cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Download ChartViz models")
    parser.add_argument("--model", type=str, help="Specific model to download")
    parser.add_argument("--progress", action="store_true", help="Output progress for external monitoring")
    parser.add_argument("--interactive", action="store_true", default=True, help="Run interactive mode")
    
    args = parser.parse_args()
    
    if args.model:
        # Single model download mode
        downloader = ModelDownloader()
        try:
            if args.progress:
                # Output progress for external monitoring
                print(f"Starting download: {args.model}")
            
            success = downloader.download_model(args.model)
            
            if success:
                if args.progress:
                    print(f"Download complete: {args.model}")
                sys.exit(0)
            else:
                if args.progress:
                    print(f"Download failed: {args.model}")
                sys.exit(1)
                
        except Exception as e:
            if args.progress:
                print(f"Download error: {args.model} - {e}")
            sys.exit(1)
    else:
        # Interactive mode
        main()


if __name__ == "__main__":
    main_cli()
