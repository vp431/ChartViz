"""
Modern dataset download utility for ChartViz.
Supports downloading and preprocessing chart QA datasets with proper validation.
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
import pandas as pd
from datasets import load_dataset, Dataset
from PIL import Image
import requests
from urllib.parse import urlparse

from config import config, DatasetConfig


console = Console()


class DatasetDownloader:
    """Handles downloading and preprocessing of chart QA datasets."""
    
    def __init__(self):
        self.config = config
        self.available_datasets = self.config.get_available_datasets()
    
    def list_available_datasets(self) -> None:
        """Display table of available datasets."""
        table = Table(title="Available Chart QA Datasets")
        table.add_column("ID", style="cyan", no_wrap=True, width=8)
        table.add_column("Name", style="green")
        table.add_column("Size (GB)", style="red", justify="right")
        table.add_column("HuggingFace Repo", style="blue")
        table.add_column("Status", style="magenta")
        
        for i, (dataset_id, dataset_config) in enumerate(self.config.datasets.items(), 1):
            status = "✓ Downloaded" if self._is_dataset_downloaded(dataset_config) else "○ Not downloaded"
            
            # Estimate dataset size based on type and samples
            estimated_size = self._estimate_dataset_size(dataset_config)
            
            table.add_row(
                str(i),
                f"{dataset_config.name} ({dataset_id})",
                f"{estimated_size:.1f}",
                dataset_config.hf_repo or "N/A",
                status
            )
        
        console.print(table)
    
    def _estimate_dataset_size(self, dataset_config) -> float:
        """Estimate dataset size in GB based on dataset type and samples."""
        # These are rough estimates based on typical dataset sizes (full datasets)
        size_estimates = {
            "chartqa": 8.5,  # ChartQA full dataset is typically around 8.5GB
            "plotqa": 12.3,  # PlotQA is larger, around 12.3GB
            "figureqa": 6.2,  # FigureQA is around 6.2GB
        }
        
        dataset_name = dataset_config.name.lower()
        for key, size in size_estimates.items():
            if key in dataset_name:
                return size
        
        # Default estimate for unknown datasets
        return 5.0
    
    def _is_dataset_downloaded(self, dataset_config: DatasetConfig) -> bool:
        """Check if dataset is already downloaded."""
        dataset_path = dataset_config.local_dir
        if not dataset_path.exists():
            return False
        
        # Check for essential files
        metadata_file = dataset_path / "metadata.json"
        # Check for either full dataset or split-specific data files
        data_file_full = dataset_path / "data_full.json"
        data_file_generic = dataset_path / "data.json"
        images_dir = dataset_path / "images"
        
        return metadata_file.exists() and (data_file_full.exists() or data_file_generic.exists() or images_dir.exists())
    
    def download_dataset(self, dataset_id: str, 
                        force_redownload: bool = False,
                        include_images: bool = True) -> bool:
        """Download a specific dataset."""
        if dataset_id not in self.available_datasets:
            console.print(f"[red]Error: Dataset '{dataset_id}' not found in configuration.[/red]")
            console.print("Available datasets:")
            self.list_available_datasets()
            return False
        
        dataset_config = self.config.get_dataset_config(dataset_id)
        
        # Check if already downloaded
        if self._is_dataset_downloaded(dataset_config) and not force_redownload:
            console.print(f"[green]✓ Dataset {dataset_id} already downloaded. Use --force to redownload.[/green]")
            return True
        
        # Always download full dataset with all splits combined
        return self._download_full_dataset(dataset_id, dataset_config, force_redownload, include_images)
    
    def _download_full_dataset(self, dataset_id: str, dataset_config: DatasetConfig, 
                              force_redownload: bool, include_images: bool) -> bool:
        """Download all available splits of a dataset and combine them."""
        console.print(f"[bold blue]Downloading {dataset_config.name} (FULL dataset - all splits)...[/bold blue]")
        console.print(f"Repository: {dataset_config.hf_repo}")
        console.print(f"Local path: {dataset_config.local_dir}")
        
        try:
            # Create local directory
            dataset_config.local_dir.mkdir(parents=True, exist_ok=True)
            images_dir = dataset_config.local_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Get available splits for the dataset
            try:
                dataset_info = load_dataset(dataset_config.hf_repo, streaming=True)
                available_splits = list(dataset_info.keys())
                console.print(f"[cyan]Available splits: {', '.join(available_splits)}[/cyan]")
            except Exception as e:
                # Fallback to common splits if we can't detect them
                available_splits = ["train", "test", "validation"]
                console.print(f"[yellow]Could not detect splits, trying common ones: {', '.join(available_splits)}[/yellow]")
            
            all_processed_data = []
            total_failed_downloads = 0
            total_examples = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                main_task = progress.add_task("Downloading all splits...", total=len(available_splits))
                
                for split_idx, split in enumerate(available_splits):
                    try:
                        progress.update(main_task, description=f"Loading {split} split...")
                        
                        # Try to load the split
                        try:
                            dataset = load_dataset(dataset_config.hf_repo, split=split, streaming=False)
                            console.print(f"[green]✓ Loaded {split} split: {len(dataset)} examples[/green]")
                        except Exception as e:
                            console.print(f"[yellow]⚠ Split '{split}' not available, skipping...[/yellow]")
                            progress.advance(main_task)
                            continue
                        
                        # Process this split
                        progress.update(main_task, description=f"Processing {split} split...")
                        split_failed_downloads = 0
                        
                        for i, example in enumerate(dataset):
                            try:
                                processed_example = self._process_example(
                                    example, 
                                    dataset_config, 
                                    images_dir,
                                    split,
                                    i,
                                    include_images=include_images
                                )
                                if processed_example:
                                    # Add split information to the example
                                    processed_example["original_split"] = split
                                    all_processed_data.append(processed_example)
                            except Exception as e:
                                split_failed_downloads += 1
                                console.print(f"[yellow]Warning: Failed to process {split} example {i}: {e}[/yellow]")
                                continue
                        
                        total_failed_downloads += split_failed_downloads
                        total_examples += len(dataset)
                        console.print(f"[green]✓ Processed {split} split: {len(dataset) - split_failed_downloads}/{len(dataset)} examples[/green]")
                        
                        progress.advance(main_task)
                        
                    except Exception as e:
                        console.print(f"[red]Error processing split {split}: {e}[/red]")
                        progress.advance(main_task)
                        continue
                
                progress.update(main_task, description="Saving combined dataset...")
            
            if not all_processed_data:
                console.print(f"[red]No data was successfully processed for {dataset_id}[/red]")
                return False
            
            # Save combined data
            self._save_dataset(dataset_config, all_processed_data, "full")
            
            # Save metadata
            metadata = {
                "dataset_id": dataset_id,
                "name": dataset_config.name,
                "hf_repo": dataset_config.hf_repo,
                "split": "full",
                "available_splits": available_splits,
                "total_examples": len(all_processed_data),
                "original_total_examples": total_examples,
                "failed_downloads": total_failed_downloads,
                "download_timestamp": pd.Timestamp.now().isoformat(),
                "includes_images": include_images,
                "schema": self._get_dataset_schema(all_processed_data[0] if all_processed_data else {})
            }
            
            metadata_file = dataset_config.local_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self._print_dataset_info(dataset_config, metadata)
            console.print(f"[green]✓ Successfully downloaded full dataset: {len(all_processed_data)} examples from {len(available_splits)} splits[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error downloading full dataset {dataset_id}: {e}[/red]")
            return False
    
    def _process_example(self, example: Dict[str, Any], 
                        dataset_config: DatasetConfig,
                        images_dir: Path,
                        split: str,
                        index: int,
                        include_images: bool = True) -> Optional[Dict[str, Any]]:
        """Process a single dataset example."""
        processed = {
            "question": example.get("query", ""), # Fix: Use 'query' instead of 'question'
            "answer": example.get("answer", ""),
            "chart_type": example.get("chart_type", "unknown"),
            "source": example.get("source", ""),
        }
        
        # Handle image
        if include_images and "image" in example:
            image_data = example["image"]
            
            # Generate unique filename
            image_filename = f"{split}_{index}.png"
            image_path = images_dir / image_filename
            
            try:
                image_saved = False
                
                if isinstance(image_data, str):
                    # URL
                    if image_data.startswith(("http://", "https://")):
                        response = requests.get(image_data, timeout=30)
                        response.raise_for_status()
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                        image_saved = True
                    else:
                        # Base64 or local path
                        console.print(f"[yellow]Warning: Unsupported image format for {image_filename}[/yellow]")
                elif hasattr(image_data, 'save'):
                    # PIL Image
                    image_data.save(image_path)
                    image_saved = True
                else:
                    console.print(f"[yellow]Warning: Unknown image type for {image_filename}[/yellow]")

                if image_saved:
                    processed["image_path"] = str(image_path.relative_to(dataset_config.local_dir))
                    processed["image_filename"] = image_filename
                    
                    # Extract basic image metadata
                    with Image.open(image_path) as img:
                        processed["image_width"] = img.width
                        processed["image_height"] = img.height
                        processed["image_mode"] = img.mode
                else:
                    processed["image_path"] = None
                
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to save image for {image_filename}: {e}[/yellow]")
                processed["image_path"] = None
        
        # Add additional metadata
        processed["example_id"] = example.get("id", f"{split}_{index}")
        processed["metadata"] = {
            k: v for k, v in example.items() 
            if k not in ["question", "answer", "image", "chart_type", "source", "id"]
        }
        
        return processed
    
    def _save_dataset(self, dataset_config: DatasetConfig, 
                     processed_data: List[Dict[str, Any]], 
                     split: str) -> None:
        """Save processed dataset to local files."""
        # Save as JSON
        data_file = dataset_config.local_dir / f"data_{split}.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy inspection
        df = pd.json_normalize(processed_data)
        csv_file = dataset_config.local_dir / f"data_{split}.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8")
        
        console.print(f"✓ Saved data to {data_file}")
        console.print(f"✓ Saved CSV to {csv_file}")
    
    def _get_dataset_schema(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Extract schema information from an example."""
        schema = {}
        for key, value in example.items():
            if isinstance(value, str):
                schema[key] = "string"
            elif isinstance(value, int):
                schema[key] = "integer"
            elif isinstance(value, float):
                schema[key] = "float"
            elif isinstance(value, bool):
                schema[key] = "boolean"
            elif isinstance(value, dict):
                schema[key] = "object"
            elif isinstance(value, list):
                schema[key] = "array"
            else:
                schema[key] = "unknown"
        return schema
    
    def _print_dataset_info(self, dataset_config: DatasetConfig, metadata: Dict[str, Any]) -> None:
        """Print information about the downloaded dataset."""
        info_panel = Panel.fit(
            f"""[bold]{dataset_config.name}[/bold]
Location: {dataset_config.local_dir}
Total Examples: {metadata['total_examples']}
Failed Downloads: {metadata['failed_downloads']}
Split: {metadata['split']}
Includes Images: {metadata['includes_images']}
Schema: {', '.join(metadata['schema'].keys())}""",
            title="Dataset Information",
            border_style="green"
        )
        console.print(info_panel)
    
    def download_all_datasets(self, force_redownload: bool = False) -> None:
        """Download all available datasets (full versions with images)."""
        console.print("[bold blue]Downloading all available datasets (FULL versions - all splits with images)...[/bold blue]")
        
        success_count = 0
        total_count = len(self.available_datasets)
        
        for dataset_id in self.available_datasets:
            console.print(f"\n[bold]Processing {dataset_id} ({success_count + 1}/{total_count})[/bold]")
            
            if self.download_dataset(dataset_id, force_redownload=force_redownload):
                success_count += 1
        
        console.print(f"\n[bold green]Downloaded {success_count}/{total_count} datasets successfully.[/bold green]")
    
    def validate_dataset(self, dataset_id: str) -> bool:
        """Validate a downloaded dataset."""
        if dataset_id not in self.available_datasets:
            console.print(f"[red]Error: Dataset '{dataset_id}' not found.[/red]")
            return False
        
        dataset_config = self.config.get_dataset_config(dataset_id)
        
        if not self._is_dataset_downloaded(dataset_config):
            console.print(f"[red]Dataset {dataset_id} is not downloaded.[/red]")
            return False
        
        console.print(f"[blue]Validating {dataset_config.name}...[/blue]")
        
        try:
            # Load metadata
            metadata_file = dataset_config.local_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Load data
            data_file = dataset_config.local_dir / "data_full.json"
            with open(data_file) as f:
                data = json.load(f)
            
            # Basic validation
            console.print(f"✓ Found {len(data)} examples")
            console.print(f"✓ Metadata intact")
            
            # Validate image files if applicable
            if metadata.get("includes_images", False):
                images_dir = dataset_config.local_dir / "images"
                image_count = len(list(images_dir.glob("*.png")))
                console.print(f"✓ Found {image_count} image files")
            
            console.print(f"[green]✓ Dataset {dataset_id} validation passed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Dataset validation failed: {e}[/red]")
            return False
    
    def cleanup_datasets(self, dataset_ids: Optional[List[str]] = None) -> None:
        """Remove downloaded datasets to free up space."""
        if dataset_ids is None:
            dataset_ids = self.available_datasets
        
        console.print("[bold yellow]Cleaning up datasets...[/bold yellow]")
        
        for dataset_id in dataset_ids:
            if dataset_id not in self.available_datasets:
                console.print(f"[red]Warning: Unknown dataset {dataset_id}[/red]")
                continue
            
            dataset_config = self.config.get_dataset_config(dataset_id)
            if dataset_config.local_dir.exists():
                import shutil
                shutil.rmtree(dataset_config.local_dir)
                console.print(f"✓ Removed {dataset_config.name}")
            else:
                console.print(f"○ {dataset_config.name} was not downloaded")


def main():
    """Main interactive interface."""
    console.print(Panel.fit(
        "[bold green]ChartViz Dataset Downloader[/bold green]\n"
        "Download and manage Chart QA datasets for model training and evaluation",
        title="Welcome",
        border_style="green"
    ))
    
    downloader = DatasetDownloader()
    
    while True:
        console.print("\n[bold cyan]Available Actions:[/bold cyan]")
        console.print("1. List available datasets")
        console.print("2. Download specific datasets")
        console.print("3. Download all datasets")
        console.print("4. Validate downloaded datasets")
        console.print("5. Cleanup downloaded datasets")
        console.print("6. Exit")
        
        try:
            choice = console.input("\n[bold yellow]Select an action (1-6): [/bold yellow]")
            
            if choice == "1":
                console.print("\n[bold green]Available datasets:[/bold green]")
                downloader.list_available_datasets()
            
            elif choice == "2":
                console.print("\n[bold green]Available datasets to download:[/bold green]")
                downloader.list_available_datasets()
                
                dataset_input = console.input("\n[bold yellow]Enter dataset numbers (e.g., 1,3) or dataset IDs (e.g., chartqa,plotqa): [/bold yellow]").strip()
                if not dataset_input:
                    console.print("[red]No datasets specified.[/red]")
                    continue
                
                # Parse input - could be numbers or dataset IDs
                dataset_ids = []
                available_datasets = list(downloader.config.datasets.keys())
                
                for item in dataset_input.split(','):
                    item = item.strip()
                    if item.isdigit():
                        # It's a number
                        idx = int(item) - 1
                        if 0 <= idx < len(available_datasets):
                            dataset_ids.append(available_datasets[idx])
                        else:
                            console.print(f"[red]Invalid dataset number: {item}[/red]")
                    else:
                        # It's a dataset ID
                        if item in available_datasets:
                            dataset_ids.append(item)
                        else:
                            console.print(f"[red]Unknown dataset ID: {item}[/red]")
                
                if not dataset_ids:
                    console.print("[red]No valid datasets specified.[/red]")
                    continue
                
                console.print(f"[blue]Selected datasets: {', '.join(dataset_ids)}[/blue]")
                console.print("[yellow]Note: Full datasets with images will be downloaded (all splits combined)[/yellow]")
                
                # Get download options
                force_response = console.input("[yellow]Force redownload if exists? (y/N): [/yellow]").strip().lower()
                force_redownload = force_response in ['y', 'yes']
                
                no_images_response = console.input("[yellow]Skip downloading images? (y/N): [/yellow]").strip().lower()
                include_images = not (no_images_response in ['y', 'yes'])
                
                # Download each selected dataset (full with images)
                success_count = 0
                for dataset_id in dataset_ids:
                    console.print(f"\n[bold]Downloading {dataset_id} (FULL dataset with images)...[/bold]")
                    success = downloader.download_dataset(
                        dataset_id,
                        force_redownload=force_redownload,
                        include_images=include_images
                    )
                    
                    if success:
                        console.print(f"[green]✓ Successfully downloaded {dataset_id}[/green]")
                        success_count += 1
                    else:
                        console.print(f"[red]✗ Failed to download {dataset_id}[/red]")
                
                console.print(f"\n[bold]Downloaded {success_count}/{len(dataset_ids)} datasets successfully.[/bold]")
            
            elif choice == "3":
                console.print("\n[bold green]This will download ALL available datasets (FULL versions with images).[/bold green]")
                console.print("[yellow]Warning: This will download very large datasets (20+ GB total).[/yellow]")
                confirm = console.input("[yellow]Are you sure? This may take hours and use significant disk space (y/N): [/yellow]").strip().lower()
                
                if confirm not in ['y', 'yes']:
                    console.print("[yellow]Download cancelled.[/yellow]")
                    continue
                
                force_response = console.input("[yellow]Force redownload existing datasets? (y/N): [/yellow]").strip().lower()
                force_redownload = force_response in ['y', 'yes']
                
                downloader.download_all_datasets(force_redownload=force_redownload)
                console.print("[green]✓ Batch download completed[/green]")
            
            elif choice == "4":
                console.print("\n[bold green]Dataset Validation[/bold green]")
                validate_choice = console.input("[yellow]Validate all datasets or specific one? (all/specific): [/yellow]").strip().lower()
                
                if validate_choice == "all":
                    console.print("[blue]Validating all downloaded datasets...[/blue]")
                    all_valid = True
                    for dataset_id in downloader.available_datasets:
                        if downloader._is_dataset_downloaded(downloader.config.get_dataset_config(dataset_id)):
                            valid = downloader.validate_dataset(dataset_id)
                            if not valid:
                                all_valid = False
                    
                    if all_valid:
                        console.print("[green]✓ All datasets validation passed[/green]")
                    else:
                        console.print("[red]✗ Some datasets failed validation[/red]")
                
                elif validate_choice == "specific":
                    dataset_id = console.input("[yellow]Enter dataset ID to validate: [/yellow]").strip()
                    if dataset_id:
                        valid = downloader.validate_dataset(dataset_id)
                        if valid:
                            console.print(f"[green]✓ {dataset_id} validation passed[/green]")
                        else:
                            console.print(f"[red]✗ {dataset_id} validation failed[/red]")
                else:
                    console.print("[red]Invalid choice.[/red]")
            
            elif choice == "5":
                console.print("\n[bold green]Downloaded datasets:[/bold green]")
                downloader.list_available_datasets()
                
                cleanup_choice = console.input("\n[yellow]Cleanup all datasets or specific ones? (all/specific/cancel): [/yellow]").strip().lower()
                
                if cleanup_choice == "cancel":
                    console.print("[yellow]Cleanup cancelled.[/yellow]")
                    continue
                elif cleanup_choice == "all":
                    confirm = console.input("[red]Are you sure you want to delete ALL downloaded datasets? (yes/N): [/red]").strip()
                    if confirm == "yes":
                        downloader.cleanup_datasets()
                        console.print("[green]✓ All datasets cleaned up[/green]")
                    else:
                        console.print("[yellow]Cleanup cancelled.[/yellow]")
                elif cleanup_choice == "specific":
                    dataset_id = console.input("[yellow]Enter dataset ID to remove: [/yellow]").strip()
                    if dataset_id:
                        confirm = console.input(f"[red]Delete {dataset_id}? (y/N): [/red]").strip().lower()
                        if confirm in ['y', 'yes']:
                            downloader.cleanup_datasets([dataset_id])
                            console.print(f"[green]✓ {dataset_id} cleaned up[/green]")
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


if __name__ == "__main__":
    main()
