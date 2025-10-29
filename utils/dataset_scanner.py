"""
Dataset scanner for detecting and listing available local datasets.
"""
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Information about a detected dataset."""
    id: str
    name: str
    path: Path
    sample_count: int
    has_images: bool
    has_questions: bool
    has_answers: bool
    size_mb: float
    format_type: str  # 'json', 'csv', 'mixed'


@dataclass
class DatasetSample:
    """Information about a dataset sample."""
    id: str
    question: str
    answer: Optional[str]
    image_path: Optional[Path]
    metadata: Optional[Dict]


class DatasetScanner:
    """Scanner for local datasets in the LocalDatasets directory."""
    
    def __init__(self, datasets_dir: str = "LocalDatasets"):
        """
        Initialize the dataset scanner.
        
        Args:
            datasets_dir: Directory containing local datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.supported_image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.supported_data_formats = ['.json', '.csv', '.jsonl']
    
    def scan_datasets(self) -> List[DatasetInfo]:
        """
        Scan the datasets directory and return information about available datasets.
        
        Returns:
            List of DatasetInfo objects for detected datasets
        """
        datasets = []
        
        if not self.datasets_dir.exists():
            return datasets
        
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_info = self._analyze_dataset_directory(dataset_dir)
                if dataset_info:
                    datasets.append(dataset_info)
        
        return sorted(datasets, key=lambda x: x.name)
    
    def _analyze_dataset_directory(self, dataset_dir: Path) -> Optional[DatasetInfo]:
        """
        Analyze a dataset directory to extract dataset information.
        
        Args:
            dataset_dir: Path to dataset directory
            
        Returns:
            DatasetInfo object if valid dataset found, None otherwise
        """
        try:
            # Find data files
            data_files = self._find_data_files(dataset_dir)
            if not data_files:
                return None
            
            # Find image directory
            images_dir = self._find_images_directory(dataset_dir)
            has_images = images_dir is not None
            
            # Count images if available
            image_count = 0
            if has_images:
                image_count = len(self._find_image_files(images_dir))
            
            # Analyze data structure
            sample_count, has_questions, has_answers, format_type = self._analyze_data_structure(data_files)
            
            # Calculate directory size
            size_mb = self._calculate_directory_size(dataset_dir)
            
            # Generate dataset ID and name
            dataset_id = dataset_dir.name.lower().replace(' ', '_')
            dataset_name = self._get_dataset_name(dataset_dir)
            
            return DatasetInfo(
                id=dataset_id,
                name=dataset_name,
                path=dataset_dir,
                sample_count=sample_count,
                has_images=has_images,
                has_questions=has_questions,
                has_answers=has_answers,
                size_mb=size_mb,
                format_type=format_type
            )
            
        except Exception as e:
            print(f"Error analyzing dataset directory {dataset_dir}: {e}")
            return None
    
    def _find_data_files(self, dataset_dir: Path) -> List[Path]:
        """Find data files in the dataset directory."""
        data_files = []
        
        for file_path in dataset_dir.iterdir():
            if file_path.is_file() and file_path.suffix in self.supported_data_formats:
                data_files.append(file_path)
        
        return data_files
    
    def _find_images_directory(self, dataset_dir: Path) -> Optional[Path]:
        """Find the images directory within the dataset."""
        # Check common image directory names
        common_names = ['images', 'imgs', 'charts', 'figures', 'plots']
        
        for name in common_names:
            img_dir = dataset_dir / name
            if img_dir.exists() and img_dir.is_dir():
                return img_dir
        
        # Check if images are in the root directory
        image_files = [f for f in dataset_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in self.supported_image_formats]
        
        if image_files:
            return dataset_dir
        
        return None
    
    def _find_image_files(self, images_dir: Path) -> List[Path]:
        """Find image files in the given directory."""
        image_files = []
        
        for file_path in images_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_image_formats:
                image_files.append(file_path)
        
        return image_files
    
    def _analyze_data_structure(self, data_files: List[Path]) -> Tuple[int, bool, bool, str]:
        """
        Analyze the structure of data files.
        
        Returns:
            Tuple of (sample_count, has_questions, has_answers, format_type)
        """
        sample_count = 0
        has_questions = False
        has_answers = False
        format_type = "unknown"
        
        for data_file in data_files:
            try:
                if data_file.suffix == '.json':
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            sample_count += len(data)
                            if data:
                                sample = data[0]
                                has_questions = 'question' in sample or 'query' in sample
                                has_answers = 'answer' in sample or 'response' in sample
                        format_type = "json"
                
                elif data_file.suffix == '.csv':
                    df = pd.read_csv(data_file)
                    sample_count += len(df)
                    columns = [col.lower() for col in df.columns]
                    has_questions = any('question' in col for col in columns)
                    has_answers = any('answer' in col for col in columns)
                    format_type = "csv"
                
                elif data_file.suffix == '.jsonl':
                    with open(data_file, 'r') as f:
                        lines = f.readlines()
                        sample_count += len(lines)
                        if lines:
                            sample = json.loads(lines[0])
                            has_questions = 'question' in sample or 'query' in sample
                            has_answers = 'answer' in sample or 'response' in sample
                        format_type = "jsonl"
                        
            except Exception as e:
                print(f"Error reading data file {data_file}: {e}")
                continue
        
        return sample_count, has_questions, has_answers, format_type
    
    def _get_dataset_name(self, dataset_dir: Path) -> str:
        """Get a readable name for the dataset."""
        # Check for metadata file
        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'name' in metadata:
                        return metadata['name']
            except:
                pass
        
        # Use directory name as fallback
        return dataset_dir.name.replace('_', ' ').title()
    
    def _calculate_directory_size(self, dataset_dir: Path) -> float:
        """Calculate the total size of a directory in MB."""
        total_size = 0
        
        for file_path in dataset_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_dataset_samples(self, dataset_id: str, limit: int = 20) -> List[DatasetSample]:
        """
        Get samples from a specific dataset.
        
        Args:
            dataset_id: ID of the dataset
            limit: Maximum number of samples to return
            
        Returns:
            List of DatasetSample objects
        """
        datasets = self.scan_datasets()
        dataset_info = next((d for d in datasets if d.id == dataset_id), None)
        
        if not dataset_info:
            return []
        
        samples = []
        data_files = self._find_data_files(dataset_info.path)
        
        for data_file in data_files:
            try:
                if data_file.suffix == '.json':
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        for i, item in enumerate(data[:limit]):
                            sample = self._create_dataset_sample(i, item, dataset_info)
                            if sample:
                                samples.append(sample)
                
                elif data_file.suffix == '.csv':
                    df = pd.read_csv(data_file)
                    for i, row in df.head(limit).iterrows():
                        sample = self._create_dataset_sample(i, row.to_dict(), dataset_info)
                        if sample:
                            samples.append(sample)
                
                if len(samples) >= limit:
                    break
                    
            except Exception as e:
                print(f"Error reading samples from {data_file}: {e}")
                continue
        
        return samples[:limit]
    
    def _create_dataset_sample(self, idx: int, data: Dict, dataset_info: DatasetInfo) -> Optional[DatasetSample]:
        """Create a DatasetSample from raw data."""
        try:
            # Extract question
            question = data.get('question') or data.get('query') or f"Sample {idx + 1}"
            
            # Extract answer
            answer = data.get('answer') or data.get('response')
            
            # Find corresponding image
            image_path = None
            if dataset_info.has_images:
                images_dir = self._find_images_directory(dataset_info.path)
                if images_dir:
                    # Try different ways to find the image
                    image_candidates = []
                    
                    # Check for direct image_path field (like ChartQA)
                    if 'image_path' in data and data['image_path']:
                        image_rel_path = data['image_path']
                        # Remove 'images/' prefix if present since images_dir already points to images folder
                        if image_rel_path.startswith('images/'):
                            image_rel_path = image_rel_path.replace('images/', '', 1)
                        image_candidates.append(image_rel_path)
                    
                    # Check for image_filename field
                    if 'image_filename' in data and data['image_filename']:
                        image_candidates.append(data['image_filename'])
                    
                    # Check for other common fields
                    if 'image' in data and data['image']:
                        image_candidates.append(data['image'])
                    
                    if 'image_name' in data and data['image_name']:
                        image_candidates.append(data['image_name'])
                    
                    # Fallback to generated name
                    image_candidates.append(f"chart_{idx}.png")
                    image_candidates.append(f"train_{idx}.png")
                    image_candidates.append(f"test_{idx}.png")
                    
                    # Try to find existing image file
                    for candidate in image_candidates:
                        potential_path = images_dir / candidate
                        if potential_path.exists():
                            image_path = potential_path
                            break
            
            return DatasetSample(
                id=f"{dataset_info.id}_sample_{idx}",
                question=question,
                answer=answer,
                image_path=image_path,
                metadata=data
            )
            
        except Exception as e:
            print(f"Error creating sample {idx}: {e}")
            return None
    
    def get_available_dataset_ids(self) -> List[str]:
        """Get list of available dataset IDs."""
        datasets = self.scan_datasets()
        return [dataset.id for dataset in datasets if dataset.sample_count > 0]
    
    def get_dataset_info_dict(self) -> Dict[str, Dict]:
        """Get dataset information as a dictionary."""
        datasets = self.scan_datasets()
        info_dict = {}
        
        for dataset in datasets:
            info_dict[dataset.id] = {
                'name': dataset.name,
                'path': str(dataset.path),
                'sample_count': dataset.sample_count,
                'has_images': dataset.has_images,
                'has_questions': dataset.has_questions,
                'has_answers': dataset.has_answers,
                'size_mb': round(dataset.size_mb, 1),
                'format': dataset.format_type,
                'status': 'ready' if dataset.sample_count > 0 else 'empty'
            }
        
        return info_dict


def scan_local_datasets() -> List[str]:
    """
    Quick function to scan and return available dataset IDs.
    
    Returns:
        List of dataset IDs that are ready to use
    """
    scanner = DatasetScanner()
    return scanner.get_available_dataset_ids()


def get_dataset_details() -> Dict[str, Dict]:
    """
    Quick function to get detailed dataset information.
    
    Returns:
        Dictionary with dataset details
    """
    scanner = DatasetScanner()
    return scanner.get_dataset_info_dict()


def get_samples_for_dataset(dataset_id: str, limit: int = 250) -> List[DatasetSample]:
    """Quick function to get samples for a dataset."""
    scanner = DatasetScanner()
    return scanner.get_dataset_samples(dataset_id, limit=limit)


if __name__ == "__main__":
    # Test the scanner
    scanner = DatasetScanner()
    datasets = scanner.scan_datasets()
    
    print("Available Datasets:")
    for dataset in datasets:
        print(f"- {dataset.name} ({dataset.id})")
        print(f"  Samples: {dataset.sample_count}")
        print(f"  Has Images: {dataset.has_images}")
        print(f"  Has Questions: {dataset.has_questions}")
        print(f"  Size: {dataset.size_mb:.1f} MB")
        print()


