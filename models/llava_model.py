"""
LLaVA 1.5 7B model implementation for chart QA with attention extraction.
Based on the Hugging Face transformers implementation with proper error handling.
"""
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from rich.console import Console
import json

from .base_model import BaseChartQAModel, ModelPrediction, AttentionOutput
from config import ModelConfig, APP_CONFIG

console = Console()


class LLaVAModel(BaseChartQAModel):
    """LLaVA 1.5 7B model implementation with attention extraction capabilities."""
    
    def __init__(self, model_config: ModelConfig, device: Optional[str] = None):
        """Initialize LLaVA model."""
        super().__init__(model_config, device)
        self.model_name = "llava_v1_5_7b"
    
    def _fix_tokenizer_config(self) -> None:
        """Fix empty or corrupted tokenizer config file."""
        tokenizer_config_path = self.config.local_dir / "tokenizer_config.json"
        
        # Check if tokenizer_config.json exists and is not empty
        if not tokenizer_config_path.exists() or tokenizer_config_path.stat().st_size == 0:
            console.print("[yellow]Fixing empty tokenizer_config.json...[/yellow]")
            
            # Create a basic tokenizer config based on LLaVA requirements
            tokenizer_config = {
                "add_bos_token": False,
                "add_eos_token": False,
                "added_tokens_decoder": {
                    "32000": {
                        "content": "<image>",
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                        "special": True
                    },
                    "32001": {
                        "content": "<pad>",
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                        "special": True
                    }
                },
                "bos_token": "<s>",
                "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
                "clean_up_tokenization_spaces": False,
                "eos_token": "</s>",
                "legacy": True,
                "model_max_length": 2048,
                "pad_token": "<pad>",
                "sp_model_kwargs": {},
                "spaces_between_special_tokens": False,
                "tokenizer_class": "LlamaTokenizer",
                "unk_token": "<unk>",
                "use_default_system_prompt": False
            }
            
            try:
                with open(tokenizer_config_path, 'w') as f:
                    json.dump(tokenizer_config, f, indent=2)
                console.print("[green]✓ Fixed tokenizer_config.json[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not fix tokenizer config: {e}[/yellow]")
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available."""
        deps = {}
        
        # Check protobuf
        try:
            import google.protobuf
            deps['protobuf'] = True
        except ImportError:
            deps['protobuf'] = False
            
        # Check transformers version
        try:
            import transformers
            version = transformers.__version__
            # LLaVA requires transformers >= 4.37.0
            from packaging import version as version_parser
            deps['transformers_version'] = version_parser.parse(version) >= version_parser.parse("4.37.0")
        except ImportError:
            deps['transformers_version'] = False
            
        # Check LlavaForConditionalGeneration availability
        try:
            from transformers import LlavaForConditionalGeneration
            deps['llava_model'] = True
        except ImportError:
            deps['llava_model'] = False
            
        return deps
    
    def load_model(self) -> None:
        """Load LLaVA model, processor, and tokenizer with comprehensive error handling."""
        try:
            console.print(f"Loading LLaVA 1.5 7B from {self.config.local_dir}")
            
            # Check dependencies first
            deps = self._check_dependencies()
            missing_deps = [dep for dep, available in deps.items() if not available]
            
            if missing_deps:
                console.print(f"[yellow]Warning: Missing dependencies: {missing_deps}[/yellow]")
                if 'protobuf' in missing_deps:
                    console.print("[yellow]Install protobuf with: pip install protobuf[/yellow]")
                if 'transformers_version' in missing_deps:
                    console.print("[yellow]Upgrade transformers with: pip install --upgrade transformers[/yellow]")
                if 'llava_model' in missing_deps:
                    console.print("[yellow]LlavaForConditionalGeneration not available in this transformers version[/yellow]")
            
            # Fix tokenizer config if needed
            self._fix_tokenizer_config()
            
            # Import LLaVA model class
            try:
                from transformers import LlavaForConditionalGeneration
                console.print("✓ LlavaForConditionalGeneration imported successfully")
            except ImportError as e:
                console.print(f"[red]Cannot import LlavaForConditionalGeneration: {e}[/red]")
                console.print("[red]Please upgrade transformers: pip install --upgrade transformers>=4.37.0[/red]")
                raise ImportError("LlavaForConditionalGeneration not available")
            
            # Load processor - check dependencies first
            if not deps.get('protobuf', False):
                raise RuntimeError("Protobuf dependency missing. Install with: pip install protobuf")
            
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                str(self.config.local_dir),
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False  # Use slow processor to avoid protobuf issues
            )
            console.print("✓ Processor loaded from local directory")
            
            # Load model with appropriate dtype and settings
            model_kwargs = {
                "local_files_only": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                # Force eager attention to enable attention extraction
                "attn_implementation": "eager"
            }
            
            # Add device mapping for CUDA with proper configuration
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if APP_CONFIG.allow_multi_gpu and gpu_count > 1:
                    model_kwargs["device_map"] = "auto"
                    console.print(f"✓ Multi-GPU enabled: Using {gpu_count} GPUs with auto device mapping")
                else:
                    target_gpu = APP_CONFIG.force_gpu_id if APP_CONFIG.force_gpu_id is not None else 0
                    if target_gpu >= gpu_count:
                        raise RuntimeError(f"Requested GPU {target_gpu} not available. Only {gpu_count} GPUs detected.")
                    model_kwargs["device_map"] = {"": f"cuda:{target_gpu}"}
                    console.print(f"✓ Single GPU mode: Using cuda:{target_gpu}")
            else:
                raise RuntimeError("CUDA not available. LLaVA requires GPU.")
            
            model_loaded = False
            
            # Load model locally only
            model_kwargs["local_files_only"] = True
            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(self.config.local_dir),
                **model_kwargs
            )
            console.print("✓ Model loaded from local directory")
            
            # Move to device if not using device_map
            if self.device != "cuda" or not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Set tokenizer from processor
            if not hasattr(self.processor, 'tokenizer'):
                raise RuntimeError("Processor does not have tokenizer attribute")
            self.tokenizer = self.processor.tokenizer
            console.print("✓ Tokenizer loaded from processor")
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                console.print("✓ Pad token set")
            
            self.is_loaded = True
            console.print(f"✅ LLaVA 1.5 7B loaded successfully on {self.device}")
            
        except Exception as e:
            console.print(f"[red]Failed to load LLaVA 1.5 7B: {e}[/red]")
            # Provide helpful error messages
            if "protobuf" in str(e).lower():
                console.print("[red]💡 Solution: Install protobuf with 'pip install protobuf'[/red]")
            elif "transformers" in str(e).lower():
                console.print("[red]💡 Solution: Upgrade transformers with 'pip install --upgrade transformers>=4.37.0'[/red]")
            elif "llava" in str(e).lower():
                console.print("[red]💡 Solution: Make sure you have the latest transformers version that supports LLaVA[/red]")
            
            raise RuntimeError(f"LLaVA 1.5 7B loading failed: {e}")
    
    def _get_model_device(self) -> torch.device:
        """Get the device where the model is located."""
        try:
            # Method 1: Check if model has a device attribute
            if hasattr(self.model, 'device'):
                return self.model.device
            
            # Method 2: Get device from first parameter
            try:
                return next(self.model.parameters()).device
            except Exception:
                pass
            
            # Method 3: Check for device map (sharded models)
            if hasattr(self.model, 'hf_device_map'):
                # For sharded models, use the device of the first layer
                device_map = self.model.hf_device_map
                if device_map:
                    first_device = list(device_map.values())[0]
                    if isinstance(first_device, int):
                        return torch.device(f'cuda:{first_device}')
                    else:
                        return torch.device(first_device)
            
            # Fallback to self.device
            return torch.device(self.device)
            
        except Exception as e:
            console.print(f"[red]Device detection failed: {e}[/red]")
            raise RuntimeError(f"Could not determine model device: {e}")

    def _move_inputs_to_device(self, inputs: Dict[str, torch.Tensor], context: str = "") -> Dict[str, torch.Tensor]:
        """Move all tensor inputs to the model's device."""
        try:
            # Simple device detection - use self.device
            target_device = torch.device(self.device)
            
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            return inputs
            
        except Exception as device_error:
            console.print(f"[yellow]Warning: {context}device placement failed: {device_error}[/yellow]")
            return inputs

    def _format_llava_prompt(self, question: str) -> str:
        """Format prompt for LLaVA using the proper template."""
        # LLaVA uses a specific prompt format for chat-style interaction
        # We need to format it properly for chart QA
        formatted_prompt = f"USER: {question}\nASSISTANT:"
        return formatted_prompt
    
    def predict(self, image: Union[Image.Image, str], question: str) -> ModelPrediction:
        """
        Make a prediction using LLaVA 1.5 7B.
        
        Args:
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            ModelPrediction with answer and confidence
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Validate and preprocess inputs
        image, question = self.validate_inputs(image, question)
        
        try:
            # HYBRID APPROACH: Use official conversation format but process separately
            # The apply_chat_template has issues with image processing in current transformers
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"This is a chart or graph. {question}"}
                    ]
                }
            ]
            
            # Generate the text prompt using chat template
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Process image and text separately (more reliable)
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Verify that we have the necessary inputs
            if "input_ids" not in inputs:
                raise RuntimeError("Processor failed to generate input_ids")
            if "pixel_values" not in inputs:
                raise RuntimeError("Processor failed to generate pixel_values")
                
            console.print(f"✓ Processed inputs. Input shape: {inputs['input_ids'].shape}, Image shape: {inputs['pixel_values'].shape}")
            
            # Move inputs to device - handle both sharded and non-sharded models
            inputs = self._move_inputs_to_device(inputs, "main ")
            
            # Generate answer with optimized parameters for accuracy
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Sufficient for chart QA answers
                    min_new_tokens=1,   # Ensure some output
                    do_sample=False,    # Deterministic for accuracy
                    num_beams=1,        # Greedy decoding for best accuracy
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode the generated answer
            generated_ids = outputs.sequences
            
            # Remove the input prompt from the generated sequence
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            # Decode answer
            answer = self.tokenizer.decode(
                new_tokens[0], 
                skip_special_tokens=True
            ).strip()
            
            # Use raw answer without cleaning
            
            # Validate answer quality
            if not answer or len(answer.strip()) == 0:
                raise RuntimeError("Model generated empty answer. Check model configuration and input processing.")
            
            # Calculate confidence from generation scores
            confidence = self._calculate_confidence(outputs.scores) if outputs.scores else 0.8
            
            processing_time = time.time() - start_time
            
            return ModelPrediction(
                answer=answer,
                confidence=confidence,
                logits=outputs.scores[-1] if outputs.scores else None,
                processing_time=processing_time
            )
            
        except Exception as e:
            console.print(f"[red]LLaVA prediction failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return ModelPrediction(
                answer=f"Error during prediction: {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    
    def _calculate_confidence(self, scores: Tuple[torch.Tensor, ...]) -> float:
        """Calculate confidence score from generation scores."""
        try:
            if not scores:
                return 0.8
            
            confidences = []
            for step_scores in scores:
                probs = torch.softmax(step_scores, dim=-1)
                max_prob = torch.max(probs, dim=-1)[0]
                confidences.append(max_prob.item())
            
            # Average confidence across generation steps
            avg_confidence = sum(confidences) / len(confidences)
            return float(avg_confidence)
            
        except Exception:
            return 0.8
    
    def extract_attention(self, image: Union[Image.Image, str], question: str) -> AttentionOutput:
        """Extract attention weights from LLaVA for explainability."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate and preprocess inputs
        image, question = self.validate_inputs(image, question)
        
        try:
            # Use same HYBRID format as prediction for consistency
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"This is a chart or graph. {question}"}
                    ]
                }
            ]
            
            # Generate the text prompt using chat template
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Process image and text separately (more reliable)
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device using helper method
            inputs = self._move_inputs_to_device(inputs, "attention ")
            
            # Forward pass with attention extraction
            with torch.no_grad():
                console.print(f"[blue]🔍 Calling model with output_attentions=True[/blue]")
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )
                console.print(f"[blue]🔍 Model outputs keys: {list(outputs.keys())}[/blue]")
                console.print(f"[blue]🔍 Attentions type: {type(outputs.attentions) if hasattr(outputs, 'attentions') else 'None'}[/blue]")
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    console.print(f"[blue]🔍 Number of attention layers: {len(outputs.attentions)}[/blue]")
                else:
                    console.print(f"[yellow]⚠️ No attentions in model outputs![/yellow]")
            
            # Process attention outputs
            processed_attention = self._process_llava_attention_outputs(
                outputs.attentions, inputs, image
            )
            
            return AttentionOutput(
                cross_attention=processed_attention.get("cross_attention"),
                text_self_attention=processed_attention.get("text_self_attention"),
                image_self_attention=processed_attention.get("image_self_attention"),
                text_tokens=processed_attention.get("text_tokens"),
                image_patch_coords=processed_attention.get("image_patch_coords"),
                predicted_answer="",  # We're not generating here
                confidence_score=0.0,
                layer_count=processed_attention.get("layer_count"),
                head_count=processed_attention.get("head_count"),
                image_size=image.size,
                patch_size=self.config.patch_size
            )
            
        except Exception as e:
            console.print(f"[red]LLaVA attention extraction failed: {e}[/red]")
            raise RuntimeError(f"Attention extraction failed: {e}")
    
    def _process_llava_attention_outputs(self, attentions: Tuple[torch.Tensor, ...],
                                       inputs: Dict[str, torch.Tensor], 
                                       image: Image.Image) -> Dict[str, Any]:
        """Process raw attention outputs from LLaVA."""
        console.print(f"[blue]🔍 Starting attention processing...[/blue]")
        console.print(f"[blue]Attentions type: {type(attentions)}, length: {len(attentions) if attentions else 'None'}[/blue]")
        
        try:
            processed = {}
            
            # Get text tokens
            text_tokens = []
            if "input_ids" in inputs:
                text_tokens = self.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0]
                )
                console.print(f"[blue]Got {len(text_tokens)} text tokens[/blue]")
            
            processed["text_tokens"] = text_tokens
            
            # Process attention layers
            if attentions:
                console.print(f"[blue]Processing {len(attentions)} attention layers[/blue]")
                # LLaVA has multiple attention layers
                # We'll focus on the last few layers for cross-modal attention
                last_attention = attentions[-1]  # Shape: [batch, heads, seq_len, seq_len]
                console.print(f"[blue]Last attention shape: {last_attention.shape}[/blue]")
                
                processed["layer_count"] = len(attentions)
                processed["head_count"] = last_attention.shape[1]
                
                # Average across attention heads for visualization
                avg_attention = last_attention.mean(dim=1)  # [batch, seq_len, seq_len]
                console.print(f"[blue]Averaged attention shape: {avg_attention.shape}[/blue]")
                
                # For LLaVA, the input sequence contains both image patches and text tokens
                # We need to identify which tokens correspond to image patches
                
                # Get the actual sequence length
                seq_len = avg_attention.shape[-1]
                
                # LLaVA typically places image tokens at the beginning of the sequence
                # The number of image patches can be estimated from the image size and patch size
                try:
                    # FIXED: More robust patch count calculation for LLaVA
                    console.print(f"[blue]🔧 Using improved patch calculation logic...[/blue]")
                    
                    # Get patch size from model config or use LLaVA default
                    patch_size = getattr(self.config, 'patch_size', 14)
                    if isinstance(patch_size, (list, tuple)):
                        patch_size = patch_size[0]
                    
                    # LLaVA standard image processing size
                    vision_size = 336  # Standard LLaVA image size
                    
                    # Calculate patches without CLS token addition (common source of +1 error)
                    patches_per_side = vision_size // patch_size  # 336/14 = 24
                    base_patches = patches_per_side * patches_per_side  # 24*24 = 576
                    
                    console.print(f"[blue]Vision size: {vision_size}, Patch size: {patch_size}[/blue]")
                    console.print(f"[blue]Patches per side: {patches_per_side}, Base patches: {base_patches}[/blue]")
                    
                    # Try different patch counts to find the correct one
                    possible_patch_counts = [
                        base_patches,           # 576 (without CLS)
                        base_patches + 1,       # 577 (with CLS) 
                        base_patches - 1,       # 575 (edge case)
                        seq_len // 3,          # Conservative estimate
                        seq_len // 4           # Very conservative
                    ]
                    
                    num_image_patches = None
                    cross_attention = None
                    
                    # Try each possible patch count
                    for trial_patches in possible_patch_counts:
                        if trial_patches <= 0 or trial_patches >= seq_len:
                            continue
                            
                        console.print(f"[blue]Trying {trial_patches} image patches...[/blue]")
                        
                        # Check if this makes sense with sequence structure
                        remaining_tokens = seq_len - trial_patches
                        if remaining_tokens < 5:  # Need at least some text tokens
                            console.print(f"[yellow]Too few text tokens ({remaining_tokens}), skipping[/yellow]")
                            continue
                        
                        # Extract potential cross-attention: text tokens attending to image patches
                        # Image tokens are at positions 4:(4+576), text tokens start after that
                        image_start = 4
                        image_end = image_start + trial_patches
                        text_start = image_end
                        
                        if text_start < seq_len:
                            # Text-to-image attention: text tokens attending to image patches
                            trial_cross = avg_attention[0, text_start:, image_start:image_end]
                        else:
                            trial_cross = torch.empty(0, 0)  # Invalid extraction
                        
                        # Validate this extraction makes sense
                        if trial_cross.numel() > 0:
                            # Check if attention values look realistic
                            cross_mean = trial_cross.mean().item()
                            cross_std = trial_cross.std().item()
                            cross_max = trial_cross.max().item()
                            
                            console.print(f"[blue]Trial cross-attention stats: mean={cross_mean:.4f}, std={cross_std:.4f}, max={cross_max:.4f}[/blue]")
                            
                            # Good cross-attention should have:
                            # 1. Reasonable variance (not all same values)
                            # 2. Meaningful range of values
                            # 3. Not be mostly zeros
                            non_zero_ratio = (trial_cross > 1e-6).float().mean().item()
                            
                            console.print(f"[blue]Non-zero ratio: {non_zero_ratio:.3f}[/blue]")
                            
                            if cross_std > 1e-4 and non_zero_ratio > 0.1 and cross_max > 0.01:
                                console.print(f"[green]✓ Found good cross-attention with {trial_patches} patches![/green]")
                                num_image_patches = trial_patches
                                cross_attention = trial_cross
                                break
                            else:
                                console.print(f"[yellow]Cross-attention quality check failed for {trial_patches} patches[/yellow]")
                    
                    # If we found good cross-attention, use it
                    if cross_attention is not None and num_image_patches is not None:
                        processed["cross_attention"] = cross_attention
                        console.print(f"[green]✓ Final cross-attention shape: {cross_attention.shape}[/green]")
                        
                        # Add debug info to help diagnose issues
                        processed["debug_info"] = {
                            "num_image_patches": num_image_patches,
                            "seq_len": seq_len,
                            "patches_per_side": patches_per_side,
                            "attention_mean": cross_attention.mean().item(),
                            "attention_std": cross_attention.std().item(),
                            "attention_max": cross_attention.max().item()
                        }
                    else:
                        raise RuntimeError("Could not find valid cross-attention in model outputs")
                    
                    # Store self-attention for text tokens
                    if seq_len > num_image_patches:
                        text_self_attention = avg_attention[0, num_image_patches:, num_image_patches:]
                        if text_self_attention.numel() > 0:
                            processed["text_self_attention"] = text_self_attention
                    
                    # Store self-attention for image tokens
                    if num_image_patches > 0:
                        image_self_attention = avg_attention[0, :num_image_patches, :num_image_patches]
                        if image_self_attention.numel() > 0:
                            processed["image_self_attention"] = image_self_attention
                            
                except Exception as patch_error:
                    console.print(f"[red]Cross-attention extraction failed: {patch_error}[/red]")
                    raise RuntimeError(f"Cross-attention extraction failed: {patch_error}")
                
            # Get image patch coordinates
            patch_coords = self.get_image_patches(image)
            processed["image_patch_coords"] = patch_coords
            
            return processed
            
        except Exception as e:
            console.print(f"[yellow]Warning: LLaVA attention processing failed: {e}[/yellow]")
            return {
                "text_tokens": [],
                "image_patch_coords": [],
                "layer_count": 0,
                "head_count": 0,
                "cross_attention": None
            }
    
    def get_image_patches(self, image: Image.Image) -> List[Tuple[int, int]]:
        """
        Get the coordinates of image patches for LLaVA attention visualization.
        FIXED: Ensures proper coordinate mapping that matches attention tensor ordering.
        
        Args:
            image: Input image
            
        Returns:
            List of (x, y) coordinates for each patch in row-major order
        """
        try:
            # Use LLaVA-specific patch calculation
            patch_size = getattr(self.config, 'patch_size', 14)
            
            # Handle patch_size being a tuple like (14, 14)
            if isinstance(patch_size, (list, tuple)):
                patch_size = patch_size[0]  # Use the first dimension
            
            # LLaVA standard vision size
            vision_size = 336  # Fixed to match attention extraction
            patches_per_side = vision_size // patch_size  # 24 for 336/14
            
            console.print(f"[blue]🔧 FIXED: Generating patches with vision_size={vision_size}, patch_size={patch_size}[/blue]")
            console.print(f"[blue]Patches per side: {patches_per_side}[/blue]")
            
            # Generate patch coordinates in ROW-MAJOR order (critical for proper mapping)
            # This must match the order that attention weights are organized in
            patches = []
            img_w, img_h = image.size
            
            for y in range(patches_per_side):
                for x in range(patches_per_side):
                    # FIXED: Map from patch grid coordinates to actual image coordinates
                    # Each patch covers (img_w / patches_per_side) pixels in width
                    # and (img_h / patches_per_side) pixels in height
                    patch_width = img_w / patches_per_side
                    patch_height = img_h / patches_per_side
                    
                    # Use center of each patch for visualization
                    actual_x = int((x + 0.5) * patch_width)
                    actual_y = int((y + 0.5) * patch_height)
                    patches.append((actual_x, actual_y))
            
            expected_count = patches_per_side * patches_per_side  # Should be 576
            console.print(f"[green]✓ Generated {len(patches)} patch coordinates (expected: {expected_count})[/green]")
            console.print(f"[blue]First 5 patches: {patches[:5]}[/blue]")
            console.print(f"[blue]Last 5 patches: {patches[-5:]}[/blue]")
            
            # Validate coordinate ordering
            if len(patches) >= 2:
                # Check that we're going left-to-right first (row-major)
                if patches[1][0] > patches[0][0]:  # Second patch should be to the right
                    console.print(f"[green]✓ Coordinate ordering verified: row-major (left-to-right, top-to-bottom)[/green]")
                else:
                    console.print(f"[yellow]⚠️  Coordinate ordering may be incorrect[/yellow]")
            
            return patches
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate patch coordinates: {e}[/yellow]")
            raise RuntimeError(f"Could not generate patch coordinates: {e}")

    def _is_attention_module(self, name: str, module: torch.nn.Module) -> bool:
        """
        Determine if a module is an attention module for LLaVA.
        
        Args:
            name: Module name
            module: PyTorch module
            
        Returns:
            True if this module should be hooked for attention extraction
        """
        attention_patterns = [
            "self_attn",
            "cross_attn",
            "attention",
            "attn",
            "multihead_attn"
        ]
        
        return any(pattern in name.lower() for pattern in attention_patterns)
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported features for this model."""
        return [
            "text_generation",
            "visual_understanding", 
            "chart_analysis",
            "conversation",
            "attention_extraction",
            "multimodal_reasoning",
            "llava_architecture",
            "advanced_statistical_analysis",
            "attention_flow_analysis",
            "layer_wise_analysis",
            "multi_head_analysis",
            "llava_specific_analysis"
        ]
    
    def run_advanced_analysis(self, analysis_type: str, image: Union[Image.Image, str], 
                            question: str, **kwargs) -> Dict[str, Any]:
        """
        Run advanced analysis on the LLaVA model's attention patterns.
        
        Args:
            analysis_type: Type of analysis to run
            image: PIL Image or path to image file
            question: Question text
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Extract attention data first
        attention_output = self.extract_attention(image, question)
        
        # Import analysis manager
        from components.advanced_analysis import AdvancedAnalysisManager
        analysis_manager = AdvancedAnalysisManager()
        
        # Process image for analysis
        processed_image = self.preprocess_image(image)
        
        # Run the requested analysis with LLaVA-specific enhancements
        analysis_result = analysis_manager.run_analysis(
            analysis_type=analysis_type,
            attention_output=attention_output,
            image=processed_image,
            question=question,
            model_type="llava",  # LLaVA-specific parameter
            **kwargs
        )
        
        # Add LLaVA-specific insights
        if analysis_result.insights:
            llava_insights = self._add_llava_specific_insights(
                analysis_result, attention_output, question
            )
            analysis_result.insights.extend(llava_insights)
        
        return {
            'analysis_type': analysis_result.analysis_type,
            'title': analysis_result.title,
            'description': analysis_result.description,
            'metrics': analysis_result.metrics,
            'insights': analysis_result.insights,
            'visualization_data': analysis_result.visualization.to_dict() if analysis_result.visualization else None,
            'raw_data': analysis_result.raw_data,
            'model_specific': 'llava_1_5_7b'
        }
    
    def _add_llava_specific_insights(self, analysis_result: Any, attention_output: AttentionOutput, 
                                   question: str) -> List[str]:
        """Add LLaVA-specific insights to analysis results."""
        insights = []
        
        # Analyze question type for LLaVA
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['describe', 'what do you see', 'explain']):
            insights.append("LLaVA: Descriptive question - model excels at detailed visual descriptions")
        elif any(word in question_lower for word in ['count', 'how many', 'number of']):
            insights.append("LLaVA: Counting question - model uses visual object detection capabilities")
        elif any(word in question_lower for word in ['color', 'red', 'blue', 'green', 'yellow']):
            insights.append("LLaVA: Color-related question - model analyzes visual appearance")
        elif any(word in question_lower for word in ['text', 'written', 'says', 'label']):
            insights.append("LLaVA: Text reading question - model uses OCR capabilities")
        
        # Analyze attention distribution for LLaVA understanding
        if attention_output.cross_attention is not None:
            cross_attn = attention_output.cross_attention
            if hasattr(cross_attn, 'detach'):
                cross_attn = cross_attn.detach().cpu().numpy()
            else:
                cross_attn = np.array(cross_attn)
            
            # LLaVA-specific attention analysis
            flat_attn = cross_attn.flatten()
            attention_peaks = np.where(flat_attn > np.percentile(flat_attn, 95))[0]
            
            if len(attention_peaks) < 10:
                insights.append("LLaVA: Highly focused attention - model concentrates on specific visual elements")
            elif len(attention_peaks) > 50:
                insights.append("LLaVA: Distributed attention - model considers many visual elements")
            
            # Check attention variance for LLaVA's visual processing
            attention_variance = np.var(flat_attn)
            if attention_variance > 0.01:
                insights.append("LLaVA: High attention variance indicates complex visual reasoning")
            else:
                insights.append("LLaVA: Low attention variance suggests straightforward visual understanding")
        
        return insights
    
    def get_attention_statistics(self, image: Union[Image.Image, str], question: str) -> Dict[str, float]:
        """
        Get statistical metrics about attention patterns for LLaVA model.
        
        Args:
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            Dictionary of statistical metrics
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Extract attention data
        attention_output = self.extract_attention(image, question)
        
        if attention_output.cross_attention is None:
            return {}
        
        # Calculate basic statistics
        cross_attn = attention_output.cross_attention
        if hasattr(cross_attn, 'detach'):
            cross_attn = cross_attn.detach().cpu().numpy()
        else:
            cross_attn = np.array(cross_attn)
        
        while len(cross_attn.shape) > 2:
            cross_attn = cross_attn.mean(axis=0)
        
        flat_attn = cross_attn.flatten()
        
        # Calculate entropy
        normalized_weights = flat_attn / np.sum(flat_attn)
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-12))
        
        # Calculate Gini coefficient
        sorted_weights = np.sort(flat_attn)
        n = len(sorted_weights)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        
        # Calculate sparsity
        threshold = np.max(flat_attn) * 0.01
        sparsity = np.sum(flat_attn < threshold) / len(flat_attn)
        
        return {
            'attention_mean': float(np.mean(flat_attn)),
            'attention_std': float(np.std(flat_attn)),
            'attention_max': float(np.max(flat_attn)),
            'attention_min': float(np.min(flat_attn)),
            'attention_entropy': float(entropy),
            'attention_gini': float(gini),
            'attention_sparsity': float(sparsity),
            'total_attention': float(np.sum(flat_attn)),
            'model_type': 'llava_1_5_7b'
        }
    
    def analyze_token_importance(self, image: Union[Image.Image, str], question: str) -> Dict[str, Any]:
        """
        Analyze the importance of different tokens in the question for LLaVA model.
        
        Args:
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            Dictionary containing token importance analysis
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Extract attention data
        attention_output = self.extract_attention(image, question)
        
        if attention_output.cross_attention is None or not attention_output.text_tokens:
            return {'tokens': [], 'importance_scores': [], 'insights': []}
        
        # Calculate token importance
        cross_attn = attention_output.cross_attention
        if hasattr(cross_attn, 'detach'):
            cross_attn = cross_attn.detach().cpu().numpy()
        else:
            cross_attn = np.array(cross_attn)
        
        while len(cross_attn.shape) > 2:
            cross_attn = cross_attn.mean(axis=0)
        
        # Sum attention across image patches for each token
        token_importance = np.sum(cross_attn, axis=1)
        token_importance = token_importance / np.sum(token_importance)  # Normalize
        
        # Create token analysis
        tokens = attention_output.text_tokens[:len(token_importance)]
        importance_scores = token_importance[:len(tokens)].tolist()
        
        # Generate insights
        insights = []
        if len(tokens) > 0:
            max_idx = np.argmax(token_importance[:len(tokens)])
            min_idx = np.argmin(token_importance[:len(tokens)])
            
            insights.append(f"Most important token: '{tokens[max_idx]}' (score: {importance_scores[max_idx]:.3f})")
            insights.append(f"Least important token: '{tokens[min_idx]}' (score: {importance_scores[min_idx]:.3f})")
            
            # LLaVA-specific analysis
            visual_words = ['image', 'picture', 'chart', 'graph', 'see', 'show', 'visible']
            visual_token_indices = [i for i, token in enumerate(tokens) 
                                  if any(vw in token.lower() for vw in visual_words)]
            
            if visual_token_indices:
                avg_visual_importance = np.mean([importance_scores[i] for i in visual_token_indices])
                insights.append(f"LLaVA: Visual-related tokens have average importance of {avg_visual_importance:.3f}")
        
        return {
            'tokens': tokens,
            'importance_scores': importance_scores,
            'insights': insights,
            'total_tokens': len(tokens),
            'attention_distribution': 'concentrated' if np.std(importance_scores) > 0.1 else 'distributed',
            'model_type': 'llava_1_5_7b'
        }
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
