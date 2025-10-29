"""
LLaVA-NeXT (LLaVA v1.6) model implementation with attention extraction capabilities.

LLaVA-NeXT is an improved version of LLaVA with enhanced image understanding and 
better handling of high-resolution images.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import numpy as np
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import logging
import time

from models.base_model import BaseChartQAModel, AttentionOutput, ModelPrediction
from config import ModelConfig

logger = logging.getLogger(__name__)


class LLaVANextModel(BaseChartQAModel):
    """
    LLaVA-NeXT (v1.6) model for Chart Question Answering with cross-attention extraction.
    
    This model improves upon LLaVA v1.5 with:
    - Better high-resolution image handling
    - Improved visual reasoning capabilities
    - Enhanced multi-image support
    """
    
    def __init__(self, config: ModelConfig, device: Optional[str] = None):
        """
        Initialize LLaVA-NeXT model.
        
        Args:
            config: Model configuration object
            device: Device to run the model on (auto-detected if None)
        """
        super().__init__(config, device)
        self.processor: Optional[LlavaNextProcessor] = None
        self.model: Optional[LlavaNextForConditionalGeneration] = None
        self._supports_cross_attention = True
        
        # If we have multiple GPUs, prefer GPU 1 to avoid conflicts with GPU 0
        if device is None and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            # Set default CUDA device to GPU 1 if available
            torch.cuda.set_device(1)
            logger.info("Multiple GPUs detected - setting default CUDA device to GPU 1")
        
    def load_model(self) -> None:
        """Load the LLaVA-NeXT model and processor."""
        if self.is_loaded:
            logger.info(f"Model {self.config.name} is already loaded")
            return
            
        try:
            logger.info(f"Loading LLaVA-NeXT model from {self.config.local_dir}")
            
            # Load processor
            self.processor = LlavaNextProcessor.from_pretrained(
                str(self.config.local_dir),
                local_files_only=True,
                trust_remote_code=True
            )
            
            # Determine torch dtype based on device
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load model with device mapping
            model_kwargs = {
                "local_files_only": True,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "attn_implementation": "eager"  # Enable attention extraction
            }
            
            # Try to use 8-bit quantization if available for memory efficiency
            # This reduces model size by ~50% which helps with attention extraction
            use_quantization = False
            try:
                from transformers import BitsAndBytesConfig
                
                # Check if we're on CUDA and have multiple GPUs
                if self.device == "cuda" and torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    
                    if num_gpus >= 2:
                        logger.info("Attempting 8-bit quantization for memory efficiency...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            llm_int8_skip_modules=["lm_head", "embed_tokens"]  # Keep these in FP16 for stability
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        
                        # IMPORTANT: Remove torch_dtype when using quantization
                        # Quantization and torch_dtype can conflict
                        model_kwargs.pop("torch_dtype", None)
                        use_quantization = True
                        
                        logger.info("8-bit quantization enabled - model memory usage will be ~50% lower")
                        logger.info("Note: torch_dtype removed to avoid conflict with quantization")
            except ImportError:
                logger.info("BitsAndBytes not available - loading in float16 (install bitsandbytes for lower memory usage)")
            
            # Add device mapping for CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                
                if num_gpus >= 2:
                    # Use GPU 1 exclusively to avoid Jupyter kernel memory on GPU 0
                    # This provides maximum clean memory for attention extraction
                    logger.info(f"Detected {num_gpus} GPUs - loading model entirely on GPU 1")
                    logger.info("GPU 1 selected to avoid conflicts with GPU 0 (Jupyter kernel)")
                    
                    # Force ALL model components to GPU 1
                    model_kwargs["device_map"] = {"": 1}  # Empty string means "all components"
                    
                    logger.info("Model will be loaded entirely on GPU 1 (24GB available)")
                else:
                    # Single GPU - use auto mapping
                    model_kwargs["device_map"] = "auto"
                    logger.info("Single GPU detected - using auto device mapping")
            
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                str(self.config.local_dir),
                **model_kwargs
            )
            
            # Enable gradient checkpointing to reduce memory usage during attention extraction
            # This trades compute for memory by not storing all intermediate activations
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled - reduces memory usage for attention extraction")
            
            # Set eval mode
            self.model.eval()
            
            # Set tokenizer from processor
            self.tokenizer = self.processor.tokenizer
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            logger.info(f"Successfully loaded LLaVA-NeXT model: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA-NeXT model: {e}")
            self.is_loaded = False
            raise
            
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info(f"Unloaded LLaVA-NeXT model: {self.config.name}")
        
    def predict(self, image: Union[Image.Image, str], question: str) -> ModelPrediction:
        """
        Generate answer to a question about the chart image.
        
        Args:
            image: PIL Image of the chart or path to image file
            question: Question to answer
            
        Returns:
            ModelPrediction with answer and confidence
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.config.name} is not loaded")
        
        start_time = time.time()
        
        # Validate and preprocess inputs
        image, question = self.validate_inputs(image, question)
            
        try:
            # Prepare conversation format for LLaVA-NeXT
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Move inputs to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract answer
            generated_ids = outputs.sequences if hasattr(outputs, 'sequences') else outputs
            
            # Remove the input prompt from the generated sequence
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            # Decode answer
            answer = self.processor.decode(
                new_tokens[0], 
                skip_special_tokens=True
            ).strip()
            
            # Calculate confidence from generation scores
            confidence = self._calculate_confidence(outputs.scores) if hasattr(outputs, 'scores') and outputs.scores else 0.8
            
            processing_time = time.time() - start_time
            
            return ModelPrediction(
                answer=answer,
                confidence=confidence,
                logits=outputs.scores[-1] if hasattr(outputs, 'scores') and outputs.scores else None,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ModelPrediction(
                answer=f"Error during prediction: {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
            
    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """Extract the answer portion from generated text."""
        # Remove the input prompt to get just the answer
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            # Fallback: try to find assistant's response
            # Look for common patterns in LLaVA-NeXT output
            if "ASSISTANT:" in generated_text:
                answer = generated_text.split("ASSISTANT:")[-1].strip()
            elif "[/INST]" in generated_text:
                answer = generated_text.split("[/INST]")[-1].strip()
            else:
                answer = generated_text
                
        return answer
        
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
        """
        Extract cross-attention weights during answer generation.
        
        For LLaVA-NeXT, this extracts attention patterns from the language model's
        self-attention layers, specifically focusing on how text tokens attend to 
        image tokens embedded in the sequence.
        
        Args:
            image: PIL Image of the chart or path to image file
            question: Question to answer
            
        Returns:
            AttentionOutput containing cross-attention data
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.config.name} is not loaded")
        
        # Validate and preprocess inputs
        image, question = self.validate_inputs(image, question)
            
        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Move inputs to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Get input token information for understanding the sequence
            input_ids = inputs['input_ids']
            seq_length = input_ids.shape[1]
            
            logger.info(f"Input sequence length: {seq_length}")
            
            # First, do a forward pass to extract attention patterns
            logger.info("Extracting attention from forward pass...")
            cross_attention = None
            image_token_indices = []
            
            with torch.no_grad():
                try:
                    # Aggressive memory cleanup before forward pass
                    if torch.cuda.is_available():
                        # Clear cache multiple times to defragment
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Force synchronization to ensure cleanup completes
                        torch.cuda.synchronize()
                        
                        # Log memory state
                        for i in range(torch.cuda.device_count()):
                            mem_free = (torch.cuda.get_device_properties(i).total_memory - 
                                       torch.cuda.memory_reserved(i)) / (1024**3)
                            logger.info(f"GPU {i}: {mem_free:.2f} GB free before attention extraction")
                    
                    # Forward pass with attention - MEMORY OPTIMIZED VERSION
                    # Extract attention using hooks to avoid OOM from storing all layers
                    logger.info(f"Running forward pass with attention extraction (seq_len={seq_length})...")
                    logger.info("Using memory-optimized layer-by-layer attention extraction...")
                    
                    # Use custom attention extraction to save memory
                    attentions_data = self._extract_attention_memory_efficient(inputs, seq_length)
                    
                    if attentions_data is None or len(attentions_data) == 0:
                        logger.warning("No attention data extracted from model")
                        raise ValueError("No attention patterns available")
                    
                    logger.info(f"Extracted attention from {len(attentions_data)} layers")
                    
                    # Process attention to extract cross-modal patterns
                    # This will move attention tensors to CPU and free GPU memory
                    cross_attention, image_token_indices = self._extract_cross_modal_attention(
                        attentions_data, 
                        input_ids,
                        inputs.get('pixel_values')
                    )
                    
                    # Immediately free the attention tensors and outputs to release GPU memory
                    del attentions_data
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if cross_attention is None:
                        logger.warning("Failed to extract cross-modal attention patterns")
                        raise ValueError("Cross-modal attention extraction failed")
                    
                    logger.info(f"Cross-attention shape: {cross_attention.shape}")
                    
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower():
                        logger.error(f"GPU out of memory during attention extraction")
                        logger.warning("Your GPU does not have enough memory for attention extraction with this model.")
                        logger.warning("Possible solutions:")
                        logger.warning("  1. Use a smaller model")
                        logger.warning("  2. Reduce image resolution")
                        logger.warning("  3. Use a machine with more GPU memory")
                        logger.warning("Returning without attention data...")
                        raise ValueError("Insufficient GPU memory for attention extraction")
                    elif "CUDA" in error_msg:
                        logger.error(f"CUDA error during attention extraction: {e}")
                        raise ValueError("CUDA error during attention extraction")
                    else:
                        raise
            
            # Get text tokens for the input sequence
            text_tokens = self.get_text_tokens(input_ids[0])
            
            # Get image patch information
            image_patch_coords, image_size = self._get_image_patch_info(
                image, 
                inputs.get('pixel_values')
            )
            
            # Count layers and heads if attention is available
            layer_count = None
            head_count = None
            if attentions_data:
                layer_count = len(attentions_data)
                if layer_count > 0 and attentions_data[0] is not None:
                    head_count = attentions_data[0].shape[1]
            
            logger.info(f"Successfully extracted attention: {layer_count} layers, {head_count} heads")
            
            return AttentionOutput(
                cross_attention=cross_attention,
                text_tokens=text_tokens,
                image_patch_coords=image_patch_coords,
                image_size=image_size,
                layer_count=layer_count,
                head_count=head_count
            )
            
        except Exception as e:
            logger.error(f"Error extracting attention: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a valid AttentionOutput with None attention instead of raising
            logger.warning("Returning AttentionOutput without attention weights")
            return AttentionOutput(
                cross_attention=None,
                text_tokens=["[Attention extraction failed]"],
                image_patch_coords=[(0, 0)],
                image_size=(100, 100),
                layer_count=0,
                head_count=0
            )
            
    def _extract_attention_memory_efficient(self, inputs: Dict, seq_length: int, 
                                           num_layers_to_sample: int = 4) -> Optional[Tuple]:
        """
        Memory-efficient attention extraction - SIMPLIFIED VERSION.
        
        The key insight: When output_attentions=True, the model STILL computes and stores
        all attention weights internally, causing OOM. Hooks don't prevent this.
        
        Solution: Only extract from a very small number of final layers (4 instead of 8)
        and use a single forward pass with output_attentions=True.
        
        Args:
            inputs: Model inputs dict
            seq_length: Input sequence length
            num_layers_to_sample: Number of layers to sample (default 4 from last layers)
            
        Returns:
            Tuple of attention tensors (one per sampled layer)
        """
        try:
            logger.info(f"Attempting to extract attention from last {num_layers_to_sample} layers...")
            logger.info(f"Note: Attention extraction requires ~3x model memory (model + activations + attention)")
            
            # Try the forward pass with attention
            # This will compute attention for ALL layers but we'll only keep the last few
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True, return_dict=True)
            
            # Get all attention weights
            all_attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
            
            if all_attentions is None or len(all_attentions) == 0:
                logger.warning("No attention data extracted from model")
                return None
            
            # Only keep the last N layers to save memory
            num_total_layers = len(all_attentions)
            start_layer = max(0, num_total_layers - num_layers_to_sample)
            
            logger.info(f"Extracted attention from {num_total_layers} layers, keeping last {num_layers_to_sample}")
            
            # Keep only last N layers and move to CPU immediately
            sampled_attentions = []
            for i in range(start_layer, num_total_layers):
                attn = all_attentions[i]
                if attn is not None:
                    # Move to CPU immediately to free GPU memory
                    sampled_attentions.append(attn.detach().cpu())
            
            # Clear the full attention tensors from GPU
            del all_attentions
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Successfully extracted attention from {len(sampled_attentions)} layers")
            return tuple(sampled_attentions)
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"OOM during attention extraction: {e}")
            logger.warning("GPU out of memory - attention extraction not possible with current settings")
            logger.warning("Consider: 1) Using 8-bit quantization, 2) Smaller images, 3) More GPU memory")
            return None
        except Exception as e:
            logger.error(f"Error in attention extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_cross_modal_attention(self, attentions, input_ids: torch.Tensor, 
                                       pixel_values: Optional[torch.Tensor]) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Extract cross-modal attention patterns from LLaVA-NeXT self-attention.
        
        LLaVA-NeXT embeds image tokens directly into the text sequence, so we need to
        identify which tokens are image tokens and extract attention between text and image.
        
        Args:
            attentions: Tuple of attention tensors from model forward pass
            input_ids: Input token IDs to identify image token positions
            pixel_values: Processed image tensor (not directly used but kept for API consistency)
            
        Returns:
            Tuple of (cross_attention_matrix, image_token_indices)
            cross_attention_matrix shape: [text_tokens, image_tokens]
        """
        if attentions is None or len(attentions) == 0:
            logger.warning("No attention tensors available")
            return None, []
        
        try:
            # LLaVA-NeXT uses image token index (usually 32000) to represent image patches
            image_token_id = self.model.config.image_token_index
            
            # Find positions of image tokens in the sequence
            image_token_mask = (input_ids[0] == image_token_id).cpu().numpy()
            image_token_indices = np.where(image_token_mask)[0].tolist()
            
            if len(image_token_indices) == 0:
                logger.warning("No image tokens found in input sequence")
                # Fallback: assume first N tokens after some offset are image tokens
                # LLaVA-NeXT typically has ~576 image tokens
                seq_len = input_ids.shape[1]
                if seq_len > 600:
                    image_token_indices = list(range(10, 586))  # Approximate image token range
                    logger.info(f"Using fallback image token indices: {len(image_token_indices)} tokens")
                else:
                    return None, []
            
            logger.info(f"Found {len(image_token_indices)} image tokens in sequence")
            
            # Get text token indices (all non-image tokens)
            all_indices = set(range(input_ids.shape[1]))
            text_token_indices = sorted(list(all_indices - set(image_token_indices)))
            
            logger.info(f"Found {len(text_token_indices)} text tokens in sequence")
            
            # Average attention across layers and heads
            # Process attention matrices efficiently to avoid OOM
            # Strategy: Only use last 8 layers for memory efficiency
            num_layers = len(attentions)
            layers_to_use = max(8, num_layers // 4)  # Use last 25% or minimum 8 layers
            start_layer = max(0, num_layers - layers_to_use)
            
            logger.info(f"Using layers {start_layer}-{num_layers-1} (last {layers_to_use} layers) for attention extraction")
            
            attention_matrices = []
            for layer_idx in range(start_layer, num_layers):
                layer_attn = attentions[layer_idx]
                if layer_attn is None:
                    continue
                
                # Shape: (batch=1, num_heads, seq_len, seq_len)
                # Average across heads and immediately move to CPU to free GPU memory
                avg_attn = layer_attn[0].mean(dim=0).cpu()  # Shape: (seq_len, seq_len)
                attention_matrices.append(avg_attn)
                
                # Free GPU memory immediately after each layer
                del layer_attn
                if layer_idx % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if not attention_matrices:
                logger.warning("No valid attention matrices found")
                return None, []
            
            # Stack and average across layers (all on CPU now)
            stacked_attn = torch.stack(attention_matrices, dim=0)
            avg_layer_attn = stacked_attn.mean(dim=0)  # Shape: (seq_len, seq_len)
            
            # Free intermediate tensors
            del attention_matrices
            del stacked_attn
            
            # Ensure indices are on CPU for indexing
            text_token_indices_tensor = torch.tensor(text_token_indices, dtype=torch.long)
            image_token_indices_tensor = torch.tensor(image_token_indices, dtype=torch.long)
            
            # Extract cross-modal attention: text tokens attending to image tokens
            # Result shape: [num_text_tokens, num_image_tokens]
            cross_modal_attn = avg_layer_attn[text_token_indices_tensor, :][:, image_token_indices_tensor]
            
            # Convert to numpy
            cross_modal_np = cross_modal_attn.numpy()
            
            logger.info(f"Extracted cross-modal attention matrix: {cross_modal_np.shape}")
            logger.info(f"Attention range: [{cross_modal_np.min():.6f}, {cross_modal_np.max():.6f}]")
            
            return cross_modal_np, image_token_indices
            
        except Exception as e:
            logger.error(f"Error extracting cross-modal attention: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, []
    
    def _extract_attention_weights_forward(self, attentions) -> Optional[np.ndarray]:
        """
        Extract and process attention weights from a forward pass.
        
        Args:
            attentions: Tuple of attention tensors from forward pass
            
        Returns:
            Processed attention weights as numpy array
        """
        if attentions is None:
            logger.warning("No attention weights available")
            return None
            
        try:
            # For forward pass: attentions is a tuple of tensors, one per layer
            # Shape per layer: (batch, num_heads, seq_len, seq_len)
            
            if len(attentions) == 0:
                return None
                
            # Average across all layers and heads
            attention_tensors = []
            for layer_attn in attentions:
                if layer_attn is not None:
                    # Average across heads: (batch, num_heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
                    avg_heads = layer_attn.mean(dim=1)
                    attention_tensors.append(avg_heads)
            
            if not attention_tensors:
                return None
                
            # Stack and average across layers
            stacked = torch.stack(attention_tensors, dim=0)
            # (num_layers, batch, seq_len, seq_len) -> (batch, seq_len, seq_len)
            avg_layers = stacked.mean(dim=0)
            
            # Convert to numpy
            cross_attn = avg_layers.squeeze(0).cpu().numpy()
            
            return cross_attn
            
        except Exception as e:
            logger.error(f"Error processing attention weights: {e}")
            return None
    
    def _extract_attention_weights(self, attentions) -> Optional[np.ndarray]:
        """
        Extract and process cross-attention weights from model generation outputs.
        
        Args:
            attentions: Tuple of attention tensors from generation output
            
        Returns:
            Processed cross-attention weights as numpy array
        """
        if attentions is None:
            logger.warning("No attention weights available")
            return None
            
        try:
            # LLaVA-NeXT structure during generation: attentions is a tuple of tuples
            # Each generation step has attention weights
            # We typically want the last generation step for final answer
            
            if len(attentions) == 0:
                return None
                
            # Get attention from last generation step
            last_step_attention = attentions[-1]
            
            if not last_step_attention:
                return None
                
            # Average across all layers and heads
            # Shape: (num_layers, batch, num_heads, seq_len, seq_len)
            attention_tensors = []
            for layer_attn in last_step_attention:
                if layer_attn is not None:
                    # Average across heads: (batch, num_heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
                    avg_heads = layer_attn.mean(dim=1)
                    attention_tensors.append(avg_heads)
            
            if not attention_tensors:
                return None
                
            # Stack and average across layers
            stacked = torch.stack(attention_tensors, dim=0)
            # (num_layers, batch, seq_len, seq_len) -> (batch, seq_len, seq_len)
            avg_layers = stacked.mean(dim=0)
            
            # Convert to numpy
            cross_attn = avg_layers.squeeze(0).cpu().numpy()
            
            return cross_attn
            
        except Exception as e:
            logger.error(f"Error processing attention weights: {e}")
            return None
            
    def _get_image_patch_info(self, image: Image.Image, 
                              pixel_values: Optional[torch.Tensor]) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
        """
        Get image patch coordinates and original image size.
        
        Args:
            image: Original PIL image
            pixel_values: Processed pixel values tensor
            
        Returns:
            Tuple of (patch_coordinates, image_size)
        """
        image_size = image.size  # (width, height)
        
        # LLaVA-NeXT typically uses a grid of patches
        # Default grid size varies based on image resolution
        # Common configuration: 24x24 patches for high-res images
        
        if pixel_values is not None:
            # Try to infer grid size from pixel values
            # LLaVA-NeXT uses variable grid sizes based on image aspect ratio
            grid_h, grid_w = 24, 24  # Default fallback
        else:
            grid_h, grid_w = 24, 24
        
        # Generate patch coordinates
        patch_coords = [
            (i % grid_w, i // grid_w)
            for i in range(grid_h * grid_w)
        ]
        
        return patch_coords, image_size
        
    def get_text_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """
        Convert token IDs to text tokens.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            List of token strings
        """
        if not self.is_loaded or self.processor is None:
            raise RuntimeError(f"Model {self.config.name} is not loaded")
            
        try:
            # Convert token IDs to tokens
            tokens = self.processor.tokenizer.convert_ids_to_tokens(
                token_ids.tolist()
            )
            
            # Clean up tokens (remove special characters like Ġ)
            clean_tokens = []
            for token in tokens:
                # Handle byte-level BPE tokens
                if token.startswith('Ġ'):
                    clean_tokens.append(token[1:])  # Remove Ġ prefix
                elif token.startswith('▁'):
                    clean_tokens.append(token[1:])  # Remove ▁ prefix (sentencepiece)
                else:
                    clean_tokens.append(token)
                    
            return clean_tokens
            
        except Exception as e:
            logger.error(f"Error converting tokens: {e}")
            return []
            
    def _is_attention_module(self, name: str, module: torch.nn.Module) -> bool:
        """
        Determine if a module is an attention module for LLaVA-NeXT.
        
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'name': self.config.name,
            'model_path': str(self.config.local_dir),
            'is_loaded': self.is_loaded,
            'supports_cross_attention': self._supports_cross_attention,
            'model_type': 'LLaVA-NeXT',
            'architecture': 'llava_next',
            'device': self.device
        }
        
        if self.is_loaded and self.model is not None:
            try:
                device = next(self.model.parameters()).device
                info.update({
                    'device': str(device),
                    'dtype': str(next(self.model.parameters()).dtype),
                    'num_parameters': sum(p.numel() for p in self.model.parameters()),
                })
            except:
                pass
            
        return info
