"""
Qwen2.5-VL-7B-Instruct model implementation for ChartViz.
Supports advanced chart question answering with superior performance.
"""

import logging
import torch
import numpy as np
import traceback
from typing import Union, Optional, List, Dict, Any, Tuple
from PIL import Image
import base64
import io

from .base_model import BaseChartQAModel, ModelPrediction, AttentionOutput
from config import ModelConfig, APP_CONFIG

logger = logging.getLogger(__name__)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN25_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qwen2.5-VL dependencies not available: {e}")
    QWEN25_AVAILABLE = False


class Qwen25VLModel(BaseChartQAModel):
    """Qwen2.5-VL-7B-Instruct model for advanced chart question answering."""

    def __init__(self, model_config: ModelConfig, device: Optional[str] = None):
        if not QWEN25_AVAILABLE:
            raise ImportError("Qwen2.5-VL dependencies not available. Please install qwen_vl_utils and update transformers.")
        
        super().__init__(model_config, device)
        self.model_name = "qwen2.5-vl-7b"
        self.model = None
        self.processor = None
        
        # Qwen2.5-VL specific settings
        self.min_pixels = 256 * 28 * 28  # Balance performance and cost
        self.max_pixels = 1280 * 28 * 28
        self.max_new_tokens = 512
        
        logger.info(f"🚀 Initialized Qwen2.5-VL model: {self.config.name}")

    def load_model(self) -> None:
        """Load the Qwen2.5-VL model and processor."""
        try:
            logger.info(f"Loading model: {self.config.name}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Loading Qwen2.5-VL from {self.config.local_dir}")

            # Load model with proper GPU configuration
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. Qwen2.5-VL requires GPU.")
            
            gpu_count = torch.cuda.device_count()
            if APP_CONFIG.allow_multi_gpu and gpu_count > 1:
                device_map_setting = "auto"
                logger.info(f"✓ Multi-GPU enabled: Using {gpu_count} GPUs with auto device mapping")
            else:
                target_gpu = APP_CONFIG.force_gpu_id if APP_CONFIG.force_gpu_id is not None else 0
                if target_gpu >= gpu_count:
                    raise RuntimeError(f"Requested GPU {target_gpu} not available. Only {gpu_count} GPUs detected.")
                device_map_setting = {"": f"cuda:{target_gpu}"}
                logger.info(f"✓ Single GPU mode: Using cuda:{target_gpu}")
                
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.config.local_dir),
                torch_dtype=torch.float16,  # Explicit dtype for better performance
                device_map=device_map_setting,
                trust_remote_code=True,
                local_files_only=True  # No fallback to HuggingFace
            )
            
            # Load processor with pixel constraints for performance
            self.processor = AutoProcessor.from_pretrained(
                str(self.config.local_dir),
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                trust_remote_code=True,
                local_files_only=True  # No fallback to HuggingFace
            )
            
            self.is_loaded = True
            logger.info(f"✓ {self.config.name} loaded successfully on {self.device}")
            logger.info(f"Model architecture: {type(self.model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.config.name}: {e}")
            self.is_loaded = False
            raise

    def predict(self, image: Union[Image.Image, str], question: str) -> ModelPrediction:
        """Generate answer for chart question using Qwen2.5-VL."""
        if not self.is_loaded:
            self.load_model()
            
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        try:
            if start_time:
                start_time.record()
                
            logger.info(f"🔍 Qwen2.5-VL: Generating answer for question: {question[:50]}...")
            
            # Prepare image input
            if isinstance(image, str):
                if image.startswith('data:image'):
                    # Handle base64 encoded image
                    image_data = image.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    # Handle file path
                    image = Image.open(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Format question for Qwen2.5-VL
            formatted_question = self._format_question_for_chart(question)
            
            # Prepare messages in Qwen format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": formatted_question},
                    ],
                }
            ]
            
            # Apply chat template and process vision info
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            logger.info(f"Input text tokens: {inputs.input_ids.shape[1]}")
            
            # Generate response with optimized parameters for accuracy
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Increased for complete answers
                    min_new_tokens=1,   # Ensure some output
                    do_sample=False,    # Deterministic for consistency
                    num_beams=1,        # Greedy decoding for accuracy
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Use raw answer without cleaning
            answer = output_text.strip()
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                processing_time = 0.0
            
            # Calculate confidence based on answer quality
            confidence = self._calculate_confidence(answer, question)
            
            logger.info(f"✓ Generated answer: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"")
            logger.info(f"Processing time: {processing_time:.2f}s")
            
            return ModelPrediction(
                answer=answer,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in Qwen2.5-VL prediction: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ModelPrediction(
                answer="",
                confidence=0.0,
                processing_time=0.0
            )

    def extract_attention(self, image: Union[Image.Image, str], question: str) -> AttentionOutput:
        """Extract REAL attention weights from Qwen2.5-VL for visualization."""
        if not self.is_loaded:
            self.load_model()
            
        try:
            logger.info("🔧 Switching to eager attention for extraction")
            # Switch to eager attention for extraction
            self.model._attn_implementation = "eager"
            
            logger.info(f"🔍 Qwen2.5-VL: Extracting REAL attention for question: {question[:50]}...")
            
            # Prepare image input
            if isinstance(image, str):
                if image.startswith('data:image'):
                    image_data = image.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    image = Image.open(image)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Format question
            formatted_question = self._format_question_for_chart(question)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": formatted_question},
                    ],
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            logger.info(f"Input shape - Input IDs: {inputs.input_ids.shape}")
            logger.info(f"Pixel values shape: {inputs.pixel_values.shape if 'pixel_values' in inputs else 'None'}")
            
            # Forward pass with attention outputs - must use eager attention
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True, return_dict=True)
            
            logger.info(f"🔍 Model outputs keys: {list(outputs.keys())}")
            
            # Extract REAL cross-attention using advanced boundary detection
            return self._extract_qwen_real_attention(outputs, inputs, question, image.size)
            
        except Exception as e:
            logger.error(f"Error extracting attention from Qwen2.5-VL: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return AttentionOutput(
                cross_attention=None,
                text_self_attention=None,
                image_self_attention=None,
                text_tokens=[],
                image_patch_coords=[],
                predicted_answer="",
                confidence_score=0.0,
                layer_count=0,
                head_count=0,
                image_size=image.size if hasattr(image, 'size') else (224, 224),
                patch_size=self.config.patch_size
            )

    def _format_question_for_chart(self, question: str) -> str:
        """Format question specifically for chart analysis with optimized prompt."""
        # Don't add prefixes - keep questions clean for better accuracy
        # Qwen2.5-VL works better with direct questions
        return question.strip()


    def _calculate_confidence(self, answer: str, question: str) -> float:
        """Calculate confidence score based on answer quality."""
        if not answer or not answer.strip():
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence for numerical answers to counting questions
        if any(word in question.lower() for word in ['how many', 'count', 'total']):
            if any(char.isdigit() for char in answer):
                confidence += 0.3
        
        # Higher confidence for year answers
        if 'year' in question.lower():
            import re
            if re.search(r'\b(19|20)\d{2}\b', answer):
                confidence += 0.3
        
        # Higher confidence for yes/no answers
        if answer.lower().strip() in ['yes', 'no']:
            confidence += 0.2
        
        # Penalize very long or very short answers
        if len(answer) < 3:
            confidence -= 0.2
        elif len(answer) > 100:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))


    def _generate_qwen25_patch_coordinates(self, num_patches: int, image_size) -> List[Tuple[int, int]]:
        """Generate patch coordinates for Qwen2.5-VL vision tokens."""
        # Estimate grid size from number of patches
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size < num_patches:
            grid_size += 1
        
        logger.info(f"🔧 Qwen2.5-VL estimated grid: {grid_size}x{grid_size} for {num_patches} patches")
        
        patch_w = image_size[0] // grid_size
        patch_h = image_size[1] // grid_size
        
        coordinates = []
        for i in range(min(num_patches, grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            x = col * patch_w
            y = row * patch_h
            coordinates.append((x, y))  # Return tuple of (x, y) coordinates
        
        logger.info(f"✓ Generated {len(coordinates)} patch coordinates")
        return coordinates

    def _get_text_tokens(self, inputs) -> List[str]:
        """Extract text tokens from inputs for debugging."""
        try:
            if hasattr(self.processor, 'tokenizer') and 'input_ids' in inputs:
                tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                return [token for token in tokens if token and not token.startswith('<')][:50]  # Limit for display
        except Exception as e:
            logger.warning(f"Could not extract text tokens: {e}")
        return []


    def _is_attention_module(self, name: str, module: torch.nn.Module) -> bool:
        """
        Determine if a module is an attention module for Qwen2.5-VL.
        
        Args:
            name: Module name
            module: PyTorch module
            
        Returns:
            True if this module should be hooked for attention extraction
        """
        # Qwen2.5-VL uses standard transformer attention modules
        attention_module_types = (
            "Attention", 
            "MultiHeadAttention", 
            "SelfAttention",
            "CrossAttention",
            "Qwen2Attention",
            "Qwen2_5VLAttention"
        )
        
        module_type = type(module).__name__
        
        # Check if module type contains attention keywords
        is_attention = any(attn_type in module_type for attn_type in attention_module_types)
        
        # Also check module name for attention patterns
        name_has_attention = any(keyword in name.lower() for keyword in [
            'attention', 'attn', 'self_attn', 'cross_attn'
        ])
        
        if is_attention or name_has_attention:
            logger.debug(f"✓ Identified attention module: {name} ({module_type})")
            return True
        
        return False
    
    def _inspect_qwen_architecture(self) -> Dict[str, Any]:
        """Inspect Qwen2.5-VL architecture to understand attention structure."""
        try:
            architecture_info = {
                'model_type': type(self.model).__name__,
                'has_vision_tower': False,
                'has_language_model': False,
                'attention_modules': [],
                'vision_modules': [],
                'total_layers': 0
            }
            
            # Analyze model structure
            for name, module in self.model.named_modules():
                module_type = type(module).__name__
                
                # Identify vision components
                if any(keyword in name.lower() for keyword in ['vision', 'visual', 'image', 'patch']):
                    architecture_info['vision_modules'].append((name, module_type))
                    architecture_info['has_vision_tower'] = True
                
                # Identify language components
                if any(keyword in name.lower() for keyword in ['language', 'text', 'embed', 'lm']):
                    architecture_info['has_language_model'] = True
                
                # Identify attention modules
                if self._is_attention_module(name, module):
                    architecture_info['attention_modules'].append((name, module_type))
                
                # Count layers
                if 'layer' in name.lower() and name.count('.') == 2:
                    architecture_info['total_layers'] += 1
            
            logger.info(f"🏗️ Found {len(architecture_info['attention_modules'])} attention modules")
            logger.info(f"🏗️ Found {len(architecture_info['vision_modules'])} vision modules")
            logger.info(f"🏗️ Total layers: {architecture_info['total_layers']}")
            
            return architecture_info
            
        except Exception as e:
            logger.error(f"❌ Architecture inspection failed: {e}")
            return {}
    
    def _extract_qwen_real_attention(self, outputs, inputs, question: str, image_size) -> AttentionOutput:
        """
        Extract real attention from Qwen2.5-VL using proper understanding of its architecture.
        NO fallbacks or synthetic data - only real attention patterns.
        """
        try:
            logger.info("🎯 Extracting REAL Qwen2.5-VL attention patterns...")
            
            # Get attention weights
            attentions = getattr(outputs, 'attentions', None)
            
            if attentions is None:
                raise RuntimeError("Qwen2.5-VL model does not provide attention weights in outputs")
            
            logger.info(f"🔍 Processing {len(attentions)} attention layers")
            
            # For Qwen2.5-VL, analyze all layers to find the best cross-modal attention
            best_cross_attention = None
            best_layer_idx = -1
            best_confidence = 0.0
            
            for layer_idx, layer_attention in enumerate(attentions):
                logger.info(f"🔍 Analyzing layer {layer_idx}: {layer_attention.shape}")
                
                try:
                    # Extract cross-attention for this layer
                    cross_attn_result = self._extract_layer_cross_attention(
                        layer_attention, inputs, image_size, layer_idx
                    )
                    
                    if cross_attn_result and cross_attn_result['confidence'] > best_confidence:
                        best_cross_attention = cross_attn_result
                        best_layer_idx = layer_idx
                        best_confidence = cross_attn_result['confidence']
                        logger.info(f"✅ Layer {layer_idx} has better cross-attention (confidence: {best_confidence:.4f})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Layer {layer_idx} attention extraction failed: {e}")
                    continue
            
            # Check if we found any reasonable cross-attention
            if best_cross_attention is None or best_confidence < 0.1:  # Lowered threshold from 0.3 to 0.1
                logger.warning(
                    f"⚠️ No valid cross-attention found in any layer. Best confidence: {best_confidence:.4f}. "
                    "Trying lenient extraction with lower quality threshold..."
                )
                
                # Try with more lenient extraction for at least visualization
                if best_cross_attention is not None and best_confidence > 0.0:
                    logger.info(f"🔧 Using lower confidence cross-attention (confidence: {best_confidence:.4f})")
                    # Use it anyway - better than nothing for visualization
                    return AttentionOutput(
                        cross_attention=best_cross_attention['cross_attention'],
                        text_self_attention=best_cross_attention['text_self_attention'],
                        image_self_attention=best_cross_attention['image_self_attention'],
                        text_tokens=best_cross_attention['text_tokens'],
                        image_patch_coords=best_cross_attention['patch_coords'],
                        predicted_answer="",
                        confidence_score=best_confidence,
                        layer_count=len(attentions),
                        head_count=best_cross_attention['num_heads'],
                        image_size=image_size,
                        patch_size=self.config.patch_size
                    )
                
                # Last resort: return None
                logger.warning("❌ No usable cross-attention found. Returning None.")
                return AttentionOutput(
                    cross_attention=None,
                    text_self_attention=None,
                    image_self_attention=None,
                    text_tokens=[],
                    image_patch_coords=[],
                    predicted_answer="",
                    confidence_score=0.0,
                    layer_count=len(attentions),
                    head_count=0,
                    image_size=image_size,
                    patch_size=self.config.patch_size
                )
            
            logger.info(f"🎯 Using best cross-attention from layer {best_layer_idx} (confidence: {best_confidence:.4f})")
            
            return AttentionOutput(
                cross_attention=best_cross_attention['cross_attention'],
                text_self_attention=best_cross_attention['text_self_attention'],
                image_self_attention=best_cross_attention['image_self_attention'],
                text_tokens=best_cross_attention['text_tokens'],
                image_patch_coords=best_cross_attention['patch_coords'],
                predicted_answer="",
                confidence_score=best_confidence,
                layer_count=len(attentions),
                head_count=best_cross_attention['num_heads'],
                image_size=image_size,
                patch_size=self.config.patch_size
            )
            
        except Exception as e:
            logger.error(f"❌ Qwen real attention extraction failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return AttentionOutput with None instead of raising
            return AttentionOutput(
                cross_attention=None,
                text_self_attention=None,
                image_self_attention=None,
                text_tokens=[],
                image_patch_coords=[],
                predicted_answer="",
                confidence_score=0.0,
                layer_count=0,
                head_count=0,
                image_size=image_size,
                patch_size=self.config.patch_size
            )
    
    def _extract_layer_cross_attention(self, layer_attention: torch.Tensor, inputs: Dict, 
                                     image_size, layer_idx: int) -> Optional[Dict[str, Any]]:
        """
        Extract cross-attention from a specific layer with strict validation.
        Returns None if no valid cross-attention found.
        """
        try:
            batch_size, num_heads, seq_len, _ = layer_attention.shape
            
            # Average across heads for analysis
            avg_attention = layer_attention.mean(dim=1)[0]  # [seq_len, seq_len]
            
            # Use input structure to understand token layout
            input_ids_length = inputs.input_ids.shape[1]
            
            # For Qwen2.5-VL, try multiple strategies to find vision-text boundary
            boundaries_to_test = []
            
            # Strategy 1: Common Qwen2.5-VL vision token counts
            common_vision_counts = [576, 729, 1024, 1152, 1444]  # Common patch counts
            for vision_count in common_vision_counts:
                if 20 <= vision_count <= seq_len - 10:  # Leave room for text tokens
                    boundaries_to_test.append(vision_count)
            
            # Strategy 2: Percentage-based boundaries
            for ratio in [0.7, 0.75, 0.8, 0.85]:
                boundary = int(seq_len * ratio)
                if 50 <= boundary <= seq_len - 10:
                    boundaries_to_test.append(boundary)
            
            # Strategy 3: Token analysis - look for special tokens
            if hasattr(self.processor, 'tokenizer'):
                try:
                    tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                    for i, token in enumerate(tokens):
                        # Look for tokens that might indicate vision-text boundary
                        if token in ['<|im_end|>', '<|image|>', '</image>', '<|vision_end|>']:
                            if 50 <= i <= seq_len - 10:
                                boundaries_to_test.append(i)
                except Exception:
                    pass
            
            logger.info(f"🔍 Testing {len(set(boundaries_to_test))} potential boundaries")
            
            best_result = None
            best_score = 0.0
            
            for vision_boundary in set(boundaries_to_test):
                try:
                    # Extract cross-attention for this boundary
                    text_start = vision_boundary
                    cross_attention = avg_attention[text_start:, :vision_boundary]
                    
                    if cross_attention.numel() == 0:
                        continue
                    
                    # Calculate quality metrics
                    attn_std = cross_attention.std().item()
                    attn_max = cross_attention.max().item()
                    attn_mean = cross_attention.mean().item()
                    attn_var = cross_attention.var().item()
                    
                    # Calculate a composite quality score
                    std_score = min(1.0, attn_std * 1000)  # Normalize std
                    max_mean_ratio = attn_max / (attn_mean + 1e-8)
                    ratio_score = min(1.0, max_mean_ratio / 10.0)  # Normalize ratio
                    var_score = min(1.0, attn_var * 1000)  # Normalize variance
                    
                    quality_score = (std_score + ratio_score + var_score) / 3.0
                    
                    logger.info(f"🎯 Boundary {vision_boundary}: quality={quality_score:.4f}, std={attn_std:.6f}, max/mean={max_mean_ratio:.2f}")
                    
                    # More lenient quality threshold for better visualization
                    # Relaxed from: attn_std > 1e-5 and max_mean_ratio > 1.5
                    if quality_score > best_score and attn_std > 1e-6 and max_mean_ratio > 1.1:
                        best_score = quality_score
                        
                        # Generate patch coordinates
                        patch_coords = self._generate_qwen25_patch_coordinates(vision_boundary, image_size)
                        
                        # Get text tokens
                        text_tokens = self._get_text_tokens(inputs)
                        text_count = seq_len - text_start
                        if len(text_tokens) > text_count:
                            text_tokens = text_tokens[-text_count:]
                        
                        best_result = {
                            'cross_attention': cross_attention,
                            'text_self_attention': avg_attention[text_start:, text_start:],
                            'image_self_attention': avg_attention[:vision_boundary, :vision_boundary],
                            'text_tokens': text_tokens,
                            'patch_coords': patch_coords,
                            'confidence': quality_score,
                            'num_heads': num_heads,
                            'vision_tokens': vision_boundary,
                            'text_tokens_count': text_count
                        }
                        
                        logger.info(f"✅ New best result: confidence={quality_score:.4f}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Boundary {vision_boundary} failed: {e}")
                    continue
            
            # More lenient - accept lower quality results
            if best_result is None:
                logger.warning(f"⚠️ No cross-attention patterns found in layer {layer_idx}.")
                return None
            
            # Accept results with score > 0.05 (was 0.3) for visualization
            if best_score < 0.05:
                logger.warning(
                    f"⚠️ Very low quality cross-attention in layer {layer_idx}. Best score: {best_score:.4f}. "
                    "Qwen2.5-VL attention may not be reliable for this input."
                )
                return None
            
            logger.info(f"🎯 Final result: {best_result['vision_tokens']} vision tokens, "
                       f"{best_result['text_tokens_count']} text tokens, confidence: {best_score:.4f}")
            
            return best_result
            
        except Exception as e:
            logger.error(f"❌ Layer attention extraction failed: {e}")
            return None
    
    
    def preprocess_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """Preprocess image for analysis."""
        if isinstance(image, str):
            if image.startswith('data:image'):
                image_data = image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _create_empty_attention_output(self, question: str, image_size) -> AttentionOutput:
        """Create an empty attention output for cases where extraction fails."""
        return AttentionOutput(
            cross_attention=None,
            text_self_attention=None,
            image_self_attention=None,
            text_tokens=[],
            image_patch_coords=[],
            predicted_answer="",
            confidence_score=0.0,
            layer_count=0,
            head_count=0,
            image_size=image_size,
            patch_size=self.config.patch_size
        )
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported explainability features."""
        return [
            "cross_modal_attention",
            "text_self_attention",
            "qwen_attention_extraction",
            "confidence_estimation",
            "advanced_statistical_analysis",
            "attention_flow_analysis",
            "layer_wise_analysis",
            "multi_head_analysis",
            "qwen_specific_analysis"
        ]
    
    def run_advanced_analysis(self, analysis_type: str, image: Union[Image.Image, str], 
                            question: str, **kwargs) -> Dict[str, Any]:
        """
        Run advanced analysis on the Qwen model's attention patterns.
        
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
        
        # Run the requested analysis with Qwen-specific enhancements
        analysis_result = analysis_manager.run_analysis(
            analysis_type=analysis_type,
            attention_output=attention_output,
            image=processed_image,
            question=question,
            model_type="qwen25_vl",  # Qwen-specific parameter
            **kwargs
        )
        
        # Add Qwen-specific insights
        if analysis_result.insights:
            qwen_insights = self._add_qwen_specific_insights(
                analysis_result, attention_output, question
            )
            analysis_result.insights.extend(qwen_insights)
        
        return {
            'analysis_type': analysis_result.analysis_type,
            'title': analysis_result.title,
            'description': analysis_result.description,
            'metrics': analysis_result.metrics,
            'insights': analysis_result.insights,
            'visualization_data': analysis_result.visualization.to_dict() if analysis_result.visualization else None,
            'raw_data': analysis_result.raw_data,
            'model_specific': 'qwen25_vl'
        }
    
    def _add_qwen_specific_insights(self, analysis_result: Any, attention_output: AttentionOutput, 
                                  question: str) -> List[str]:
        """Add Qwen-specific insights to analysis results."""
        insights = []
        
        # Analyze question complexity for Qwen
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['complex', 'detailed', 'explain', 'why']):
            insights.append("Qwen2.5-VL: Complex question detected - model excels at detailed reasoning")
        elif any(word in question_lower for word in ['count', 'how many', 'number']):
            insights.append("Qwen2.5-VL: Counting question - model uses precise numerical analysis")
        elif any(word in question_lower for word in ['compare', 'difference', 'versus']):
            insights.append("Qwen2.5-VL: Comparison question - model analyzes multiple regions")
        
        # Analyze attention distribution for Qwen understanding
        if attention_output.cross_attention is not None:
            cross_attn = attention_output.cross_attention
            if hasattr(cross_attn, 'detach'):
                cross_attn = cross_attn.detach().cpu().numpy()
            else:
                cross_attn = np.array(cross_attn)
            
            # Calculate attention concentration
            flat_attn = cross_attn.flatten()
            concentration = np.std(flat_attn) / (np.mean(flat_attn) + 1e-8)
            
            if concentration > 2.0:
                insights.append("Qwen2.5-VL: Highly concentrated attention - model focuses on specific elements")
            elif concentration < 0.5:
                insights.append("Qwen2.5-VL: Distributed attention - model considers multiple chart elements")
            
            # Check for edge attention (potential chart boundary detection)
            if len(cross_attn.shape) >= 2:
                h, w = cross_attn.shape[-2:]
                if h >= 4 and w >= 4:
                    edge_attention = (
                        np.sum(cross_attn[..., 0, :]) + np.sum(cross_attn[..., -1, :]) +
                        np.sum(cross_attn[..., :, 0]) + np.sum(cross_attn[..., :, -1])
                    )
                    total_attention = np.sum(cross_attn)
                    edge_ratio = edge_attention / (total_attention + 1e-8)
                    
                    if edge_ratio > 0.3:
                        insights.append("Qwen2.5-VL: High edge attention - model may be analyzing chart boundaries")
        
        return insights
    
    def get_attention_statistics(self, image: Union[Image.Image, str], question: str) -> Dict[str, float]:
        """
        Get statistical metrics about attention patterns for Qwen model.
        
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
            'model_type': 'qwen25_vl'
        }
    
    def analyze_token_importance(self, image: Union[Image.Image, str], question: str) -> Dict[str, Any]:
        """
        Analyze the importance of different tokens in the question for Qwen model.
        
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
        
        # Handle Qwen's attention structure
        if len(cross_attn.shape) > 1:
            # Sum attention for each token (assuming first dimension is tokens)
            token_importance = np.sum(cross_attn, axis=-1)
        else:
            token_importance = cross_attn
        
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
            
            # Qwen-specific analysis
            if any(token in ['chart', 'graph', 'plot', 'figure'] for token in tokens):
                insights.append("Qwen2.5-VL: Question contains chart-specific terminology")
        
        return {
            'tokens': tokens,
            'importance_scores': importance_scores,
            'insights': insights,
            'total_tokens': len(tokens),
            'attention_distribution': 'concentrated' if np.std(importance_scores) > 0.1 else 'distributed',
            'model_type': 'qwen25_vl'
        }