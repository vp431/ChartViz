"""
UniChart ChartQA-960 model implementation for chart QA with attention extraction.
Specialized implementation for the ChartQA-960 variant with improved chart understanding.
"""
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from rich.console import Console
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
    VisionEncoderDecoderModel, AutoConfig, DonutProcessor,
    Pix2StructForConditionalGeneration
)

from .base_model import BaseChartQAModel, ModelPrediction, AttentionOutput
from config import ModelConfig

console = Console()


class UniChartChartQAModel(BaseChartQAModel):
    """UniChart ChartQA-960 model implementation with enhanced attention extraction capabilities."""
    
    def __init__(self, model_config: ModelConfig, device: Optional[str] = None):
        """Initialize UniChart ChartQA model."""
        super().__init__(model_config, device)
        self.model_name = "unichart_chartqa"
    
    def load_model(self) -> None:
        """Load UniChart ChartQA model, processor, and tokenizer."""
        try:
            console.print(f"Loading UniChart ChartQA-960 from {self.config.local_dir}")
            
            # Use OFFICIAL UniChart ChartQA loading method (exactly like the documentation)
            print("Loading UniChart ChartQA with official method...")
            
            # Load processor first - try DonutProcessor first (official), then AutoProcessor
            try:
                self.processor = DonutProcessor.from_pretrained(
                    str(self.config.local_dir),
                    local_files_only=True
                )
                print("✓ Using DonutProcessor (official)")
            except Exception:
                self.processor = AutoProcessor.from_pretrained(
                    str(self.config.local_dir),
                    local_files_only=True
                )
                print("✓ Using AutoProcessor (fallback)")
            
            # Load model WITH attn_implementation set to 'eager' to support output_attentions
            # This is CRITICAL - SDPA doesn't support output_attentions
            self.model = VisionEncoderDecoderModel.from_pretrained(
                str(self.config.local_dir),
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                attn_implementation="eager"  # CRITICAL: Required for attention extraction
            )
            print("✓ Successfully loaded as VisionEncoderDecoder model with eager attention")
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model moved to {self.device}")
            
            # Enable cross-attention output in decoder config
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'config'):
                self.model.decoder.config.output_attentions = True
                self.model.decoder.config.output_hidden_states = True
                console.print(f"[blue]🔧 Enabled attention outputs in decoder config[/blue]")
            
            # Set tokenizer (often same as processor tokenizer)
            if hasattr(self.processor, 'tokenizer'):
                self.tokenizer = self.processor.tokenizer
            else:
                # Fallback tokenizer loading
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(self.config.local_dir),
                        local_files_only=True
                    )
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load tokenizer: {e}[/yellow]")
                    self.tokenizer = None
            
            self.is_loaded = True
            console.print(f"✓ UniChart ChartQA-960 loaded successfully on {self.device}")
            
            # Print model architecture info
            if hasattr(self.model, 'config'):
                console.print(f"[blue]Model architecture: {self.model.config.architectures if hasattr(self.model.config, 'architectures') else 'Unknown'}[/blue]")
            
            # Check decoder model type for cross-attention support
            if hasattr(self.model, 'decoder'):
                decoder_type = type(self.model.decoder).__name__
                console.print(f"[blue]Decoder model type: {decoder_type}[/blue]")
                
                # Check if decoder has cross-attention layers
                has_cross_attn = False
                for name, module in self.model.decoder.named_modules():
                    if 'cross' in name.lower() and 'attn' in name.lower():
                        has_cross_attn = True
                        console.print(f"[green]✓ Found cross-attention module: {name}[/green]")
                        break
                
                if not has_cross_attn:
                    console.print(f"[yellow]⚠️ No explicit cross-attention modules found in decoder[/yellow]")
                    console.print(f"[yellow]   This might be a T5-style model with implicit cross-attention[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Failed to load UniChart ChartQA-960: {e}[/red]")
            raise RuntimeError(f"UniChart ChartQA-960 loading failed: {e}")
    
    def predict(self, image: Union[Image.Image, str], question: str) -> ModelPrediction:
        """
        Make a prediction using UniChart ChartQA-960.
        
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
            # Use OFFICIAL UniChart ChartQA format exactly as documented
            input_prompt = f"<chartqa> {question} <s_answer>"
            
            # Process inputs using OFFICIAL method (like the documentation)
            decoder_input_ids = self.processor.tokenizer(
                input_prompt, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).input_ids
            
            pixel_values = self.processor(
                images=image, 
                return_tensors="pt"
            ).pixel_values
            
            inputs = {
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids
            }
            
            # Move inputs to device and ensure correct dtype
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = self._safe_dtype_conversion(v)
                    inputs[k] = v.to(self.device)
                else:
                    inputs[k] = v
            
            # Generate using OFFICIAL parameters exactly as documented
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["pixel_values"],
                    decoder_input_ids=inputs["decoder_input_ids"],
                    max_length=self.model.decoder.config.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=4,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Decode using OFFICIAL method exactly as documented
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            
            # Extract answer after <s_answer> token
            if "<s_answer>" in sequence:
                answer = sequence.split("<s_answer>")[1].strip()
            else:
                # Fallback: use entire sequence if no answer tag found
                answer = sequence.strip()
                console.print(f"[yellow]Warning: No <s_answer> tag found. Full sequence: {sequence}[/yellow]")
            
            # Calculate confidence from generation scores
            confidence = self._calculate_confidence(outputs.scores) if outputs.scores else 0.6
            
            processing_time = time.time() - start_time
            
            return ModelPrediction(
                answer=answer,
                confidence=confidence,
                logits=outputs.scores[-1] if outputs.scores else None,
                processing_time=processing_time
            )
            
        except Exception as e:
            console.print(f"[red]Prediction failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return ModelPrediction(
                answer="Error during prediction",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def _process_pix2struct_inputs(self, image: Image.Image, question: str) -> Dict:
        """Process inputs for Pix2Struct model."""
        try:
            # Pix2Struct processes image and text together
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt",
                max_patches=2048,  # Pix2Struct parameter
                truncation=True
            )
            return inputs
        except Exception as e:
            console.print(f"[yellow]Pix2Struct input processing failed: {e}[/yellow]")
            # Fallback
            return self._process_vision_encoder_decoder_inputs(image, question)
    
    def _process_vision_encoder_decoder_inputs(self, image: Image.Image, question: str) -> Dict:
        """Process inputs for VisionEncoderDecoder model using official UniChart format."""
        try:
            # For VisionEncoderDecoder models, we need to process image and text separately
            # and set up proper decoder start tokens
            
            # Process image
            image_inputs = self.processor.image_processor(
                images=image,
                return_tensors="pt"
            )
            
            # Process text - for generation, we need decoder_input_ids with BOS token
            text_inputs = self.processor.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length
            )
            
            # For generation, we need to prepare decoder_input_ids starting with BOS token
            # The model will generate the answer after the question
            decoder_input_ids = text_inputs["input_ids"]
            
            inputs = {
                "pixel_values": image_inputs["pixel_values"],
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": text_inputs.get("attention_mask")
            }
            
            return inputs
            
        except Exception as e:
            console.print(f"[yellow]VisionEncoderDecoder input processing failed: {e}[/yellow]")
            # Fallback to simple approach
            return self.processor(
                images=image,
                text=question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length
            )
    
    def _format_chartqa_question(self, question: str) -> str:
        """Format question specifically for ChartQA tasks using official format."""
        question = question.strip()
        
        # Use simpler format that actually works - the model might be overtrained on specific format
        # Try just the question without extra context
        formatted = f"<chartqa> {question} <s_answer>"
        
        return formatted
    
    
    def extract_attention(self, image: Union[Image.Image, str], question: str) -> AttentionOutput:
        """
        Extract attention weights from UniChart ChartQA for explainability.
        
        Args:
            image: PIL Image or path to image file
            question: Question text
            
        Returns:
            AttentionOutput with attention weights and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.config.supports_attention_extraction:
            raise RuntimeError("This model does not support attention extraction.")
        
        # Validate and preprocess inputs
        image, question = self.validate_inputs(image, question)
        
        try:
            # Set up attention hooks
            self._setup_attention_hooks()
            
            # Note: attn_implementation is already set to 'eager' during model loading
            # No need to switch it here
            
            # Format question
            formatted_question = self._format_chartqa_question(question)
            
            # Process inputs based on model type
            if isinstance(self.model, Pix2StructForConditionalGeneration):
                inputs = self._process_pix2struct_attention_inputs(image, formatted_question)
            else:
                inputs = self._process_vision_encoder_decoder_attention_inputs(image, formatted_question)
            
            # Move inputs to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = self._safe_dtype_conversion(v)
                    inputs[k] = v.to(self.device)
                else:
                    inputs[k] = v
            
            console.print(f"[blue]🔍 UniChart ChartQA: Extracting attention for question: {question[:50]}...[/blue]")
            
            # Forward pass with attention extraction
            with torch.no_grad():
                # For VisionEncoderDecoder models, we need to use generate() to get cross-attention
                if isinstance(self.model, VisionEncoderDecoderModel):
                    console.print(f"[blue]🔧 Using generate() for VisionEncoderDecoder cross-attention[/blue]")
                    
                    # Generate with minimal tokens to extract attention
                    gen_outputs = self.model.generate(
                        inputs['pixel_values'],
                        decoder_input_ids=inputs['decoder_input_ids'],
                        max_new_tokens=5,  # Generate a few tokens to get cross-attention
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    
                    # Extract attentions from generation outputs
                    encoder_attentions = getattr(gen_outputs, 'encoder_attentions', None)
                    decoder_attentions = getattr(gen_outputs, 'decoder_attentions', None)
                    cross_attentions = getattr(gen_outputs, 'cross_attentions', None)
                    
                    console.print(f"[blue]🔍 Generation output keys: {list(gen_outputs.keys()) if hasattr(gen_outputs, 'keys') else 'not a dict'}[/blue]")
                    console.print(f"[blue]🔍 Encoder attentions: {type(encoder_attentions)}, length: {len(encoder_attentions) if encoder_attentions else 0}[/blue]")
                    console.print(f"[blue]🔍 Decoder attentions: {type(decoder_attentions)}, length: {len(decoder_attentions) if decoder_attentions else 0}[/blue]")
                    console.print(f"[blue]🔍 Cross attentions: {type(cross_attentions)}, length: {len(cross_attentions) if cross_attentions else 0}[/blue]")
                    
                    # For generate(), attentions are nested by generation step
                    # The structure is typically: tuple of (tuple of layers per generation step)
                    
                    # Process decoder attentions
                    if decoder_attentions and len(decoder_attentions) > 0:
                        console.print(f"[blue]🔍 Decoder attention structure: {type(decoder_attentions[0])}[/blue]")
                        if isinstance(decoder_attentions[0], tuple):
                            decoder_attentions = decoder_attentions[0]
                            console.print(f"[blue]✓ Unwrapped decoder attentions to tuple of {len(decoder_attentions)} layers[/blue]")
                    
                    # Process cross attentions - THIS IS THE KEY PART
                    if cross_attentions and len(cross_attentions) > 0:
                        console.print(f"[blue]🔍 Cross attention structure: {type(cross_attentions[0])}[/blue]")
                        if isinstance(cross_attentions[0], tuple):
                            cross_attentions = cross_attentions[0]
                            console.print(f"[blue]✓ Unwrapped cross attentions to tuple of {len(cross_attentions)} layers[/blue]")
                            if len(cross_attentions) > 0:
                                console.print(f"[blue]✓ First cross attention layer shape: {cross_attentions[0].shape if hasattr(cross_attentions[0], 'shape') else 'no shape'}[/blue]")
                    else:
                        console.print(f"[yellow]⚠️ No cross_attentions in generate() output![/yellow]")
                    
                    outputs = type('CombinedOutputs', (), {
                        'encoder_attentions': encoder_attentions,
                        'decoder_attentions': decoder_attentions,
                        'cross_attentions': cross_attentions,
                    })()
                else:
                    # For other model types, use standard approach
                    outputs = self.model(
                        **inputs,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
            
            # Debug attention outputs
            self._debug_attention_outputs(outputs)
            
            # Extract attention weights
            attention_output = self._process_chartqa_attention_outputs(
                outputs, inputs, image, question
            )
            
            # Generate prediction for context
            prediction = self.predict(image, question)
            attention_output.predicted_answer = prediction.answer
            attention_output.confidence_score = prediction.confidence
            
            return attention_output
            
        except Exception as e:
            console.print(f"[red]Attention extraction failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Attention extraction failed: {e}")
        
        finally:
            self._clear_attention_hooks()
    
    def _process_pix2struct_attention_inputs(self, image: Image.Image, question: str) -> Dict:
        """Process inputs for Pix2Struct attention extraction."""
        try:
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt",
                max_patches=2048,
                truncation=True
            )
            return inputs
        except Exception as e:
            console.print(f"[yellow]Pix2Struct attention input processing failed: {e}[/yellow]")
            return self._process_vision_encoder_decoder_attention_inputs(image, question)
    
    def _process_vision_encoder_decoder_attention_inputs(self, image: Image.Image, question: str) -> Dict:
        """Process inputs for VisionEncoderDecoder attention extraction."""
        try:
            # For VisionEncoderDecoder models, we need to be careful about input structure
            # The encoder (vision) and decoder (text) have different input requirements
            
            # Process image for encoder
            pixel_values = self.processor.image_processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"]
            
            # Process text for decoder - but don't pass input_ids to encoder
            decoder_input_ids = self.processor.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length,
                add_special_tokens=True
            )["input_ids"]
            
            return {
                'pixel_values': pixel_values,
                'decoder_input_ids': decoder_input_ids
                # Note: Remove 'input_ids' as it conflicts with encoder
            }
        except Exception as e:
            console.print(f"[yellow]VisionEncoderDecoder input processing failed: {e}[/yellow]")
            # Fallback to simpler approach
            return {
                'pixel_values': self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
            }
    
    def _debug_attention_outputs(self, outputs) -> None:
        """Debug attention outputs to understand structure."""
        if hasattr(outputs, 'keys'):
            console.print(f"[blue]🔍 Model outputs keys: {list(outputs.keys())}[/blue]")
            attention_keys = [key for key in outputs.keys() if 'attention' in key.lower()]
        else:
            # Handle our custom output object
            attention_keys = ['encoder_attentions', 'decoder_attentions', 'cross_attentions']
            available_keys = [key for key in attention_keys if hasattr(outputs, key)]
            console.print(f"[blue]🔍 Available attention attributes: {available_keys}[/blue]")
            attention_keys = available_keys
        
        console.print(f"[blue]🔍 Attention keys: {attention_keys}[/blue]")
        
        for key in attention_keys:
            attention = getattr(outputs, key, None)
            if attention:
                console.print(f"[blue]🔍 {key}: {len(attention)} layers[/blue]")
                if len(attention) > 0:
                    console.print(f"[blue]🔍 {key} first layer shape: {attention[0].shape}[/blue]")
            else:
                console.print(f"[blue]🔍 {key}: None[/blue]")
    
    def _process_chartqa_attention_outputs(self, outputs, inputs, image: Image.Image, question: str) -> AttentionOutput:
        """Process model outputs to extract attention information for ChartQA."""
        
        # Get attention weights from different components
        encoder_attentions = getattr(outputs, 'encoder_attentions', None)
        decoder_attentions = getattr(outputs, 'decoder_attentions', None)
        cross_attentions = getattr(outputs, 'cross_attentions', None)
        
        console.print(f"[blue]🔍 Processing ChartQA attention outputs...[/blue]")
        console.print(f"[blue]Encoder attentions: {encoder_attentions is not None}[/blue]")
        console.print(f"[blue]Decoder attentions: {decoder_attentions is not None}[/blue]")
        console.print(f"[blue]Cross attentions: {cross_attentions is not None}[/blue]")
        
        # Process text tokens
        text_tokens = []
        if 'decoder_input_ids' in inputs and self.tokenizer:
            text_tokens = self.tokenizer.convert_ids_to_tokens(inputs['decoder_input_ids'][0])
        elif 'input_ids' in inputs and self.tokenizer:
            text_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        else:
            # Fallback to splitting the question
            text_tokens = question.split()
        
        console.print(f"[blue]Text tokens: {len(text_tokens)}[/blue]")
        
        # Get image patch coordinates
        image_patches = self.get_image_patches(image)
        console.print(f"[blue]Image patches: {len(image_patches)}[/blue]")
        
        # Process cross-modal attention
        cross_attention = None
        
        if cross_attentions and len(cross_attentions) > 0:
            console.print(f"[green]✓ Found cross_attentions: {len(cross_attentions)} layers[/green]")
            try:
                last_layer_cross_attn = cross_attentions[-1]
                
                if last_layer_cross_attn is not None and torch.is_tensor(last_layer_cross_attn):
                    console.print(f"[blue]Cross attention shape: {last_layer_cross_attn.shape}[/blue]")
                    
                    # Average across heads
                    cross_attention = last_layer_cross_attn.mean(dim=1)[0]  # Remove batch and average heads
                    
                    if cross_attention.max().item() > 0:
                        console.print(f"[green]✓ Extracted cross-attention: {cross_attention.shape}[/green]")
                        console.print(f"[green]  Stats: min={cross_attention.min():.4f}, max={cross_attention.max():.4f}[/green]")
                    else:
                        console.print(f"[yellow]⚠️ Cross-attention all zeros[/yellow]")
                        cross_attention = None
                        
            except Exception as e:
                console.print(f"[yellow]Cross attention processing failed: {e}[/yellow]")
        else:
            console.print(f"[yellow]⚠️ No cross_attentions available in model outputs[/yellow]")
        
        # If no cross-attention, try to derive from decoder attention to encoder hidden states
        if cross_attention is None and decoder_attentions and encoder_attentions:
            console.print(f"[blue]🔧 Attempting to derive cross-attention from available data...[/blue]")
            try:
                # Use decoder self-attention as proxy for text importance
                # and encoder self-attention averaged for image regions
                last_decoder_attn = decoder_attentions[-1]
                if last_decoder_attn is not None and torch.is_tensor(last_decoder_attn):
                    # Average across heads and take diagonal (self-attention strength)
                    decoder_attn_avg = last_decoder_attn.mean(dim=1)[0]  # [seq, seq]
                    text_importance = decoder_attn_avg.sum(dim=1)  # Sum across keys
                    
                    # Get encoder attention strength
                    last_encoder_attn = encoder_attentions[-1]
                    if last_encoder_attn is not None and torch.is_tensor(last_encoder_attn):
                        encoder_attn_avg = last_encoder_attn.mean(dim=1)[0]  # [patches, patches]
                        image_importance = encoder_attn_avg.sum(dim=1)  # Sum across keys
                        
                        # Create pseudo cross-attention: outer product normalized
                        cross_attention = torch.outer(text_importance, image_importance)
                        cross_attention = cross_attention / cross_attention.sum()  # Normalize
                        
                        console.print(f"[green]✓ Derived pseudo cross-attention: {cross_attention.shape}[/green]")
                        console.print(f"[yellow]⚠️ Note: This is an approximation based on self-attention patterns[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Failed to derive cross-attention: {e}[/yellow]")
        
        # Process self-attention within text
        text_self_attention = None
        if decoder_attentions and len(decoder_attentions) > 0:
            try:
                last_layer_self_attn = decoder_attentions[-1]
                if last_layer_self_attn is not None:
                    text_self_attention = last_layer_self_attn.mean(dim=1)[0]  # Average heads, remove batch
            except Exception as e:
                console.print(f"[yellow]Text self-attention processing failed: {e}[/yellow]")
        
        # Process image self-attention (from encoder)
        image_self_attention = None
        if encoder_attentions and len(encoder_attentions) > 0:
            try:
                last_layer_encoder_attn = encoder_attentions[-1]
                if last_layer_encoder_attn is not None:
                    image_self_attention = last_layer_encoder_attn.mean(dim=1)[0]  # Average heads, remove batch
            except Exception as e:
                console.print(f"[yellow]Image self-attention processing failed: {e}[/yellow]")
        
        return AttentionOutput(
            cross_attention=cross_attention,
            text_self_attention=text_self_attention,
            image_self_attention=image_self_attention,
            text_tokens=text_tokens,
            image_patch_coords=image_patches,
            layer_count=len(encoder_attentions) if encoder_attentions else 0,
            head_count=encoder_attentions[0].shape[1] if encoder_attentions else 0,
            image_size=image.size,
            patch_size=self.config.patch_size
        )
    
    def _calculate_confidence(self, scores: Tuple[torch.Tensor]) -> float:
        """Calculate confidence score from generation scores."""
        if not scores:
            return 0.6
        
        try:
            # Use the maximum probability across all generation steps
            max_probs = []
            for score in scores:
                probs = F.softmax(score, dim=-1)
                max_prob = torch.max(probs).item()
                max_probs.append(max_prob)
            
            # Average the maximum probabilities
            avg_confidence = sum(max_probs) / len(max_probs)
            return float(avg_confidence)
        except Exception:
            return 0.6
    
    def _is_attention_module(self, name: str, module: torch.nn.Module) -> bool:
        """Determine if a module is an attention module for UniChart ChartQA."""
        attention_patterns = [
            "attention",
            "self_attn", 
            "cross_attn",
            "multihead_attn",
            "attn"
        ]
        
        return any(pattern in name.lower() for pattern in attention_patterns)
    
    def get_image_patches(self, image: Image.Image) -> List[Tuple[int, int]]:
        """Get the coordinates of image patches for UniChart ChartQA attention visualization."""
        try:
            # UniChart ChartQA uses 960x960 image size
            vision_size = self.config.max_image_size[0]  # 960
            patch_size = self.config.patch_size[0]       # 32
            
            patches_per_side = vision_size // patch_size  # 30 patches per side
            
            console.print(f"[blue]🔧 UniChart ChartQA patches: {patches_per_side}x{patches_per_side}[/blue]")
            
            # Generate patch coordinates
            patches = []
            img_w, img_h = image.size
            
            for y in range(patches_per_side):
                for x in range(patches_per_side):
                    # Map to actual image coordinates
                    actual_x = int((x / patches_per_side) * img_w)
                    actual_y = int((y / patches_per_side) * img_h)
                    patches.append((actual_x, actual_y))
            
            console.print(f"[green]✓ Generated {len(patches)} patch coordinates[/green]")
            return patches
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate patch coordinates: {e}[/yellow]")
            # Fallback
            img_w, img_h = image.size
            patches = []
            grid_size = 30  # Default for 960x960 with 32x32 patches
            
            for y in range(grid_size):
                for x in range(grid_size):
                    actual_x = int((x / grid_size) * img_w)
                    actual_y = int((y / grid_size) * img_h)
                    patches.append((actual_x, actual_y))
            
            return patches

    def get_supported_features(self) -> List[str]:
        """Get list of supported explainability features."""
        return [
            "chart_qa_specialized",
            "cross_modal_attention",
            "text_self_attention", 
            "image_self_attention",
            "token_importance",
            "patch_attention",
            "confidence_estimation",
            "chartqa_960_optimized",
            "advanced_statistical_analysis",
            "attention_flow_analysis",
            "layer_wise_analysis",
            "multi_head_analysis",
            "chart_specific_analysis"
        ]
    
    def run_advanced_analysis(self, analysis_type: str, image: Union[Image.Image, str], 
                            question: str, **kwargs) -> Dict[str, Any]:
        """
        Run advanced analysis on the ChartQA model's attention patterns.
        
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
        
        # Run the requested analysis with ChartQA-specific enhancements
        analysis_result = analysis_manager.run_analysis(
            analysis_type=analysis_type,
            attention_output=attention_output,
            image=processed_image,
            question=question,
            model_type="chartqa",  # ChartQA-specific parameter
            **kwargs
        )
        
        return {
            'analysis_type': analysis_result.analysis_type,
            'title': analysis_result.title,
            'description': analysis_result.description,
            'metrics': analysis_result.metrics,
            'insights': analysis_result.insights,
            'visualization_data': analysis_result.visualization.to_dict() if analysis_result.visualization else None,
            'raw_data': analysis_result.raw_data,
            'model_specific': 'chartqa_960'
        }
