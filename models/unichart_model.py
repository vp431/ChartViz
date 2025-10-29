"""
UniChart model implementation for chart QA with attention extraction.
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
    VisionEncoderDecoderModel, AutoConfig, DonutProcessor
)

from .base_model import BaseChartQAModel, ModelPrediction, AttentionOutput
from config import ModelConfig, APP_CONFIG

console = Console()


class UniChartModel(BaseChartQAModel):
    """UniChart model implementation with attention extraction capabilities."""
    
    def __init__(self, model_config: ModelConfig, device: Optional[str] = None):
        """Initialize UniChart model."""
        super().__init__(model_config, device)
        self.model_name = "unichart"
    
    def load_model(self) -> None:
        """Load UniChart model, processor, and tokenizer."""
        try:
            console.print(f"Loading UniChart from {self.config.local_dir}")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. UniChart requires GPU.")
            
            # Load processor (use DonutProcessor for official compatibility)
            try:
                self.processor = DonutProcessor.from_pretrained(
                    str(self.config.local_dir),
                    local_files_only=True
                )
                print("✓ Using DonutProcessor (official)")
            except Exception:
                # Fallback to AutoProcessor if DonutProcessor fails
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
            self.tokenizer = self.processor.tokenizer
            
            self.is_loaded = True
            console.print(f"✓ UniChart loaded successfully")
            
        except Exception as e:
            console.print(f"[red]Failed to load UniChart: {e}[/red]")
            raise RuntimeError(f"UniChart loading failed: {e}")
    
    def predict(self, image: Union[Image.Image, str], question: str) -> ModelPrediction:
        """
        Make a prediction using UniChart.
        
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
            # Use OFFICIAL UniChart format exactly as documented
            # Simple format works best - don't add extra context
            input_prompt = f"<chartqa> {question} <s_answer>"
            
            # Process image and decoder input separately (official method)
            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values
            
            # Process text as decoder input
            decoder_input_ids = self.processor.tokenizer(
                input_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            
            # Combine inputs
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
            
            # Generate answer using official UniChart parameters
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
            
            # Decode the generated tokens
            generated_ids = outputs.sequences
            
            # Use official UniChart decoding method
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
            confidence = self._calculate_confidence(outputs.scores) if outputs.scores else 0.5
            
            processing_time = time.time() - start_time
            
            return ModelPrediction(
                answer=answer,
                confidence=confidence,
                logits=outputs.scores[-1] if outputs.scores else None,
                processing_time=processing_time
            )
            
        except Exception as e:
            console.print(f"[red]Prediction failed: {e}[/red]")
            return ModelPrediction(
                answer="Error during prediction",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def extract_attention(self, image: Union[Image.Image, str], question: str) -> AttentionOutput:
        """
        Extract attention weights from UniChart for explainability.
        
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
            
            # For VisionEncoderDecoder, separate image and text processing for attention extraction
            # Process image for encoder
            pixel_values = self.processor.image_processor(
                images=image,
                return_tensors="pt"
            )["pixel_values"]
            
            # Process text for decoder input
            decoder_input_ids = self.processor.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length,
                add_special_tokens=True
            )["input_ids"]
            
            # Move inputs to device and ensure correct dtype
            pixel_values = self._safe_dtype_conversion(pixel_values).to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
            
            console.print(f"[blue]🔍 UniChart: Extracting attention for question: {question[:50]}...[/blue]")
            console.print(f"[blue]Pixel values shape: {pixel_values.shape}[/blue]")
            console.print(f"[blue]Decoder input shape: {decoder_input_ids.shape}[/blue]")
            
            # Forward pass with attention extraction - use generate() for cross-attention
            with torch.no_grad():
                # For VisionEncoderDecoder models, we need to use generate() to get cross-attention
                console.print(f"[blue]🔧 Using generate() for VisionEncoderDecoder cross-attention[/blue]")
                
                gen_outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=5,  # Generate a few tokens to extract attention
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
                
                # Create outputs object
                outputs = type('GenerationOutputs', (), {
                    'encoder_attentions': encoder_attentions,
                    'decoder_attentions': decoder_attentions,
                    'cross_attentions': cross_attentions,
                })()
            
            # Debug: Check what attention outputs we get
            console.print(f"[blue]🔍 Model outputs attributes: {dir(outputs)}[/blue]")
            if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions:
                console.print(f"[blue]🔍 Encoder attentions: {len(outputs.encoder_attentions)} layers[/blue]")
            if hasattr(outputs, 'decoder_attentions') and outputs.decoder_attentions:
                console.print(f"[blue]🔍 Decoder attentions: {len(outputs.decoder_attentions)} layers[/blue]")
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                console.print(f"[blue]🔍 Cross attentions: {len(outputs.cross_attentions)} layers[/blue]")
            else:
                console.print(f"[yellow]⚠️ No cross_attentions found in outputs[/yellow]")
            
            # Create inputs dict for processing
            inputs = {
                'pixel_values': pixel_values,
                'decoder_input_ids': decoder_input_ids,
                'input_ids': decoder_input_ids  # For backward compatibility
            }
            
            # Extract attention weights
            attention_output = self._process_attention_outputs(
                outputs, inputs, image, question
            )
            
            # Generate prediction for context
            prediction = self.predict(image, question)
            attention_output.predicted_answer = prediction.answer
            attention_output.confidence_score = prediction.confidence
            
            return attention_output
            
        except Exception as e:
            console.print(f"[red]Attention extraction failed: {e}[/red]")
            raise RuntimeError(f"Attention extraction failed: {e}")
        
        finally:
            self._clear_attention_hooks()
    
    def _process_attention_outputs(self, outputs, inputs, image: Image.Image, question: str) -> AttentionOutput:
        """Process model outputs to extract attention information."""
        
        # Get attention weights from different components
        encoder_attentions = outputs.encoder_attentions if hasattr(outputs, 'encoder_attentions') else None
        decoder_attentions = outputs.decoder_attentions if hasattr(outputs, 'decoder_attentions') else None
        cross_attentions = outputs.cross_attentions if hasattr(outputs, 'cross_attentions') else None
        
        console.print(f"[blue]🔍 Processing attention outputs...[/blue]")
        console.print(f"[blue]Encoder attentions: {encoder_attentions is not None}[/blue]")
        console.print(f"[blue]Decoder attentions: {decoder_attentions is not None}[/blue]")
        console.print(f"[blue]Cross attentions: {cross_attentions is not None}[/blue]")
        
        # Process text tokens
        if 'input_ids' in inputs:
            text_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        else:
            raise RuntimeError("Could not extract text tokens from inputs")
        
        console.print(f"[blue]Text tokens: {len(text_tokens)}[/blue]")
        
        # Get image patch coordinates
        image_patches = self.get_image_patches(image)
        console.print(f"[blue]Image patches: {len(image_patches)}[/blue]")
        
        # Process cross-modal attention (text to image)
        cross_attention = None
        
        # Method 1: Try cross_attentions (VisionEncoderDecoder models)
        if cross_attentions and len(cross_attentions) > 0:
            console.print(f"[green]✓ Found cross_attentions: {len(cross_attentions)} layers[/green]")
            try:
                # Use the last layer's cross attention
                last_layer_cross_attn = cross_attentions[-1]  # [batch, heads, seq_len, encoder_seq_len]
                
                if last_layer_cross_attn is not None:
                    console.print(f"[blue]Cross attention shape: {last_layer_cross_attn.shape}[/blue]")
                    console.print(f"[blue]Cross attention type: {type(last_layer_cross_attn)}[/blue]")
                    
                    # Check if it's actually a tensor with data
                    if torch.is_tensor(last_layer_cross_attn) and last_layer_cross_attn.numel() > 0:
                        # Average across heads for simplicity
                        cross_attention = last_layer_cross_attn.mean(dim=1)  # [batch, seq_len, encoder_seq_len]
                        cross_attention = cross_attention[0]  # Remove batch dimension
                        
                        # Verify we have meaningful data
                        if cross_attention.max().item() > 0:
                            console.print(f"[green]✓ Extracted real cross-attention: {cross_attention.shape}[/green]")
                            console.print(f"[green]  Stats: min={cross_attention.min():.4f}, max={cross_attention.max():.4f}, mean={cross_attention.mean():.4f}[/green]")
                        else:
                            console.print(f"[yellow]⚠️ Cross-attention all zeros[/yellow]")
                            cross_attention = None
                    else:
                        console.print(f"[yellow]⚠️ Cross-attention tensor is empty or invalid[/yellow]")
                else:
                    console.print(f"[yellow]⚠️ Cross-attention is None[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Cross attention extraction failed: {e}[/yellow]")
                import traceback
                traceback.print_exc()
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
        
        # If still no cross-attention, provide clear error message
        if cross_attention is None:
            console.print(f"[yellow]⚠️ No cross-attention available - some visualizations may not work[/yellow]")
        
        # Process self-attention within text
        text_self_attention = None
        if decoder_attentions and len(decoder_attentions) > 0:
            # Use the last layer's self attention
            last_layer_self_attn = decoder_attentions[-1]  # [batch, heads, seq_len, seq_len]
            if last_layer_self_attn is not None:
                text_self_attention = last_layer_self_attn.mean(dim=1)[0]  # Average heads, remove batch
        
        # Process image self-attention (from encoder)
        image_self_attention = None
        if encoder_attentions and len(encoder_attentions) > 0:
            # Use the last layer's encoder self attention
            last_layer_encoder_attn = encoder_attentions[-1]  # [batch, heads, seq_len, seq_len]
            if last_layer_encoder_attn is not None:
                image_self_attention = last_layer_encoder_attn.mean(dim=1)[0]  # Average heads, remove batch
        
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
        """
        Calculate confidence score from generation scores.
        
        Args:
            scores: Tuple of generation scores for each timestep
            
        Returns:
            Confidence score between 0 and 1
        """
        if not scores:
            return 0.5
        
        # Use the maximum probability across all generation steps
        max_probs = []
        for score in scores:
            probs = F.softmax(score, dim=-1)
            max_prob = torch.max(probs).item()
            max_probs.append(max_prob)
        
        # Average the maximum probabilities
        avg_confidence = sum(max_probs) / len(max_probs)
        return float(avg_confidence)
    
    def _is_attention_module(self, name: str, module: torch.nn.Module) -> bool:
        """
        Determine if a module is an attention module for UniChart/Pix2Struct.
        
        Args:
            name: Module name
            module: PyTorch module
            
        Returns:
            True if this module should be hooked for attention extraction
        """
        # Pix2Struct attention module patterns
        attention_patterns = [
            "attention",
            "self_attn", 
            "cross_attn",
            "multihead_attn"
        ]
        
        return any(pattern in name.lower() for pattern in attention_patterns)
    
    def get_image_patches(self, image: Image.Image) -> List[Tuple[int, int]]:
        """
        Get the coordinates of image patches for UniChart attention visualization.
        
        Args:
            image: Input image
            
        Returns:
            List of (x, y) coordinates for each patch
        """
        try:
            # UniChart uses 960x960 image size with patch-based vision encoder
            # Use config values instead of hardcoded ones
            vision_size = self.config.max_image_size[0]  # Should be 960
            patch_size = self.config.patch_size[0]       # Should be 32 from config
            
            patches_per_side = vision_size // patch_size  # 30 patches per side (960/32)
            total_patches = patches_per_side * patches_per_side  # 900 patches
            
            console.print(f"[blue]🔧 UniChart patches: {patches_per_side}x{patches_per_side} = {total_patches}[/blue]")
            
            # For visualization, we'll use a smaller grid to make it manageable
            # Reduce to ~24x24 = 576 patches for compatibility with visualization
            visual_patches_per_side = 24
            
            patches = []
            img_w, img_h = image.size
            
            for y in range(visual_patches_per_side):
                for x in range(visual_patches_per_side):
                    # Map to actual image coordinates
                    actual_x = int((x / visual_patches_per_side) * img_w)
                    actual_y = int((y / visual_patches_per_side) * img_h)
                    patches.append((actual_x, actual_y))
            
            console.print(f"[green]✓ Generated {len(patches)} patch coordinates for visualization[/green]")
            return patches
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate patch coordinates: {e}[/yellow]")
            raise RuntimeError(f"Could not generate patch coordinates: {e}")

    def get_supported_features(self) -> List[str]:
        """Get list of supported explainability features."""
        return [
            "cross_modal_attention",
            "text_self_attention", 
            "image_self_attention",
            "token_importance",
            "patch_attention",
            "confidence_estimation",
            "advanced_statistical_analysis",
            "attention_flow_analysis",
            "layer_wise_analysis",
            "multi_head_analysis"
        ]
    
    def run_advanced_analysis(self, analysis_type: str, image: Union[Image.Image, str], 
                            question: str, **kwargs) -> Dict[str, Any]:
        """
        Run advanced analysis on the model's attention patterns.
        
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
        
        # Run the requested analysis
        analysis_result = analysis_manager.run_analysis(
            analysis_type=analysis_type,
            attention_output=attention_output,
            image=processed_image,
            question=question,
            **kwargs
        )
        
        return {
            'analysis_type': analysis_result.analysis_type,
            'title': analysis_result.title,
            'description': analysis_result.description,
            'metrics': analysis_result.metrics,
            'insights': analysis_result.insights,
            'visualization_data': analysis_result.visualization.to_dict() if analysis_result.visualization else None,
            'raw_data': analysis_result.raw_data
        }
    
    def get_attention_statistics(self, image: Union[Image.Image, str], question: str) -> Dict[str, float]:
        """
        Get statistical metrics about attention patterns.
        
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
        cross_attn = attention_output.cross_attention.detach().cpu().numpy()
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
            'total_attention': float(np.sum(flat_attn))
        }
    
    def analyze_token_importance(self, image: Union[Image.Image, str], question: str) -> Dict[str, Any]:
        """
        Analyze the importance of different tokens in the question.
        
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
        cross_attn = attention_output.cross_attention.detach().cpu().numpy()
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
            
            # Find question words vs other tokens
            question_words = ['what', 'where', 'when', 'why', 'how', 'which', 'who']
            question_token_indices = [i for i, token in enumerate(tokens) 
                                    if token.lower().strip('.,!?') in question_words]
            
            if question_token_indices:
                avg_question_importance = np.mean([importance_scores[i] for i in question_token_indices])
                avg_other_importance = np.mean([importance_scores[i] for i in range(len(tokens)) 
                                              if i not in question_token_indices])
                
                if avg_question_importance > avg_other_importance:
                    insights.append("Question words receive higher attention than other tokens")
                else:
                    insights.append("Content words receive higher attention than question words")
        
        return {
            'tokens': tokens,
            'importance_scores': importance_scores,
            'insights': insights,
            'total_tokens': len(tokens),
            'attention_distribution': 'concentrated' if np.std(importance_scores) > 0.1 else 'distributed'
        }
