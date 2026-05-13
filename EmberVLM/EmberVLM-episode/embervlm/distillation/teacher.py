"""
Teacher Model Wrapper for Distillation

Wraps larger VLM models for knowledge distillation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TeacherWrapper(nn.Module):
    """
    Wrapper for teacher VLM models used in distillation.

    Supports:
    - Qwen-VL-Chat
    - LLaVA
    - Other HuggingFace VLM models
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-chat",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit

        self.model = None
        self.processor = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load teacher model based on model_name."""
        logger.info(f"Loading teacher model: {self.model_name}")

        try:
            if "qwen" in self.model_name.lower():
                self._load_qwen_vl()
            elif "llava" in self.model_name.lower():
                self._load_llava()
            else:
                self._load_generic_vlm()
        except Exception as e:
            logger.warning(f"Failed to load teacher model: {e}")
            logger.warning("Using dummy teacher (returns zeros)")
            self.model = None

    def _load_qwen_vl(self):
        """Load Qwen-VL model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "Qwen/Qwen-VL-Chat"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                load_in_8bit=self.load_in_8bit,
            )

            self.model.eval()
            logger.info("Loaded Qwen-VL-Chat teacher model")

        except Exception as e:
            logger.warning(f"Failed to load Qwen-VL: {e}")
            raise

    def _load_llava(self):
        """Load LLaVA model."""
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor

            model_id = "llava-hf/llava-1.5-7b-hf"

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer = self.processor.tokenizer

            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=self.dtype,
                load_in_8bit=self.load_in_8bit,
            )

            self.model.eval()
            logger.info("Loaded LLaVA teacher model")

        except Exception as e:
            logger.warning(f"Failed to load LLaVA: {e}")
            raise

    def _load_generic_vlm(self):
        """Load generic VLM from HuggingFace."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=self.dtype,
            )

            self.model.eval()
            logger.info(f"Loaded generic VLM: {self.model_name}")

        except Exception as e:
            logger.warning(f"Failed to load generic VLM: {e}")
            raise

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Get teacher outputs for distillation.

        Args:
            pixel_values: Input images [B, C, H, W]
            input_ids: Token IDs [B, seq_len]
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return hidden states

        Returns:
            Dictionary with logits and optionally hidden states
        """
        if self.model is None:
            # Return dummy outputs
            batch_size, seq_len = input_ids.shape
            vocab_size = 50257  # Default

            return {
                'logits': torch.zeros(batch_size, seq_len, vocab_size, device=input_ids.device),
                'hidden_states': None,
            }

        try:
            # Prepare inputs for Idefics3 (SmolVLM) models
            # Idefics3 expects: pixel_values (B, num_images, C, H, W) and pixel_attention_mask (B, num_images, H, W)
            teacher_pixel_values = pixel_values
            pixel_attention_mask = None
            
            # Check if model is Idefics3 and needs special formatting
            is_idefics3 = hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type') and \
                         self.model.config.model_type == 'idefics3'
            
            if is_idefics3 and pixel_values.ndim == 4:
                # Add num_images dimension: (B, C, H, W) -> (B, 1, C, H, W)
                teacher_pixel_values = pixel_values.unsqueeze(1)
                # Create spatial attention mask: (B, num_images, H, W)
                batch_size = pixel_values.shape[0]
                height = pixel_values.shape[2]
                width = pixel_values.shape[3]
                pixel_attention_mask = torch.ones(
                    (batch_size, 1, height, width),
                    dtype=torch.long,
                    device=pixel_values.device
                )
            
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=teacher_pixel_values,
                attention_mask=attention_mask,
                pixel_attention_mask=pixel_attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            return {
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states if output_hidden_states else None,
            }

        except Exception as e:
            logger.warning(f"Teacher forward failed: {e}")
            batch_size, seq_len = input_ids.shape
            vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 50257

            return {
                'logits': torch.zeros(batch_size, seq_len, vocab_size, device=input_ids.device),
                'hidden_states': None,
            }

    @torch.no_grad()
    def generate_response(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate response from teacher model.

        Args:
            image: Input image (PIL Image or path)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response string
        """
        if self.model is None:
            return "Teacher model not available."

        try:
            if self.processor is not None:
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                ).to(self.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

                response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            else:
                # Qwen-style inference
                response = self.model.chat(
                    self.tokenizer,
                    query=prompt,
                    image=image,
                    history=None,
                )
                if isinstance(response, tuple):
                    response = response[0]

            return response

        except Exception as e:
            logger.warning(f"Teacher generation failed: {e}")
            return f"Generation failed: {str(e)}"

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.model is not None and hasattr(self.model, 'config'):
            return self.model.config.vocab_size
        return 50257

    def get_hidden_size(self) -> int:
        """Get hidden dimension."""
        if self.model is not None and hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        return 4096


class OfflineTeacher:
    """
    Offline teacher for pre-generating distillation data.

    Generates teacher outputs and saves them for efficient training.
    """

    def __init__(
        self,
        teacher: TeacherWrapper,
        output_dir: str,
    ):
        self.teacher = teacher
        self.output_dir = output_dir

        import os
        os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def generate_distillation_data(
        self,
        dataloader,
        save_logits: bool = True,
        save_hidden: bool = False,
    ):
        """
        Generate and save teacher outputs for entire dataset.

        Args:
            dataloader: Data loader
            save_logits: Whether to save logits
            save_hidden: Whether to save hidden states
        """
        import os
        from tqdm import tqdm

        logger.info(f"Generating distillation data to {self.output_dir}")

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            pixel_values = batch['pixel_values'].to(self.teacher.device)
            input_ids = batch['input_ids'].to(self.teacher.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.teacher.device)

            outputs = self.teacher(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=save_hidden,
            )

            # Save outputs
            batch_data = {
                'input_ids': input_ids.cpu(),
            }

            if save_logits and outputs['logits'] is not None:
                batch_data['teacher_logits'] = outputs['logits'].cpu()

            if save_hidden and outputs['hidden_states'] is not None:
                batch_data['teacher_hidden'] = outputs['hidden_states'][-1].cpu()

            output_path = os.path.join(self.output_dir, f'batch_{batch_idx:06d}.pt')
            torch.save(batch_data, output_path)

        logger.info(f"Generated distillation data for {batch_idx + 1} batches")

