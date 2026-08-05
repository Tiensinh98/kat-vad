"""Soft-prompt CLIP text encoder — port of the baseline's ``SoftPromptCLIPTextModel``.

Wraps a frozen HF ``CLIPTextModel`` and injects ``num_soft_prompts`` learnable
embeddings around the tokenized text: ``[BOS, pre-prompts, tokens, post-prompts,
EOS...]``. Only the prompt embedding table is trainable.

The CLIP model can be injected (``clip_model=``) so unit tests run on a tiny
random-weight config with no download; production code uses ``from_pretrained``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import CLIPTextModel
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from core import constants

# Sentinel written into expanded input ids at soft-prompt positions; must be
# lower than every real token id so EOS pooling via argmax is unaffected.
SOFT_PROMPT_SENTINEL_ID = -1
# HF CLIP quirk: checkpoints with the pre-#24773 (incorrect) eos_token_id pool
# at the highest-id token instead of the first true EOS.
LEGACY_CLIP_EOS_TOKEN_ID = 2


class SoftPromptCLIPTextModel(nn.Module):
    """Frozen CLIP text tower + trainable soft prompts (half pre, half post)."""

    def __init__(
        self,
        model_name_or_path: str = constants.CLIP_MODEL_NAME,
        num_soft_prompts: int = constants.NUM_SOFT_PROMPTS,
        clip_model: CLIPTextModel | None = None,
    ) -> None:
        super().__init__()
        self.num_soft_prompts = num_soft_prompts
        if clip_model is not None:
            self.model = clip_model
        elif model_name_or_path == constants.CLIP_MODEL_NAME:
            self.model = CLIPTextModel.from_pretrained(
                model_name_or_path,
                revision=constants.CLIP_MODEL_REVISION,
                use_safetensors=True,
            )
        else:  # non-default backbone: caller-controlled, unpinned
            self.model = CLIPTextModel.from_pretrained(model_name_or_path)  # nosec B615
        self.config = self.model.config
        self.prompt_embedding: nn.Embedding | None = (
            nn.Embedding(num_soft_prompts, self.config.hidden_size)
            if num_soft_prompts > 0
            else None
        )
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def max_text_tokens(self) -> int:
        """Token budget left for real text once soft prompts are inserted."""
        return constants.CLIP_MAX_TOKENS - self.num_soft_prompts

    def _insert_soft_prompts(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Splice pre/post prompt embeddings into the token embedding sequence.

        Returns ``(hidden_states, expanded_input_ids, expanded_attention_mask)``.
        Mirrors the baseline exactly: EOS is the highest token id, content length
        per row is ``argmax(input_ids) - 1``, and post-prompts overwrite the
        original EOS slot (a later padding EOS becomes the pooled position).
        """
        embeddings = self.model.text_model.embeddings
        if self.prompt_embedding is None:
            raise RuntimeError("_insert_soft_prompts called without prompt embeddings")
        batch, seq_len = input_ids.shape
        pre_num = self.num_soft_prompts // 2
        post_num = self.num_soft_prompts // 2
        content_lengths = input_ids.argmax(dim=-1) - 1  # (B,)

        soft_ids = torch.arange(self.num_soft_prompts, device=input_ids.device)
        soft_embed = self.prompt_embedding(soft_ids.unsqueeze(0).expand(batch, -1))

        expanded_ids = torch.full(
            (batch, self.num_soft_prompts + seq_len),
            int(input_ids.max()),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        expanded_ids[:, 0] = input_ids[:, 0]
        expanded_ids[:, 1 + pre_num : pre_num + seq_len] = input_ids[:, 1:]
        hidden_states = embeddings(input_ids=expanded_ids)
        hidden_states[:, 1 : 1 + pre_num] = soft_embed[:, :pre_num]
        expanded_ids[:, 1 : 1 + pre_num] = SOFT_PROMPT_SENTINEL_ID

        expanded_mask = attention_mask.new_zeros(expanded_ids.shape)
        for b in range(batch):
            post_start = 1 + pre_num + int(content_lengths[b])
            hidden_states[b, post_start : post_start + post_num] = soft_embed[b, -post_num:]
            expanded_ids[b, post_start : post_start + post_num] = SOFT_PROMPT_SENTINEL_ID
            expanded_mask[b, : int(expanded_ids[b].argmax()) + 1] = 1
        return hidden_states, expanded_ids, expanded_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        use_soft_prompt: bool = True,
    ) -> BaseModelOutputWithPooling:
        text_model = self.model.text_model

        if (
            self.prompt_embedding is None
            or self.num_soft_prompts == 0
            or not use_soft_prompt
        ):
            hidden_states = text_model.embeddings(
                input_ids=input_ids, position_ids=position_ids
            )
        else:
            if attention_mask is None:
                raise ValueError("attention_mask is required when using soft prompts")
            hidden_states, input_ids, attention_mask = self._insert_soft_prompts(
                input_ids, attention_mask
            )

        causal_attention_mask = _create_4d_causal_attention_mask(
            hidden_states.shape[:-1], hidden_states.dtype, device=hidden_states.device
        )
        use_flash_attention = getattr(text_model, "_use_flash_attention_2", False)
        if attention_mask is not None and not use_flash_attention:
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )

        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        last_hidden_state = text_model.final_layer_norm(encoder_outputs[0])

        batch_index = torch.arange(
            last_hidden_state.shape[0], device=last_hidden_state.device
        )
        if text_model.eos_token_id == LEGACY_CLIP_EOS_TOKEN_ID:
            eos_index = input_ids.to(
                dtype=torch.int, device=last_hidden_state.device
            ).argmax(dim=-1)
        else:
            eos_index = (
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                 == text_model.eos_token_id)
                .int()
                .argmax(dim=-1)
            )
        pooled_output = last_hidden_state[batch_index, eos_index]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


__all__ = ["SoftPromptCLIPTextModel"]
