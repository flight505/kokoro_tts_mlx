"""ALBERT-based attention and transformer layers."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AlbertConfig:
    """Configuration for ALBERT model."""

    # Required fields (from model config)
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    num_hidden_layers: int = 12

    # Optional fields with defaults
    vocab_size: int = 178
    embedding_size: int = 128
    num_hidden_groups: int = 1
    inner_group_num: int = 1
    hidden_act: str = "gelu_new"
    dropout: float = 0.1  # General dropout
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


class AlbertEmbeddings(nn.Module):
    """ALBERT embeddings: word + position + token_type."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        seq_length = input_ids.shape[1]

        if position_ids is None:
            position_ids = mx.arange(seq_length)[None, :]

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)

        return embeddings


class AlbertAttention(nn.Module):
    """Multi-head self-attention for ALBERT."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape

        # Linear projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape for multi-head attention
        def reshape_for_scores(x):
            return x.reshape(
                batch_size, seq_length, self.num_attention_heads, self.attention_head_size
            ).transpose(0, 2, 1, 3)

        query_layer = reshape_for_scores(query_layer)
        key_layer = reshape_for_scores(key_layer)
        value_layer = reshape_for_scores(value_layer)

        # Attention scores
        attention_scores = query_layer @ key_layer.transpose(0, 1, 3, 2)
        attention_scores = attention_scores / mx.sqrt(
            mx.array(self.attention_head_size, dtype=attention_scores.dtype)
        )

        # Apply attention mask
        if attention_mask is not None:
            # Expand mask for broadcasting: (batch, 1, 1, seq_len)
            attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + (1.0 - attention_mask) * -1e9

        attention_probs = mx.softmax(attention_scores, axis=-1)

        # Context
        context_layer = attention_probs @ value_layer
        context_layer = context_layer.transpose(0, 2, 1, 3)
        context_layer = context_layer.reshape(batch_size, seq_length, self.hidden_size)

        # Output projection with residual
        projected = self.dense(context_layer)
        output = self.LayerNorm(projected + hidden_states)

        return output


class AlbertIntermediate(nn.Module):
    """Feed-forward intermediate layer."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return hidden_states


class AlbertOutput(nn.Module):
    """Feed-forward output layer with residual."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self, hidden_states: mx.array, input_tensor: mx.array
    ) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlbertLayer(nn.Module):
    """Single ALBERT layer with attention and FFN."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.attention = AlbertAttention(config)
        self.intermediate = AlbertIntermediate(config)
        self.output = AlbertOutput(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class AlbertLayerGroup(nn.Module):
    """Group of ALBERT layers (for parameter sharing)."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.albert_layers = [
            AlbertLayer(config) for _ in range(config.inner_group_num)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        for layer in self.albert_layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class AlbertEncoder(nn.Module):
    """ALBERT encoder with embedding projection and layer groups."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size, config.hidden_size
        )
        self.albert_layer_groups = [
            AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        for i in range(self.config.num_hidden_layers):
            group_idx = i % self.config.num_hidden_groups
            hidden_states = self.albert_layer_groups[group_idx](
                hidden_states, attention_mask
            )

        return hidden_states


class CustomAlbert(nn.Module):
    """Full ALBERT model for Kokoro."""

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass.

        Returns:
            sequence_output: (batch, seq_len, hidden_size)
            pooled_output: (batch, hidden_size)
        """
        embedding_output = self.embeddings(
            input_ids, token_type_ids=token_type_ids
        )
        sequence_output = self.encoder(embedding_output, attention_mask)

        # Pool using first token
        pooled_output = self.pooler(sequence_output[:, 0])
        pooled_output = mx.tanh(pooled_output)

        return sequence_output, pooled_output
