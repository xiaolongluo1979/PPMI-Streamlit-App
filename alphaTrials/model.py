#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import keras

from tensorflow.keras.layers import MultiHeadAttention
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def get_angles(pos, i, d_model):
    """Calculate positional encoding angles."""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """Create positional encoding matrix."""
        # Use TensorFlow operations instead of NumPy for symbolic tensors
    position = tf.cast(position, tf.float32)
    d_model = tf.cast(d_model, tf.float32)
    
    # Create position and dimension indices
    pos_indices = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
    dim_indices = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    
    # Calculate angles
    angle_rads = pos_indices / tf.pow(10000.0, (2 * (dim_indices // 2)) / d_model)
    
    # Apply sin and cos
    angle_rads = tf.where(tf.equal(dim_indices % 2, 0), 
                          tf.sin(angle_rads), 
                          tf.cos(angle_rads))
    
    pos_encoding = angle_rads[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    """Positional embedding layer for transformer."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x):
        # For 3D input (batch, seq_len, features), we don't need padding
        # Just scale and add positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # Use tf.shape instead of x.shape for symbolic tensors
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos_enc = positional_encoding(seq_len, d_model)
        x = x + pos_enc
        return x

class BaseAttention(tf.keras.layers.Layer):
    """Base attention layer."""
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    """Cross attention layer for decoder."""
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class GlobalSelfAttention(BaseAttention):
    """Global self-attention layer."""
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):
    """Causal self-attention layer."""
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    """Feed-forward network layer."""
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

@keras.saving.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
    """Single encoder layer."""
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.layers.Layer):
    """Encoder for processing context (X_tensor)."""
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Add input projection layer to transform input features to d_model
        self.input_projection = tf.keras.layers.Dense(d_model)
        
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding and dropout
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        # Process through encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    """Single decoder layer."""
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x

@keras.saving.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    """Decoder for generating predictions."""
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Add input projection layer to transform y_history to d_model
        self.input_projection = tf.keras.layers.Dense(d_model)
        
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding and dropout
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        # Process through decoder layers
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x

@keras.saving.register_keras_serializable()
class PPMIDTransformer(tf.keras.Model):
    """Transformer model for PPMI longitudinal data prediction with confidence limits."""
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate, 
                 num_outcomes, sequence_length, n_mc_samples=50):
        super().__init__()
        # Store parameters for serialization
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.num_outcomes = num_outcomes
        self.sequence_length = sequence_length
        self.n_mc_samples = n_mc_samples
        
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(num_outcomes)

    def call(self, inputs, training=False):
        X_context, y_history = inputs

        if training:
            # Normal training pass
            enc_output = self.encoder(X_context, training=True)
            dec_output = self.decoder(y_history, enc_output, training=True)
            final_output = self.final_layer(dec_output)
            return final_output
        else:
            # Inference with MC dropout for confidence limits
            preds = []
            for _ in range(self.n_mc_samples):
                enc_output = self.encoder(X_context, training=True)   # Force dropout on
                dec_output = self.decoder(y_history, enc_output, training=True)
                preds.append(self.final_layer(dec_output))
            
            preds = tf.stack(preds, axis=0)  # [n_samples, batch, seq, num_outcomes]
            mean_pred = tf.reduce_mean(preds, axis=0)
            
            # Calculate percentiles using TensorFlow operations
            # Sort predictions along the sample dimension
            preds_sorted = tf.sort(preds, axis=0)
            n_samples = tf.shape(preds_sorted)[0]
            
            # Calculate percentile indices
            lower_idx = tf.cast(tf.round(0.025 * tf.cast(n_samples, tf.float32)), tf.int32)
            upper_idx = tf.cast(tf.round(0.975 * tf.cast(n_samples, tf.float32)), tf.int32)
            
            # Ensure indices are within bounds
            lower_idx = tf.maximum(0, lower_idx)
            upper_idx = tf.minimum(n_samples - 1, upper_idx)
            
            # Extract percentiles
            lower = preds_sorted[lower_idx]
            upper = preds_sorted[upper_idx]

            return mean_pred, lower, upper
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'num_outcomes': self.num_outcomes,
            'sequence_length': self.sequence_length,
            'n_mc_samples': self.n_mc_samples
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        # Extract the parameters we need
        num_layers = config.get('num_layers')
        d_model = config.get('d_model')
        num_heads = config.get('num_heads')
        dff = config.get('dff')
        dropout_rate = config.get('dropout_rate')
        num_outcomes = config.get('num_outcomes')
        sequence_length = config.get('sequence_length')
        n_mc_samples = config.get('n_mc_samples', 50)
        
        # Create and return the model
        return cls(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            num_outcomes=num_outcomes,
            sequence_length=sequence_length,
            n_mc_samples=n_mc_samples
        )

