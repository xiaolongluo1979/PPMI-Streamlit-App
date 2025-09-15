#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay


import tensorflow as tf
import keras

from tensorflow.keras.layers import MultiHeadAttention
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import joblib
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from config import FITTED_MODELS_PATH, DATA_PATH, MODEL_FILES, DATA_FILES
except ImportError:
    # Fallback to relative paths if config not available
    FITTED_MODELS_PATH = "./fittedModels"
    DATA_PATH = "./data"
    MODEL_FILES = {
        'transformer_model': 'ppmi_transformer_model_final.keras',
        'baseline_scaler': 'baseline_Xnum_scaler.pkl',
        'dfy_scaler': 'dfy_scaler.pkl',
        'ynum_means': 'ynum_means.pkl',
        'mmrm_models': {
            'NP1PTOT': 'mmrm_model_NP1PTOT.pkl',
            'NP2PTOT': 'mmrm_model_NP2PTOT.pkl',
            'NP3TOT': 'mmrm_model_NP3TOT.pkl'
        }
    }
    DATA_FILES = {
        'complete_data': 'PPMI_complete_with_imputation.csv'
    }




def convert_features_to_tensor(df, feature_cols, default_value=0.0):
    """
    Convert features from DataFrame to TensorFlow tensor.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset with all columns
    feature_cols : list
        List of feature column names
    default_value : float
        Default value for missing entries (default: 0.0)
        
    Returns:
    --------
    X_tensor : tensorflow.Tensor
        Tensor of features with shape (n_patients, n_timepoints, n_features)
    """
    import tensorflow as tf
    
    print(f"\n=== Converting Features to Tensor ===")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    
    # Group by PATNO and create ragged tensors
    ragged_tensors = df.groupby('PATNO')[feature_cols].apply(
        lambda x: tf.ragged.constant(x.values)
    ).values
    
    # Stack ragged tensors
    ragged_tensor = tf.ragged.stack(ragged_tensors.tolist(), axis=0)
    
    print(f"Ragged tensor shape: {ragged_tensor.shape}")
    print(f"Bounding shape: {ragged_tensor.bounding_shape()}")
    
    # Convert to regular tensor with specified shape
    X_tensor = ragged_tensor.to_tensor(
        default_value=default_value, 
        shape=[
            ragged_tensor.shape[0], 
            ragged_tensor.bounding_shape(axis=1),
            ragged_tensor.shape[2]
        ]
    )
    
    # Cast to float32
    X_tensor = tf.cast(X_tensor, dtype=tf.float32)
    
    print(f"Final tensor shape: {X_tensor.shape}")
    print(f"Tensor dtype: {X_tensor.dtype}")
    print(f"Non-zero elements: {tf.reduce_sum(tf.cast(tf.not_equal(X_tensor, default_value), tf.int32))}")
    
    return X_tensor

def convert_outcomes_to_tensor(df, outcome_cols, default_value=0.0):
    """
    Convert outcome variables from DataFrame to TensorFlow tensor.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset with all columns
    outcome_cols : list
        List of outcome column names (e.g., ['NP1PTOT_imputed', 'NP2PTOT_imputed', 'NP3TOT_imputed'])
    default_value : float
        Default value for missing entries (default: 0.0)
        
    Returns:
    --------
    y_tensor : tensorflow.Tensor
        Tensor of outcomes with shape (n_patients, n_timepoints, n_outcomes)
    """
    import tensorflow as tf
    
    print(f"\n=== Converting Outcomes to Tensor ===")
    print(f"Outcome columns: {len(outcome_cols)}")
    print(f"Outcome columns: {outcome_cols}")
    
    # Group by PATNO and create ragged tensors
    ragged_tensors = df.groupby('PATNO')[outcome_cols].apply(
        lambda x: tf.ragged.constant(x.values)
    ).values
    
    # Stack ragged tensors
    ragged_tensor = tf.ragged.stack(ragged_tensors.tolist(), axis=0)
    
    print(f"Ragged tensor shape: {ragged_tensor.shape}")
    print(f"Bounding shape: {ragged_tensor.bounding_shape()}")
    
    # Convert to regular tensor with specified shape
    y_tensor = ragged_tensor.to_tensor(
        default_value=default_value, 
        shape=[
            ragged_tensor.shape[0], 
            ragged_tensor.bounding_shape(axis=1),
            ragged_tensor.shape[2]
        ]
    )
    
    # Cast to float32
    y_tensor = tf.cast(y_tensor, dtype=tf.float32)
    
    print(f"Final tensor shape: {y_tensor.shape}")
    print(f"Tensor dtype: {y_tensor.dtype}")
    print(f"Non-zero elements: {tf.reduce_sum(tf.cast(tf.not_equal(y_tensor, default_value), tf.int32))}")
    
    return y_tensor

def convert_imputation_status_to_tensor(df, status_cols, default_value=False):
    """
    Convert imputation status from DataFrame to TensorFlow tensor.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset with all columns
    status_cols : list
        List of imputation status column names (e.g., ['NP1PTOT_is_nan', 'NP2PTOT_is_nan', 'NP3TOT_is_nan'])
    default_value : bool
        Default value for missing entries (default: False)
        
    Returns:
    --------
    mask_tensor : tensorflow.Tensor
        Tensor of imputation status with shape (n_patients, n_timepoints, n_outcomes)
    """
    import tensorflow as tf
    
    print(f"\n=== Converting Imputation Status to Tensor ===")
    print(f"Status columns: {len(status_cols)}")
    print(f"Status columns: {status_cols}")
    
    # Group by PATNO and create ragged tensors
    ragged_tensors = df.groupby('PATNO')[status_cols].apply(
        lambda x: tf.ragged.constant(x.values)
    ).values
    
    # Stack ragged tensors
    ragged_tensor = tf.ragged.stack(ragged_tensors.tolist(), axis=0)
    
    print(f"Ragged tensor shape: {ragged_tensor.shape}")
    print(f"Bounding shape: {ragged_tensor.bounding_shape()}")
    
    # Convert to regular tensor with specified shape
    mask_tensor = ragged_tensor.to_tensor(
        default_value=default_value, 
        shape=[
            ragged_tensor.shape[0], 
            ragged_tensor.bounding_shape(axis=1),
            ragged_tensor.shape[2]
        ]
    )
    
    # Cast to boolean
    mask_tensor = tf.cast(mask_tensor, dtype=tf.bool)
    
    print(f"Final tensor shape: {mask_tensor.shape}")
    print(f"Tensor dtype: {mask_tensor.dtype}")
    print(f"True values (imputed): {tf.reduce_sum(tf.cast(mask_tensor, tf.int32))}")
    print(f"False values (original): {tf.reduce_sum(tf.cast(tf.logical_not(mask_tensor), tf.int32))}")
    print(f"Imputation rate: {tf.reduce_mean(tf.cast(mask_tensor, tf.float32))*100:.1f}%")
    
    return mask_tensor

def create_weight_tensor(df, cohort_col='COHORT_DEFINITION', ps_cols=None, default_weight=1.0):
    """
    Create a weight tensor based on cohort definition and propensity scores.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset with all columns
    cohort_col : str
        Name of the cohort definition column (default: 'COHORT_DEFINITION')
    ps_cols : list
        List of propensity score column names. If None, uses default PPMI columns.
        (default: ['PS_Healthy_Control', 'PS_Parkinsons_Disease', 'PS_Prodromal', 'PS_SWEDD'])
    default_weight : float
        Default weight for missing or invalid entries (default: 1.0)
        
    Returns:
    --------
    weight_tensor : tensorflow.Tensor
        Weight tensor with shape (n_patients, n_timepoints, 1)
    """
    import tensorflow as tf
    
    # Set default propensity score columns if not provided
    if ps_cols is None:
        ps_cols = ['PS_Healthy_Control', 'PS_Parkinsons_Disease', 'PS_Prodromal', 'PS_SWEDD']
    
    print(f"\n=== Creating Weight Tensor ===")
    print(f"Cohort column: {cohort_col}")
    print(f"Propensity score columns: {ps_cols}")
    
    # Get unique PATNOs and sort them
    patno_list = sorted(df['PATNO'].unique())
    n_patients = len(patno_list)
    
    # Get unique time values and sort them
    time_list = sorted(df['year'].unique())
    n_timepoints = len(time_list)
    
    print(f"Number of patients: {n_patients}")
    print(f"Number of timepoints: {n_timepoints}")
    
    # Initialize weight tensor
    weight_tensor = tf.zeros((n_patients, n_timepoints, 1), dtype=tf.float32)
    
    # Create cohort to PS column mapping
    cohort_ps_mapping = {
        'Healthy Control': 'PS_Healthy_Control',
        "Parkinson's Disease": 'PS_Parkinsons_Disease',
        'Prodromal': 'PS_Prodromal',
        'SWEDD': 'PS_SWEDD'
    }
    
    print(f"Cohort to PS mapping: {cohort_ps_mapping}")
    
    # Create a copy of the dataframe and add a weight column
    df_weight = df.copy()
    df_weight['weight'] = default_weight
    
    # Efficiently assign weights based on cohort
    for cohort, ps_col in cohort_ps_mapping.items():
        if ps_col in df_weight.columns:
            mask = df_weight[cohort_col] == cohort
            df_weight.loc[mask, 'weight'] = df_weight.loc[mask, ps_col].fillna(default_weight)
    
    # Handle invalid weights (negative or NaN)
    df_weight['weight'] = df_weight['weight'].apply(
        lambda x: default_weight if pd.isna(x) or x < 0 else x
    )
    
    # Group by PATNO and create ragged tensors for weights
    ragged_tensors = df_weight.groupby('PATNO')['weight'].apply(
        lambda x: tf.ragged.constant(x.values)
    ).values
    
    # Stack ragged tensors
    ragged_tensor = tf.ragged.stack(ragged_tensors.tolist(), axis=0)
    
    print(f"Ragged tensor shape: {ragged_tensor.shape}")
    print(f"Bounding shape: {ragged_tensor.bounding_shape()}")
    
    # Convert to regular tensor with specified shape
    weight_tensor = ragged_tensor.to_tensor(
        default_value=default_weight, 
        shape=[
            ragged_tensor.shape[0], 
            ragged_tensor.bounding_shape(axis=1)
        ]
    )
    
    # Add the third dimension (1) for weights
    weight_tensor = tf.expand_dims(weight_tensor, axis=-1)
    
    # Cast to float32
    weight_tensor = tf.cast(weight_tensor, dtype=tf.float32)
    
    # Print summary statistics
    print(f"\n=== Weight Tensor Summary ===")
    print(f"Weight tensor shape: {weight_tensor.shape}")
    print(f"Weight tensor dtype: {weight_tensor.dtype}")
    print(f"Non-zero weights: {tf.reduce_sum(tf.cast(tf.not_equal(weight_tensor, 0), tf.int32))}")
    print(f"Weight range: [{tf.reduce_min(weight_tensor):.4f}, {tf.reduce_max(weight_tensor):.4f}]")
    print(f"Mean weight: {tf.reduce_mean(weight_tensor):.4f}")
    print(f"Standard deviation: {tf.math.reduce_std(weight_tensor):.4f}")
    
    # Print cohort-specific statistics
    print(f"\nCohort-specific weight statistics:")
    for cohort, ps_col in cohort_ps_mapping.items():
        if ps_col in df.columns:
            cohort_weights = df[df[cohort_col] == cohort][ps_col]
            valid_weights = cohort_weights[cohort_weights >= 0]
            if len(valid_weights) > 0:
                print(f"  {cohort}: mean={valid_weights.mean():.4f}, std={valid_weights.std():.4f}, count={len(valid_weights)}")
    
    return weight_tensor

def create_tf_dataset_and_split(X_tensor, y_tensor, mask_tensor, weight_tensor, train_ratio=0.8, val_ratio=0.2, shuffle_buffer_size=None):
    """
    Create TensorFlow dataset and split into training and validation parts.
    Includes y history as part of the input for next-step prediction.
    
    Parameters:
    -----------
    X_tensor : tensorflow.Tensor
        Feature tensor with shape (n_patients, n_timepoints, n_features)
    y_tensor : tensorflow.Tensor
        Outcome tensor with shape (n_patients, n_timepoints, n_outcomes)
    mask_tensor : tensorflow.Tensor
        Mask tensor with shape (n_patients, n_timepoints, n_outcomes)
    weight_tensor : tensorflow.Tensor
        Weight tensor with shape (n_patients, n_timepoints, 1)
    train_ratio : float
        Ratio for training set (default: 0.8)
    val_ratio : float
        Ratio for validation set (default: 0.2)
    shuffle_buffer_size : int
        Buffer size for shuffling. If None, uses dataset size.
        
    Returns:
    --------
    train_dataset : tf.data.Dataset
        Training dataset with 5 tensors: (X_context, y_history, y_target, mask_target, weight_target)
    val_dataset : tf.data.Dataset
        Validation dataset with 5 tensors: (X_context, y_history, y_target, mask_target, weight_target)
    dataset_info : dict
        Information about the dataset splits
    """
    import tensorflow as tf
    
    print(f"\n=== Creating TensorFlow Dataset and Splits ===")
    print(f"Original tensor shapes:")
    print(f"  X_tensor: {X_tensor.shape}")
    print(f"  y_tensor: {y_tensor.shape}")
    print(f"  mask_tensor: {mask_tensor.shape}")
    print(f"  weight_tensor: {weight_tensor.shape}")
    
    print(f"\nDataset tensor shapes (transformer input/output structure):")
    print(f"  X_context: {X_tensor[:, :-1, :].shape}")
    print(f"  y_history: {y_tensor[:, :-1, :].shape}")
    print(f"  y_target: {y_tensor[:, 1:, :].shape}")
    print(f"  mask_target: {mask_tensor[:, 1:, :].shape}")
    print(f"  weight_target: {weight_tensor[:, 1:, :].shape}")
    
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"⚠️  Warning: Ratios sum to {total_ratio}, not 1.0")
    
    # Create dataset from tensor slices for transformer model
    # Input tensors: (X_context, y_history) - for transformer encoder/decoder
    # Target tensors: (y_target, mask_target, weight_target) - for loss computation
    dataset = tf.data.Dataset.from_tensor_slices((
        X_tensor[:, :-1, :],      # X_context (t=0 to t=T-1) - transformer input
        y_tensor[:, :-1, :],      # y_history (t=0 to t=T-1) - transformer input
        y_tensor[:, 1:, :],       # y_target (t=1 to t=T) - transformer output target
        mask_tensor[:, 1:, :],    # mask_target (t=1 to t=T) - for loss weighting
        weight_tensor[:, 1:, :],  # weight_target (t=1 to t=T) - for loss weighting
    ))
    
    # Get dataset size
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    print(f"Dataset size: {dataset_size}")
    
    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    remaining_size = dataset_size - train_size - val_size
    
    print(f"Split sizes:")
    print(f"  Training: {train_size} ({train_size/dataset_size*100:.1f}%)")
    print(f"  Validation: {val_size} ({val_size/dataset_size*100:.1f}%)")
    if remaining_size > 0:
        print(f"  Remaining: {remaining_size} ({remaining_size/dataset_size*100:.1f}%)")
    
    # Set shuffle buffer size if not provided
    if shuffle_buffer_size is None:
        shuffle_buffer_size = dataset_size
        print(f"Using shuffle buffer size: {shuffle_buffer_size}")
    
    # Shuffle the dataset
    print(f"Shuffling dataset with buffer size: {shuffle_buffer_size}")
    dataset_shuffled = dataset.shuffle(
        buffer_size=shuffle_buffer_size, 
        reshuffle_each_iteration=False,
        seed=42  # For reproducibility
    )
    
    # Create the splits
    train_dataset = dataset_shuffled.take(train_size)
    val_dataset = dataset_shuffled.skip(train_size).take(val_size)
    
    # Verify split sizes
    train_actual_size = tf.data.experimental.cardinality(train_dataset).numpy()
    val_actual_size = tf.data.experimental.cardinality(val_dataset).numpy()
    
    print(f"\nActual split sizes:")
    print(f"  Training: {train_actual_size}")
    print(f"  Validation: {val_actual_size}")
    
    # Create dataset info
    dataset_info = {
        'total_size': dataset_size,
        'train_size': train_actual_size,
        'val_size': val_actual_size,
        'train_ratio': train_actual_size / dataset_size,
        'val_ratio': val_actual_size / dataset_size,
        'shuffle_buffer_size': shuffle_buffer_size,
        'tensor_shapes': {
            'X_context': X_tensor[:, :-1, :].shape,
            'y_history': y_tensor[:, :-1, :].shape,
            'y_target': y_tensor[:, 1:, :].shape,
            'mask_target': mask_tensor[:, 1:, :].shape,
            'weight_target': weight_tensor[:, 1:, :].shape
        }
    }
    
    print(f"\n=== Dataset Creation Complete ===")
    print(f"Training dataset ready: {train_actual_size} samples")
    print(f"Validation dataset ready: {val_actual_size} samples")
    
    return train_dataset, val_dataset, dataset_info


def impute_y_tensor_with_model(X_tensor, y_tensor, mask_tensor, weight_tensor, model, use_confidence_intervals=True):
    """
    Impute y_tensor using a given PPMIDTransformer model for values indicated by mask_tensor.
    Returns three tensors: mean predictions, lower limits, and upper limits.
    
    Args:
        X_tensor: Input features tensor (n_patients, n_timepoints, n_features)
        y_tensor: Target outcomes tensor (n_patients, n_timepoints, n_outcomes)
        mask_tensor: Boolean mask indicating which values to impute (n_patients, n_timepoints, n_outcomes)
        weight_tensor: Weight tensor for propensity score weighting (n_patients, n_timepoints, 1)
        model: Trained PPMIDTransformer model
        use_confidence_intervals: Whether to use MC dropout for confidence intervals (default: True)
    
    Returns:
        Tuple of (X_tensor, y_tensor_imputed, mask_tensor, weight_tensor, lower_limits, upper_limits)
    """
    import tensorflow as tf
    
    print(f"\n=== Imputing y_tensor with PPMIDTransformer Model ===")
    print(f"Input tensor shapes:")
    print(f"  X_tensor: {X_tensor.shape}")
    print(f"  y_tensor: {y_tensor.shape}")
    print(f"  mask_tensor: {mask_tensor.shape}")
    
    y_tensor_imputed = tf.identity(y_tensor)
    n_patients, n_timepoints, n_outcomes = y_tensor.shape
    
    imputed_values = tf.reduce_sum(tf.cast(mask_tensor, tf.int32))
    print(f"Imputed values to replace: {imputed_values.numpy()}")
    
    X_context = X_tensor[:, :-1, :]
    y_history = y_tensor_imputed[:, :-1, :]
    mask_target = mask_tensor[:, 1:, :]
    
    print(f"Model input shapes:")
    print(f"  X_context: {X_context.shape}")
    print(f"  y_history: {y_history.shape}")
    print(f"  mask_target: {mask_target.shape}")
    
    # Always use training=False to get confidence intervals
    print(f"Using confidence intervals mode with training=False...")
    model_output = model((X_context, y_history), training=False)
    print(f"Model output type: {type(model_output)}")
    print(f"Model output length: {len(model_output) if hasattr(model_output, '__len__') else 'Not iterable'}")
    
    if len(model_output) == 3:
        mean_pred, lower, upper = model_output
    else:
        print(f"ERROR: Model returned {len(model_output)} values, expected 3")
        print(f"Model output: {model_output}")
        raise ValueError(f"Model returned {len(model_output)} values, expected 3")
    
    predictions = mean_pred
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Lower limits shape: {lower.shape}")
    print(f"Upper limits shape: {upper.shape}")
    
    # Apply predictions only where mask_target is True (imputed values)
    # Create the target slice for imputation (t=1 to t=T)
    y_target_imputed = y_tensor_imputed[:, 1:, :]
    
    # Use tf.where for vectorized conditional assignment
    y_target_imputed = tf.where(mask_target, predictions, y_target_imputed)
    
    # Update the imputed tensor
    y_tensor_imputed = tf.concat([y_tensor_imputed[:, :1, :], y_target_imputed], axis=1)
    
    # Verify imputation
    changed_values = tf.reduce_sum(tf.cast(tf.not_equal(y_tensor, y_tensor_imputed), tf.int32))
    expected_changed = tf.reduce_sum(tf.cast(mask_target, tf.int32))  # Only count target timepoints
    print(f"Values changed during imputation: {changed_values.numpy()}")
    print(f"Expected imputed values (target timepoints): {expected_changed.numpy()}")
    
    if changed_values.numpy() == expected_changed.numpy():
        print(f"✅ Imputation successful!")
    else:
        print(f"⚠️  Warning: Number of changed values doesn't match expected imputed values")
    
    return X_tensor, y_tensor_imputed, mask_tensor, weight_tensor, lower, upper

### derive training and validating tensorflow data from tensor data
def get_trainval_batches(X_tensor, y_tensor, mask_tensor, weight_tensor):
    # Create TensorFlow dataset and split into training/validation
    train_dataset, val_dataset, dataset_info = create_tf_dataset_and_split(
        X_tensor, y_tensor, mask_tensor, weight_tensor
    )
    # batch datasets
    train_batches=train_dataset.shuffle(10000).batch(200).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_batches=val_dataset.shuffle(10000).batch(200).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_batches, val_batches

### derive predictions from the model
def y_pred( X_tensor, y_tensor, model):
    X_context = X_tensor[:, :-1, :]
    y_history = y_tensor[:, :-1, :]
    # Get predictions from model
    return model((X_context, y_history), training=False)
    
### impute missing y_tensor using the model along with confidence intervals
def y_model_impute(model, y_tensor_original, X_tensor_original, mask_tensor_original):
    current_y_tensor=tf.identity(y_tensor_original)
    current_lower=tf.identity(y_tensor_original)
    current_upper=tf.identity(y_tensor_original)
    
    N = tf.shape(y_tensor_original)[0]
    F = tf.shape(y_tensor_original)[2]
    timepoints=y_tensor_original.shape[1]-1

    for j in range(1, timepoints+1):
    
        print('j= ',j)
        mean_pred, lower, upper = y_pred(X_tensor_original, current_y_tensor, model)
        # print(f"mean_pred shape: {mean_pred.shape}")
        # print(f"lower shape: {lower.shape}")
        # print(f"upper shape: {upper.shape}")
        
        mask_slice = mask_tensor_original[:, j, :]  # (N, F) boolean
        # indices shape: (N, 2) -> [[0, j], [1, j], ...]
        row_idx = tf.range(N, dtype=tf.int32)
        col_idx = tf.fill([N], tf.cast(j, tf.int32))
        indices = tf.stack([row_idx, col_idx], axis=1)  # (N, 2)
        
        # tensor_scatter_nd_update 
        orig_slice = current_y_tensor[:, j, :]      # (N, F)
        mean_slice = mean_pred[:, j-1, :]           # (N, F)
        updated_slice = tf.where(mask_slice, mean_slice, orig_slice)  # (N, F)
        current_y_tensor = tf.tensor_scatter_nd_update(
            current_y_tensor,
            indices=indices,
            updates=updated_slice
        )
    
        print('check the update',mask_tensor_original[0,j,:],current_y_tensor[0,j,:]==mean_pred[0,j-1,:])
    
        orig_slice = current_lower[:, j, :]      # (N, F)
        lower_slice = lower[:, j-1, :]           # (N, F)
        updated_slice = tf.where(mask_slice, lower_slice, orig_slice)  # (N, F)
        current_lower = tf.tensor_scatter_nd_update(
            current_lower,
            indices=indices,
            updates=updated_slice
        )
        
        orig_slice = current_upper[:, j, :]      # (N, F)
        upper_slice = upper[:, j-1, :]           # (N, F)
        updated_slice = tf.where(mask_slice, upper_slice, orig_slice)  # (N, F)
        current_upper = tf.tensor_scatter_nd_update(
            current_upper,
            indices=indices,
            updates=updated_slice
        )
    return current_y_tensor, current_lower, current_upper


# loss function that factors mask and weight
# mask and weight are included in y_true along with label

@tf.keras.utils.register_keras_serializable()
def mse_loss(y_true, y_pred):
    label, mask, weight= y_true
    loss = tf.square(label - y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= (1-mask)
    loss *= weight 
    loss = tf.reduce_sum(loss)/tf.reduce_sum((1-mask))
    return loss


#### visulization, preprocessing

def load_and_plot_np2ptot():
    """
    Load the imputed data and plot year vs NP2PTOT and NP2PTOT_imputed 
    for three random PATNO
    
    Returns:
    --------
    df : pandas.DataFrame
        The loaded dataset
    common_row_number : int
        The common number of records per PATNO
    Xcol : list
        List of feature column names
    iynum : list
        List of imputed outcome column names
    isynum : list
        List of imputation status column names
    """
    
    # Load the data
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_PATH, DATA_FILES['complete_data']))
    
    print(f"Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique participants: {df['PATNO'].nunique()}")
    
    # Check if each PATNO has the same number of records
    patno_counts = df['PATNO'].value_counts()
    unique_counts = patno_counts.unique()
    
    print(f"\n=== PATNO Record Count Verification ===")
    print(f"Number of unique record counts: {len(unique_counts)}")
    print(f"Unique record counts: {unique_counts}")
    
    if len(unique_counts) == 1:
        common_row_number = unique_counts[0]
        print(f"✅ All PATNOs have the same number of records: {common_row_number}")
        print(f"Common row number for later use: {common_row_number}")
    else:
        print(f"⚠️  PATNOs have different numbers of records!")
        print(f"Record count distribution:")
        print(patno_counts.value_counts().sort_index())
        common_row_number = None
    
    # Check if required columns exist
    required_cols = ['PATNO', 'year', 'NP2PTOT', 'NP2PTOT_imputed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Get random PATNO (up to 3, or fewer if not enough available)
    random_patnos = df['PATNO'].unique()
    n_patients = len(random_patnos)
    
    if n_patients >= 3:
        selected_patnos = np.random.choice(random_patnos, size=3, replace=False)
    else:
        selected_patnos = random_patnos  # Use all available patients
    
    print(f"Selected PATNO: {selected_patnos}")
    
    # Create subplots (adjust number based on available patients)
    n_subplots = len(selected_patnos)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6*n_subplots, 6))
    if n_subplots == 1:
        axes = [axes]  # Ensure axes is always a list
    fig.suptitle(f'Year vs NP2PTOT and NP2PTOT_imputed for {n_subplots} Participant(s)', 
                 fontsize=16, fontweight='bold')
    
    for i, patno in enumerate(selected_patnos):
        # Filter data for this participant
        participant_data = df[df['PATNO'] == patno].copy()
        participant_data = participant_data.sort_values('year')
        
        ax = axes[i]
        
        # Plot original NP2PTOT values (where NP2PTOT_is_nan == 0)
        original_mask = participant_data['NP2PTOT_is_nan'] == 0
        if original_mask.any():
            ax.scatter(participant_data.loc[original_mask, 'year'], 
                      participant_data.loc[original_mask, 'NP2PTOT'], 
                      color='blue', alpha=0.7, s=60, label='Original NP2PTOT', zorder=3)
        
        # Plot imputed NP2PTOT values (where NP2PTOT_is_nan == 1)
        imputed_mask = participant_data['NP2PTOT_is_nan'] == 1
        if imputed_mask.any():
            ax.scatter(participant_data.loc[imputed_mask, 'year'], 
                      participant_data.loc[imputed_mask, 'NP2PTOT_imputed'], 
                      color='red', alpha=0.7, s=60, marker='s', label='Imputed NP2PTOT', zorder=3)
        
        # Connect points with lines for better visualization
        ax.plot(participant_data['year'], participant_data['NP2PTOT_imputed'], 
               color='red', alpha=0.5, linestyle='--', linewidth=1, zorder=1)
        
        if original_mask.any():
            ax.plot(participant_data.loc[original_mask, 'year'], 
                   participant_data.loc[original_mask, 'NP2PTOT'], 
                   color='blue', alpha=0.5, linewidth=2, zorder=2)
        
        # Get cohort information for this participant
        cohort = participant_data['COHORT_DEFINITION'].iloc[0] if 'COHORT_DEFINITION' in participant_data.columns else 'Unknown'
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('NP2PTOT Score', fontsize=12)
        ax.set_title(f'Participant {patno} ({cohort})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        n_original = original_mask.sum()
        n_imputed = imputed_mask.sum()
        ax.text(0.02, 0.98, f'Original: {n_original}\nImputed: {n_imputed}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot instead of displaying it
    save_path = '../fittedModels/np2ptot_participants_visualization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 NP2PTOT participants plot saved to: {save_path}")
    
    # Close the figure to prevent display and free memory
    plt.close(fig)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for patno in selected_patnos:
        participant_data = df[df['PATNO'] == patno]
        original_count = (participant_data['NP2PTOT_is_nan'] == 0).sum()
        imputed_count = (participant_data['NP2PTOT_is_nan'] == 1).sum()
        total_count = len(participant_data)
        
        print(f"Participant {patno}:")
        print(f"  Total observations: {total_count}")
        print(f"  Original NP2PTOT: {original_count}")
        print(f"  Imputed NP2PTOT: {imputed_count}")
        print(f"  Imputation rate: {imputed_count/total_count*100:.1f}%")
        print()
    
    # Define column groups for tensor formatting
    ### feature columns
    Xcol=['LEDD', 'year','ENROLL_AGE',
       'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT',  'SEX_1.0',
       'CHLDBEAR_missing_1', 'CHLDBEAR_1.0', 'CHLDBEAR_999.0',
       'HOWLIVE_missing_1', 'HOWLIVE_2.0', 'HOWLIVE_4.0', 'HOWLIVE_999.0',
       "COHORT_DEFINITION_encoded_Parkinson's Disease",
       'COHORT_DEFINITION_encoded_Prodromal',
       'COHORT_DEFINITION_encoded_SWEDD']
    ### imputed outcome columns
    iynum=['NP1PTOT_imputed',  'NP2PTOT_imputed','NP3TOT_imputed']
    ### imputation status
    isynum=['NP1PTOT_is_nan', 'NP2PTOT_is_nan',  'NP3TOT_is_nan']
    
    return df, common_row_number, Xcol, iynum, isynum


def plot_np2ptot_from_data(df, title_suffix="Testing Data"):
    """
    Plot year vs NP2PTOT and NP2PTOT_imputed for three random PATNO from the provided dataset.
    Modified version of load_and_plot_np2ptot() that works with existing data instead of loading it.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset with imputed values (e.g., dfy_with_imputed)
    title_suffix : str
        Suffix to add to the plot title (default: "Testing Data")
        
    Returns:
    --------
    common_row_number : int
        The common number of records per PATNO
    Xcol : list
        List of feature column names
    iynum : list
        List of imputed outcome column names
    isynum : list
        List of imputation status column names
    """
    
    print(f"Plotting NP2PTOT data from {title_suffix}...")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique participants: {df['PATNO'].nunique()}")
    
    # Check if each PATNO has the same number of records
    patno_counts = df['PATNO'].value_counts()
    unique_counts = patno_counts.unique()
    
    print(f"\n=== PATNO Record Count Verification ===")
    print(f"Number of unique record counts: {len(unique_counts)}")
    print(f"Unique record counts: {unique_counts}")
    
    if len(unique_counts) == 1:
        common_row_number = unique_counts[0]
        print(f"✅ All PATNOs have the same number of records: {common_row_number}")
        print(f"Common row number for later use: {common_row_number}")
    else:
        print(f"⚠️  PATNOs have different numbers of records!")
        print(f"Record count distribution:")
        print(patno_counts.value_counts().sort_index())
        common_row_number = None
    
    # Check if required columns exist
    required_cols = ['PATNO', 'year', 'NP2PTOT', 'NP2PTOT_imputed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None, None
    
    # Get random PATNO (up to 3, or fewer if not enough available)
    random_patnos = df['PATNO'].unique()
    n_patients = len(random_patnos)
    
    if n_patients >= 3:
        selected_patnos = np.random.choice(random_patnos, size=3, replace=False)
    else:
        selected_patnos = random_patnos  # Use all available patients
    
    print(f"Selected PATNO: {selected_patnos}")
    
    # Create subplots (adjust number based on available patients)
    n_subplots = len(selected_patnos)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6*n_subplots, 6))
    if n_subplots == 1:
        axes = [axes]  # Ensure axes is always a list
    fig.suptitle(f'Year vs NP2PTOT and NP2PTOT_imputed for {n_subplots} Participant(s) ({title_suffix})', 
                 fontsize=16, fontweight='bold')
    
    for i, patno in enumerate(selected_patnos):
        # Filter data for this participant
        participant_data = df[df['PATNO'] == patno].copy()
        participant_data = participant_data.sort_values('year')
        
        ax = axes[i]
        
        # Plot original NP2PTOT values (where NP2PTOT_is_nan == 0)
        original_mask = participant_data['NP2PTOT_is_nan'] == 0
        if original_mask.any():
            ax.scatter(participant_data.loc[original_mask, 'year'], 
                      participant_data.loc[original_mask, 'NP2PTOT'], 
                      color='blue', alpha=0.7, s=60, label='Original NP2PTOT', zorder=3)
        
        # Plot imputed NP2PTOT values (where NP2PTOT_is_nan == 1)
        imputed_mask = participant_data['NP2PTOT_is_nan'] == 1
        if imputed_mask.any():
            ax.scatter(participant_data.loc[imputed_mask, 'year'], 
                      participant_data.loc[imputed_mask, 'NP2PTOT_imputed'], 
                      color='red', alpha=0.7, s=60, marker='s', label='Imputed NP2PTOT', zorder=3)
        
        # Connect points with lines for better visualization
        ax.plot(participant_data['year'], participant_data['NP2PTOT_imputed'], 
               color='red', alpha=0.5, linestyle='--', linewidth=1, zorder=1)
        
        if original_mask.any():
            ax.plot(participant_data.loc[original_mask, 'year'], 
                   participant_data.loc[original_mask, 'NP2PTOT'], 
                   color='blue', alpha=0.5, linewidth=2, zorder=2)
        
        # Get cohort information for this participant
        cohort = participant_data['COHORT_DEFINITION'].iloc[0] if 'COHORT_DEFINITION' in participant_data.columns else 'Unknown'
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('NP2PTOT Score', fontsize=12)
        ax.set_title(f'Participant {patno} ({cohort})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        n_original = original_mask.sum()
        n_imputed = imputed_mask.sum()
        ax.text(0.02, 0.98, f'Original: {n_original}\nImputed: {n_imputed}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for patno in selected_patnos:
        participant_data = df[df['PATNO'] == patno]
        original_count = (participant_data['NP2PTOT_is_nan'] == 0).sum()
        imputed_count = (participant_data['NP2PTOT_is_nan'] == 1).sum()
        total_count = len(participant_data)
        
        print(f"Participant {patno}:")
        print(f"  Total observations: {total_count}")
        print(f"  Original NP2PTOT: {original_count}")
        print(f"  Imputed NP2PTOT: {imputed_count}")
        print(f"  Imputation rate: {imputed_count/total_count*100:.1f}%")
        print()
    
    # Define column groups for tensor formatting
    ### feature columns
    Xcol=['LEDD', 'year','ENROLL_AGE',
       'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT',  'SEX_10',
       'CHLDBEAR_missing_1', 'CHLDBEAR_1.0', 'CHLDBEAR_999.0',
       'HOWLIVE_missing_1', 'HOWLIVE_2.0', 'HOWLIVE_4.0', 'HOWLIVE_999.0',
       'COHORT_DEFINITION_encoded_Parkinsons_Disease',
       'COHORT_DEFINITION_encoded_Prodromal',
       'COHORT_DEFINITION_encoded_SWEDD']
    ### imputed outcome columns
    iynum=['NP1PTOT_imputed',  'NP2PTOT_imputed','NP3TOT_imputed']
    ### imputation status
    isynum=['NP1PTOT_is_nan', 'NP2PTOT_is_nan',  'NP3TOT_is_nan']
    
    return common_row_number, Xcol, iynum, isynum

def train_model_with_iterations(model, X_tensor, y_tensor, mask_tensor, weight_tensor, 
                               optimizer=None, loss_fn=None, 
                               max_epochs=5, n_iterations=2, patience=5, 
                               verbose=True, save_best_model=True, save_path=None):
    """
    Train a model with multiple iterations and early stopping.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The model to train
    X_tensor : tf.Tensor
        Input features tensor
    y_tensor : tf.Tensor
        Target values tensor
    mask_tensor : tf.Tensor
        Mask tensor for imputed values
    weight_tensor : tf.Tensor
        Weight tensor for loss computation
    optimizer : tf.keras.optimizers.Optimizer, optional
        Optimizer to use. Defaults to Adam with lr=0.001
    loss_fn : callable, optional
        Loss function to use. Defaults to mse_loss
    max_epochs : int, default=5
        Maximum epochs per iteration
    n_iterations : int, default=2
        Number of training iterations
    patience : int, default=5
        Early stopping patience
    verbose : bool, default=True
        Whether to print training progress
    save_best_model : bool, default=True
        Whether to save the best model
    save_path : str, optional
        Path to save the best model. If None, uses default path
        
    Returns:
    --------
    dict : Training results containing:
        - best_model: The best model across all iterations
        - best_val_loss: Best validation loss achieved
        - best_iteration: Iteration number of best model
        - all_train_losses: List of training losses per iteration
        - all_val_losses: List of validation losses per iteration
        - all_iteration_metrics: Detailed metrics per iteration
    """
    import tensorflow as tf
    import numpy as np
    import os
    
    # Set default optimizer and loss function
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if loss_fn is None:
        loss_fn = mse_loss
    
    # Training parameters
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Store training history
    train_losses_iteration = []
    val_losses_iteration = []
    
    if verbose:
        print(f"\n🚀 Starting {n_iterations}-iteration training with imputation...")
        print(f"Max epochs per iteration: {max_epochs}")
        print(f"Early stopping patience: {patience}")
    
    # Store original tensors for reference
    X_tensor_original = X_tensor
    y_tensor_original = y_tensor
    mask_tensor_original = mask_tensor
    weight_tensor_original = weight_tensor
    
    # Track best model across all iterations
    best_val_loss_overall = float('inf')
    best_model = None
    best_iteration = 0
    
    # Store training history for all iterations
    all_train_losses = []
    all_val_losses = []
    all_iteration_metrics = []

    current_y_tensor = y_tensor
    
    for i_impute in range(n_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {i_impute + 1}/{n_iterations}")
            print(f"{'='*60}")
        
        # Reset training history for this iteration
        train_losses_iteration = []
        val_losses_iteration = []
        
        # Reset early stopping for this iteration
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_batches, val_batches = get_trainval_batches(X_tensor, current_y_tensor, mask_tensor, weight_tensor)
    
        for epoch in range(max_epochs):
            if verbose:
                print(f"\nEpoch {epoch+1}/{max_epochs}")
            
            # Training
            train_loss_epoch = []
            train_mae_epoch = []
            
            for batch_idx, (x, yh, y, masky, weighty) in enumerate(train_batches):
                with tf.GradientTape() as tape:
                    ypred = model((x, yh), training=True)
                    loss = loss_fn((y, masky, weighty), ypred)
                    mae = tf.reduce_mean(tf.abs(y - ypred))
                
                # Compute gradients and update weights
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                train_loss_epoch.append(loss.numpy())
                train_mae_epoch.append(mae.numpy())
                
                if verbose and batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}: loss={loss.numpy():.6f}, mae={mae.numpy():.6f}")
            
            # Validation
            val_loss_epoch = []
            val_mae_epoch = []
            
            for x, yh, y, masky, weighty in val_batches:
                ypred = model((x, yh), training=True)
                loss = loss_fn((y, masky, weighty), ypred)
                mae = tf.reduce_mean(tf.abs(y - ypred))
                val_loss_epoch.append(loss.numpy())
                val_mae_epoch.append(mae.numpy())
            
            # Calculate epoch averages
            avg_train_loss = np.mean(train_loss_epoch)
            avg_train_mae = np.mean(train_mae_epoch)
            avg_val_loss = np.mean(val_loss_epoch)
            avg_val_mae = np.mean(val_mae_epoch)
            
            train_losses_iteration.append(avg_train_loss)
            val_losses_iteration.append(avg_val_loss)
            
            if verbose:
                print(f"  Epoch {epoch+1} Summary:")
                print(f"    Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f}")
                print(f"    Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}")
            
            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if verbose:
                    print(f"    ✅ Validation loss improved! Best: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if verbose:
                    print(f"    ⏳ No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"    🛑 Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Store iteration results
        all_train_losses.append(train_losses_iteration)
        all_val_losses.append(val_losses_iteration)
        
        # Calculate final metrics for this iteration
        final_train_loss = train_losses_iteration[-1] if train_losses_iteration else float('inf')
        final_val_loss = val_losses_iteration[-1] if val_losses_iteration else float('inf')
        
        iteration_metrics = {
            'iteration': i_impute + 1,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'epochs_trained': len(train_losses_iteration)
        }
        all_iteration_metrics.append(iteration_metrics)
        
        if verbose:
            print(f"\n✅ Iteration {i_impute + 1} completed!")
            print(f"Final validation loss: {final_val_loss:.6f}")
            print(f"Total epochs trained: {len(train_losses_iteration)}")
        
        # Check if this is the best model across all iterations
        if final_val_loss < best_val_loss_overall:
            best_val_loss_overall = final_val_loss
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())
            best_iteration = i_impute + 1
            if verbose:
                print(f"🏆 NEW BEST MODEL! Val Loss: {best_val_loss_overall:.6f} (Iteration {best_iteration})")
        else:
            if verbose:
                print(f"📊 Current best: Val Loss {best_val_loss_overall:.6f} (Iteration {best_iteration})")
    
    # Save the best model if requested
    if save_best_model and best_model is not None:
        if save_path is None:
            save_path = '../fittedModels/ppmi_transformer_model_final.keras'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_model.save(save_path)
        if verbose:
            print(f"\n{'='*60}")
            print(f"SAVING BEST MODEL")
            print(f"{'='*60}")
            print(f"Best model: Iteration {best_iteration}")
            print(f"Best validation loss: {best_val_loss_overall:.6f}")
            print(f"✅ Best model saved to: {save_path}")
    
    # Print overall training summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total iterations: {n_iterations}")
        print(f"Best iteration: {best_iteration}")
        print(f"Best validation loss: {best_val_loss_overall:.6f}")
        print(f"Total epochs trained: {sum(len(iter_losses) for iter_losses in all_train_losses)}")
    
    # Return training results
    return {
        'best_model': best_model,
        'best_val_loss': best_val_loss_overall,
        'best_iteration': best_iteration,
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses,
        'all_iteration_metrics': all_iteration_metrics,
        'save_path': save_path if save_best_model else None
    }


def plot_training_results(training_results, save_path=None, show_plot=False):
    """
    Create comprehensive training plots from training results.
    
    Parameters:
    -----------
    training_results : dict
        Results dictionary from train_model_with_iterations function
    save_path : str, optional
        Path to save the plot. If None, plot is not saved
    show_plot : bool, default=False
        Whether to display the plot (plots are saved by default, not displayed)
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data from training results
    all_train_losses = training_results['all_train_losses']
    all_val_losses = training_results['all_val_losses']
    all_iteration_metrics = training_results['all_iteration_metrics']
    best_iteration = training_results['best_iteration']
    best_val_loss = training_results['best_val_loss']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Results Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training losses for all iterations
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_train_losses)))
    
    for i, train_losses in enumerate(all_train_losses):
        if train_losses:  # Only plot if there are losses
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 
                    label=f'Iteration {i+1}', 
                    marker='o', linewidth=2, markersize=4,
                    color=colors[i])
    
    ax1.set_title('Training Loss Across All Iterations', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation losses for all iterations
    ax2 = axes[0, 1]
    for i, val_losses in enumerate(all_val_losses):
        if val_losses:  # Only plot if there are losses
            epochs = range(1, len(val_losses) + 1)
            ax2.plot(epochs, val_losses, 
                    label=f'Iteration {i+1}', 
                    marker='s', linewidth=2, markersize=4,
                    color=colors[i])
    
    ax2.set_title('Validation Loss Across All Iterations', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final losses comparison by iteration
    ax3 = axes[1, 0]
    iterations = [metrics['iteration'] for metrics in all_iteration_metrics]
    final_train_losses = [metrics['final_train_loss'] for metrics in all_iteration_metrics]
    final_val_losses = [metrics['final_val_loss'] for metrics in all_iteration_metrics]
    
    ax3.plot(iterations, final_train_losses, 'o-', 
            label='Final Train Loss', linewidth=3, markersize=8, color='blue')
    ax3.plot(iterations, final_val_losses, 's-', 
            label='Final Val Loss', linewidth=3, markersize=8, color='red')
    
    # Highlight the best iteration
    ax3.axvline(x=best_iteration, color='green', linestyle='--', alpha=0.7, 
               label=f'Best Iteration ({best_iteration})')
    ax3.scatter([best_iteration], [best_val_loss], color='green', s=100, zorder=5)
    
    ax3.set_title('Final Losses by Iteration', fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Final Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Epochs trained per iteration
    ax4 = axes[1, 1]
    epochs_trained = [metrics['epochs_trained'] for metrics in all_iteration_metrics]
    
    bars = ax4.bar(iterations, epochs_trained, color=colors[:len(iterations)], alpha=0.7)
    ax4.set_title('Epochs Trained per Iteration', fontweight='bold')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Epochs Trained')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, epochs in zip(bars, epochs_trained):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{epochs}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight the best iteration bar
    if best_iteration <= len(bars):
        bars[best_iteration - 1].set_edgecolor('green')
        bars[best_iteration - 1].set_linewidth(3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add summary text box
    summary_text = f"""Training Summary:
• Best Iteration: {best_iteration}
• Best Validation Loss: {best_val_loss:.6f}
• Total Epochs: {sum(epochs_trained)}
• Total Iterations: {len(iterations)}"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved to: {save_path}")
    
    # Close the figure to prevent display and free memory
    plt.close(fig)
    
    # Show plot if requested (but this will be False by default)
    if show_plot:
        plt.show()
    
    return fig


def plot_loss_comparison(all_train_losses, all_val_losses, save_path=None, show_plot=False):
    """
    Create a simple side-by-side comparison of training and validation losses.
    
    Parameters:
    -----------
    all_train_losses : list
        List of training loss lists for each iteration
    all_val_losses : list
        List of validation loss lists for each iteration
    save_path : str, optional
        Path to save the plot. If None, plot is not saved
    show_plot : bool, default=False
        Whether to display the plot (plots are saved by default, not displayed)
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training vs Validation Loss Comparison', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_train_losses)))
    
    # Plot training losses
    for i, train_losses in enumerate(all_train_losses):
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 
                    label=f'Iteration {i+1}', 
                    marker='o', linewidth=2, markersize=4,
                    color=colors[i])
    
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation losses
    for i, val_losses in enumerate(all_val_losses):
        if val_losses:
            epochs = range(1, len(val_losses) + 1)
            ax2.plot(epochs, val_losses, 
                    label=f'Iteration {i+1}', 
                    marker='s', linewidth=2, markersize=4,
                    color=colors[i])
    
    ax2.set_title('Validation Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Loss comparison plot saved to: {save_path}")
    
    # Close the figure to prevent display and free memory
    plt.close(fig)
    
    # Show plot if requested (but this will be False by default)
    if show_plot:
        plt.show()
    
    return fig


def preprocess_baseline_data(baseline_df, fitted_models_path="./fittedModels", verbose=True):
    """
    Preprocess baseline data for transformer model: handle missing values, create flags, add b_ prefix.
    This is a reusable function that can be used by other parts of the application.
    
    Parameters:
    -----------
    baseline_df : pandas.DataFrame
        Baseline data to preprocess
    fitted_models_path : str
        Path to the fittedModels directory for loading ynum_means
    verbose : bool
        Whether to print progress messages (default: True)
        
    Returns:
    --------
    baseline_processed : pandas.DataFrame
        Preprocessed baseline data ready for transformer model
    """
    
    # Define variable lists (same as development data)
    ivar = ['PATNO']
    tx = ['COHORT_DEFINITION']
    ynum = ['NP1PTOT', 'NP2PTOT', 'NP3TOT']
    Xcat = ['SEX', 'CHLDBEAR_missing', 'CHLDBEAR', 'HOWLIVE_missing', 'HOWLIVE']
    Xnum = ['ENROLL_AGE', 'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT']
    
    # Load ynum_means from fittedModels
    ynum_means_path = os.path.join(fitted_models_path, 'ynum_means.pkl')
    if os.path.exists(ynum_means_path):
        ynum_means = joblib.load(ynum_means_path)
        if verbose:
            print(f"Loaded ynum_means: {list(ynum_means.keys())}")
    else:
        if verbose:
            print("Warning: ynum_means.pkl not found!")
        ynum_means = {}
    
    # Process baseline data for transformer model (add b_ prefix and handle missing values)
    baseline_processed = baseline_df.copy()
    
    # Handle NaN values and create missing flags
    # Always create missing flag columns for CHLDBEAR and HOWLIVE (even if no missing values)
    required_missing_flags = ['CHLDBEAR', 'HOWLIVE']
    for col in required_missing_flags:
        if col in baseline_processed.columns:
            flag_col_name = f"{col}_missing"
            baseline_processed[flag_col_name] = baseline_processed[col].isnull().astype(int)
    
    # Handle any other columns with NaN values
    nan_columns = baseline_processed.columns[baseline_processed.isnull().any()].tolist()
    for col in nan_columns:
        if col not in required_missing_flags:  # Skip if already handled above
            nan_count = baseline_processed[col].isnull().sum()
            if nan_count > 0:
                # Create flag column for this column
                flag_col_name = f"{col}_missing"
                baseline_processed[flag_col_name] = baseline_processed[col].isnull().astype(int)
                
                # Replace NaN based on column type
                if col in ynum:
                    # Use loaded mean for clinical scores
                    mean_key = f"{col}_mean"
                    if mean_key in ynum_means:
                        col_mean = ynum_means[mean_key]
                        baseline_processed[col] = baseline_processed[col].fillna(col_mean)
                        if verbose:
                            print(f"    Created flag column: {flag_col_name}")
                            print(f"    Replaced {nan_count} NaN values with loaded mean ({col_mean:.2f})")
                    else:
                        # Fallback to calculated mean
                        col_mean = baseline_processed[col].mean()
                        baseline_processed[col] = baseline_processed[col].fillna(col_mean)
                        if verbose:
                            print(f"    Created flag column: {flag_col_name}")
                            print(f"    Replaced {nan_count} NaN values with calculated mean ({col_mean:.2f})")
                else:
                    # Replace NaN with 999 for other columns
                    baseline_processed[col] = baseline_processed[col].fillna(999)
                    if verbose:
                        print(f"    Created flag column: {flag_col_name}")
                        print(f"    Replaced {nan_count} NaN values with 999")
    
    # Fill any remaining NaN values
    for col in baseline_processed.columns:
        if baseline_processed[col].isnull().any():
            if col in ynum:
                col_mean = baseline_processed[col].mean()
                baseline_processed[col] = baseline_processed[col].fillna(col_mean)
            else:
                baseline_processed[col] = baseline_processed[col].fillna(999)
    
    # Add "b_" prefix to ynum columns (same as development data)
    if verbose:
        print("\nAdding 'b_' prefix to ynum columns...")
    ynum_rename_dict = {col: f"b_{col}" for col in ynum}
    baseline_processed = baseline_processed.rename(columns=ynum_rename_dict)
    
    if verbose:
        print(f"Renamed {len(ynum)} ynum columns with 'b_' prefix:")
        for old_name, new_name in ynum_rename_dict.items():
            print(f"  {old_name} -> {new_name}")
    
    # Select final columns for baseline_df
    baseline_processed = baseline_processed[ivar + Xcat + Xnum + tx]
    
    if verbose:
        print("\nFinal baseline data shape:", baseline_processed.shape)
        print("Final columns:", len(baseline_processed.columns))
    
    return baseline_processed


def load_testing_data(baseline_data=None, longitudinal_data=None, base_path="./data", fitted_models_path="./fittedModels"):
    """
    Load testing data and apply the same preprocessing as development data.
    Can accept baseline and longitudinal data directly or load from CSV file.
    
    Parameters:
    -----------
    baseline_data : dict, list, or None
        Baseline data (dict for single patient, list for multiple, None to load from CSV)
    longitudinal_data : list or None
        Longitudinal data list (if None, will load from CSV)
    base_path : str
        Base path to the PPMI directory (default: "/Users/xiaolongluo/Documents/PPMI")
    fitted_models_path : str
        Path to the fittedModels directory (default: "/Users/xiaolongluo/Documents/PPMI/fittedModels")
        
    Returns:
    --------
    baseline_processed : pandas.DataFrame
        Preprocessed baseline data
    dfy : pandas.DataFrame
        Longitudinal data
    ivar : list
        Identifier variables
    tvar : list
        Time variables
    Xt : list
        Time-varying predictors
    ynum : list
        Numerical outcome variables
    tx : list
        Treatment variables
    """
    
    if baseline_data is not None and longitudinal_data is not None:
        # Use provided data
        print("Using provided baseline and longitudinal data...")
        
        # Convert to DataFrames (handle both single dict and list of dicts)
        if isinstance(baseline_data, dict):
            baseline_df = pd.DataFrame([baseline_data])
        else:
            baseline_df = pd.DataFrame(baseline_data)
            
        longitudinal_df = pd.DataFrame(longitudinal_data)
        
        # Create combined dataset for processing
        df = longitudinal_df.merge(
            baseline_df[['PATNO', 'ENROLL_AGE', 'SEX', 'CHLDBEAR', 'COHORT_DEFINITION', 'HOWLIVE']], 
            on='PATNO', 
            how='left'
        )
        
        # Add baseline observation (year 0) to the combined dataset
        baseline_obs = baseline_df.copy()
        df = pd.concat([baseline_obs, df], ignore_index=True)
        
        # Sort by PATNO and year
        df = df.sort_values(['PATNO', 'year']).reset_index(drop=True)
        
        print("Combined data shape:", df.shape)
        print("Combined data columns:", len(df.columns))
        
    else:
        # Load testing data from CSV (original behavior)
        df = pd.read_csv(f"{base_path}/data/testing_data.csv")
        
        print("Testing data shape:", df.shape)
        print("Testing data columns:", len(df.columns))
        
        # Convert INFODT to datetime for sorting
        df['INFODT'] = pd.to_datetime(df['INFODT'])
        
        # Create baseline dataframe by taking the earliest (by INFODT) row for each PATNO
        baseline_df = df.loc[df.groupby('PATNO')['INFODT'].idxmin()].copy()
        
        print("Baseline data shape:", baseline_df.shape)
        print("Unique participants in baseline:", baseline_df['PATNO'].nunique())
        
        # Drop administrative columns (same as development data)
        admin_cols_to_drop = [
            'REC_ID', 'PAG_NAME', 'LAST_UPDATE', 'COHORT', 'ENROLL_DATE',
            'ENROLL_STATUS', 'STATUS_DATE', 'SCREENEDAM', 'INEXPAGE',
            'AV133STDY', 'NXTAUSTDY', 'DATELIG', 'PPMI_ONLINE_ENROLL', 
            'data_split', 'EVENT_ID', 'INFODT', 'ORIG_ENTRY'
        ]
        
        # Only drop columns that exist in the dataframe
        cols_to_drop = [col for col in admin_cols_to_drop if col in baseline_df.columns]
        baseline_df = baseline_df.drop(columns=cols_to_drop)
        
        print("Columns dropped:", cols_to_drop)
        print("Baseline data shape after dropping admin columns:", baseline_df.shape)
    
    # Define variable lists (same as development data)
    ivar = ['PATNO']
    tvar = ['year']
    tx = ['COHORT_DEFINITION']
    ynum = ['NP1PTOT', 'NP2PTOT', 'NP3TOT']
    Xt = ['LEDD', 'year']
    
    # Create dfy (longitudinal data)
    dfy = df[ivar + Xt + ynum]
    
    # Use the reusable preprocessing function
    baseline_processed = preprocess_baseline_data(baseline_df, fitted_models_path)
    
    return baseline_processed, dfy, ivar, tvar, Xt, ynum, tx



def normalize_testing_data(baseline_df, dfy, Xcat, Xnum, tx, ynum, Xt, fitted_models_path="./fittedModels"):
    """
    Normalize testing data using the saved scalers from fittedModels:
    - One-hot encode categorical variables (Xcat and tx)
    - Apply saved scalers to numerical variables (Xnum)
    - Apply saved scalers to ynum and Xt[:-1] in dfy
    
    Parameters:
    -----------
    fitted_models_path : str
        Path to the fittedModels directory (default: "/Users/xiaolongluo/Documents/PPMI/fittedModels")
    """
    print("\n=== Starting Testing Data Normalization ===")
    
    # Load saved scalers from fittedModels
    print("Loading saved scalers...")
    
    # Load baseline_Xnum_scaler
    baseline_scaler_path = os.path.join(fitted_models_path, 'baseline_Xnum_scaler.pkl')
    if os.path.exists(baseline_scaler_path):
        baseline_scaler = joblib.load(baseline_scaler_path)
        print(f"  Loaded baseline_Xnum_scaler")
    else:
        print(f"  Warning: {baseline_scaler_path} not found!")
        baseline_scaler = None
    
    # Load dfy_scaler
    dfy_scaler_path = os.path.join(fitted_models_path, 'dfy_scaler.pkl')
    if os.path.exists(dfy_scaler_path):
        dfy_scaler = joblib.load(dfy_scaler_path)
        print(f"  Loaded dfy_scaler")
    else:
        print(f"  Warning: {dfy_scaler_path} not found!")
        dfy_scaler = None
    
    # 1. Process baseline_df
    print("\n1. Processing baseline_df...")
    
    # Create a copy to avoid modifying original
    baseline_normalized = baseline_df.copy()
    
    # One-hot encode categorical variables (Xcat only)
    categorical_cols = Xcat
    print(f"One-hot encoding categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        if col in baseline_normalized.columns:
            # Create one-hot encoded columns
            dummies = pd.get_dummies(baseline_normalized[col], prefix=col, drop_first=True)
            
            # Drop original column and add dummies
            baseline_normalized = baseline_normalized.drop(columns=[col])
            baseline_normalized = pd.concat([baseline_normalized, dummies], axis=1)
            
            # Rename columns to match development data preprocessing (SEX_1.0 -> SEX_10)
            if col == 'SEX':
                # Check if SEX_1.0 exists in the dummies and rename it
                if 'SEX_1.0' in dummies.columns:
                    dummies = dummies.rename(columns={'SEX_1.0': 'SEX_10'})
                    # Drop the old SEX_1.0 from baseline_normalized and add the renamed version
                    if 'SEX_1.0' in baseline_normalized.columns:
                        baseline_normalized = baseline_normalized.drop(columns=['SEX_1.0'])
                    baseline_normalized = pd.concat([baseline_normalized, dummies], axis=1)
                    print(f"  {col} -> {list(dummies.columns)} (renamed to match development data)")
                else:
                    print(f"  {col} -> {list(dummies.columns)}")
            else:
                print(f"  {col} -> {list(dummies.columns)}")
    
    # One-hot encode tx columns but keep original versions too
    print(f"One-hot encoding tx columns while keeping originals: {tx}")
    for col in tx:
        if col in baseline_normalized.columns:
            # Create one-hot encoded columns
            dummies = pd.get_dummies(baseline_normalized[col], prefix=f"{col}_encoded", drop_first=True)
            
            # Keep original column and add dummies
            baseline_normalized = pd.concat([baseline_normalized, dummies], axis=1)
            
            print(f"  {col} -> {list(dummies.columns)} (original retained)")
    
    # Ensure all expected COHORT_DEFINITION columns exist for consistency with development data
    print("\nEnsuring all expected COHORT_DEFINITION columns exist...")
    expected_columns = [
        "COHORT_DEFINITION_encoded_Parkinson's Disease",
        'COHORT_DEFINITION_encoded_Prodromal', 
        'COHORT_DEFINITION_encoded_SWEDD'
    ]
    
    for expected_col in expected_columns:
        if expected_col not in baseline_normalized.columns:
            # Create missing column with all zeros
            baseline_normalized[expected_col] = 0
            print(f"  Created missing column: {expected_col} (all zeros)")
        else:
            print(f"  Column exists: {expected_col}")
    
    # Also create the version without apostrophe that the models expect
    print("\nCreating alternative column names for model compatibility...")
    alternative_columns = [
        'COHORT_DEFINITION_encoded_Parkinsons_Disease',  # Without apostrophe
        'COHORT_DEFINITION_encoded_Prodromal', 
        'COHORT_DEFINITION_encoded_SWEDD'
    ]
    
    for alt_col in alternative_columns:
        if alt_col not in baseline_normalized.columns:
            # Find the corresponding column with apostrophe
            if alt_col == 'COHORT_DEFINITION_encoded_Parkinsons_Disease':
                source_col = "COHORT_DEFINITION_encoded_Parkinson's Disease"
                if source_col in baseline_normalized.columns:
                    baseline_normalized[alt_col] = baseline_normalized[source_col]
                    print(f"  Created {alt_col} from {source_col}")
                else:
                    baseline_normalized[alt_col] = 0
                    print(f"  Created {alt_col} with zeros")
            else:
                baseline_normalized[alt_col] = 0
                print(f"  Created {alt_col} with zeros")
        else:
            print(f"  Alternative column exists: {alt_col}")
    
    # Apply saved scaler to numerical variables (Xnum)
    print(f"\nApplying saved scaler to numerical columns: {Xnum}")
    
    # Filter Xnum to only include columns that exist in baseline_normalized
    available_Xnum = [col for col in Xnum if col in baseline_normalized.columns]
    
    if available_Xnum and baseline_scaler is not None:
        # Apply the saved scaler (transform, not fit_transform)
        baseline_normalized[available_Xnum] = baseline_scaler.transform(baseline_normalized[available_Xnum])
        print(f"  Scaled columns using saved scaler: {available_Xnum}")
    else:
        print("  No numerical columns found for scaling or scaler not available")
    
    # 2. Process dfy
    print("\n2. Processing dfy...")
    
    # Create a copy to avoid modifying original
    dfy_normalized = dfy.copy()
    
    # Apply saved scaler to ynum and Xt[:-1] (excluding the last element of Xt)
    dfy_cols_to_scale = ynum + Xt[:-1]
    print(f"Applying saved scaler to dfy columns: {dfy_cols_to_scale}")
    
    # Filter to only include columns that exist in dfy_normalized
    available_dfy_cols = [col for col in dfy_cols_to_scale if col in dfy_normalized.columns]
    
    if available_dfy_cols and dfy_scaler is not None:
        # Apply the saved scaler (transform, not fit_transform)
        dfy_normalized[available_dfy_cols] = dfy_scaler.transform(dfy_normalized[available_dfy_cols])
        print(f"  Scaled columns using saved scaler: {available_dfy_cols}")
    else:
        print("  No dfy columns found for scaling or scaler not available")
    
    # 3. Print summary
    print("\n=== Normalization Summary ===")
    print(f"Baseline_df shape: {baseline_normalized.shape}")
    print(f"dfy shape: {dfy_normalized.shape}")
    print(f"Scalers used: baseline_Xnum_scaler, dfy_scaler")
    
    return baseline_normalized, dfy_normalized

def combine_testing_data(baseline_df, dfy_normalized, ivar, tx):
    """
    Combine baseline and longitudinal data for testing purposes.
    Sets propensity scores to 1 for consistency with development data structure.
    
    Parameters:
    -----------
    baseline_df : pandas.DataFrame
        Normalized baseline data
    dfy_normalized : pandas.DataFrame
        Normalized longitudinal data
    ivar : list
        List of identifier variables (e.g., ['PATNO'])
    tx : list
        List of treatment/exposure variables (e.g., ['COHORT_DEFINITION'])
        
    Returns:
    --------
    dfy_with_baseline : pandas.DataFrame
        Complete dataset with longitudinal and baseline data merged
    """
    print(f"\n=== Combining Testing Data ===")
    
    # Create propensity score columns with value 1 for consistency
    print("Creating propensity score columns with value 1 for consistency...")
    
    # Get unique values from the treatment variable to create PS columns
    if tx and tx[0] in baseline_df.columns:
        unique_values = baseline_df[tx[0]].unique()
        print(f"Unique {tx[0]} values: {unique_values}")
        
        # Create propensity score columns with value 1
        for value in unique_values:
            # Clean the value name for column name
            clean_value = str(value).replace(' ', '_').replace("'", '').replace('-', '_')
            ps_col_name = f"PS_{clean_value}"
            baseline_df[ps_col_name] = 1.0
            print(f"  Created {ps_col_name} = 1.0")
    else:
        print(f"Warning: {tx[0]} not found in baseline_df")
    
    print(f"Baseline with PS shape: {baseline_df.shape}")
    print(f"PS columns: {[col for col in baseline_df.columns if col.startswith('PS_')]}")
    
    # Merge baseline_with_ps into dfy_normalized by ivar
    print(f"\n=== Merging Baseline Data into Longitudinal Data ===")
    
    # Merge baseline data (including propensity scores) into longitudinal data
    dfy_with_baseline = dfy_normalized.merge(baseline_df, on=ivar[0], how='left', suffixes=('', '_baseline'))
    
    print(f"Longitudinal data shape before merge: {dfy_normalized.shape}")
    print(f"Longitudinal data shape after merge: {dfy_with_baseline.shape}")
    print(f"Number of columns added: {dfy_with_baseline.shape[1] - dfy_normalized.shape[1]}")
    
    # Check for any missing values in the merge
    missing_patno = dfy_normalized[ivar[0]].nunique() - dfy_with_baseline[ivar[0]].nunique()
    if missing_patno > 0:
        print(f"Warning: {missing_patno} participants missing from merge")
    else:
        print("All participants successfully merged")
    
    print(f"\n=== Complete Testing Dataset Summary ===")
    print(f"Complete longitudinal dataset shape: {dfy_with_baseline.shape}")
    print(f"Columns in final dataset: {list(dfy_with_baseline.columns)}")
    
    return dfy_with_baseline

def extend_testing_data_with_future_timepoints(dfy_with_baseline, tvar, ynum, kmax=41, tdelta=0.5):
    """
    Extend testing dataset with future timepoints using the same logic as PPMIpreprocess.py.
    This function replicates the extend_dataset_with_future_timepoints function for testing data.
    
    Parameters:
    -----------
    dfy_with_baseline : pandas.DataFrame
        Combined testing dataset with baseline and longitudinal data
    tvar : list
        List of time variables (e.g., ['year'])
    ynum : list
        List of outcome variables to set as NaN for future timepoints
    kmax : int
        Maximum number of timepoints per participant (default: 41)
    tdelta : float
        Time increment between timepoints (default: 0.5)
        
    Returns:
    --------
    dfy_extended : pandas.DataFrame
        Extended testing dataset with future timepoints
    """
    print(f"\n=== Extending Testing Dataset with Future Timepoints ===")
    print(f"Target Kmax: {kmax}")
    print(f"Time increment: {tdelta}")
    print(f"Time variable: {tvar}")
    print(f"Outcome variables to set as NaN: {ynum}")
    
    # Create a copy to avoid modifying original
    dfy_extended = dfy_with_baseline.copy()
    
    # 1. Calculate current timepoint counts
    print("\n1. Calculating current timepoint counts...")
    time_counts = dfy_extended.groupby('PATNO')[tvar[0]].count()
    
    print(f"Current timepoint counts summary:")
    print(f"  Min timepoints per PATNO: {time_counts.min()}")
    print(f"  Max timepoints per PATNO: {time_counts.max()}")
    print(f"  Mean timepoints per PATNO: {time_counts.mean():.2f}")
    print(f"  Current max time range: {dfy_extended[tvar[0]].min():.2f} to {dfy_extended[tvar[0]].max():.2f}")
    
    # 2. Summary of current timepoints
    print(f"\n2. Summary of current timepoints...")
    
    # Get current max time for each PATNO
    current_max_times = dfy_extended.groupby('PATNO')[tvar[0]].max()
    
    print(f"Current timepoint counts summary:")
    print(f"  Min timepoints per PATNO: {time_counts.min()}")
    print(f"  Max timepoints per PATNO: {time_counts.max()}")
    print(f"  Mean timepoints per PATNO: {time_counts.mean():.2f}")
    print(f"  Current max time range: {current_max_times.min():.2f} to {current_max_times.max():.2f}")
    
    # 3. Using Kmax parameter
    print(f"\n3. Using Kmax parameter...")
    
    print(f"Kmax = {kmax} (from function parameter)")
    
    # 4. Calculate rows to add for each PATNO
    print(f"\n4. Calculating rows to add for each PATNO...")
    
    rows_to_add = {}
    for patno in dfy_with_baseline['PATNO'].unique():
        current_count = len(dfy_with_baseline[dfy_with_baseline['PATNO'] == patno])
        if current_count < kmax:
            rows_to_add[patno] = kmax - current_count
    
    rows_to_add_series = pd.Series(rows_to_add)
    
    print(f"Rows to add summary:")
    if len(rows_to_add_series) > 0:
        print(f"  Min rows to add: {rows_to_add_series.min()}")
        print(f"  Max rows to add: {rows_to_add_series.max()}")
        print(f"  Mean rows to add: {rows_to_add_series.mean():.2f}")
        print(f"  Total new rows to add: {rows_to_add_series.sum()}")
    else:
        print(f"  Min rows to add: 0")
        print(f"  Max rows to add: 0")
        print(f"  Mean rows to add: 0.00")
        print(f"  Total new rows to add: 0")
    
    # 5. Create new timepoint rows
    print(f"\n5. Creating new timepoint rows...")
    
    new_rows_list = []
    
    for patno, num_rows in rows_to_add.items():
        if num_rows > 0:
            # Get baseline data for this PATNO
            baseline_row = dfy_extended[dfy_extended['PATNO'] == patno].iloc[0]
            
            # Create new rows with proper time values
            for i in range(num_rows):
                new_row = baseline_row.copy()
                
                # Set time variables to NaN
                for col in ynum:
                    new_row[col] = np.nan
                
                # Set time variable to future timepoints with incremental tdelta
                current_max_time = dfy_extended[dfy_extended['PATNO'] == patno][tvar[0]].max()
                new_time = current_max_time + (i + 1) * tdelta
                new_row[tvar[0]] = new_time
                
                new_rows_list.append(new_row)
    
    # 6. Combine datasets
    print(f"\n6. Combining datasets...")
    
    if new_rows_list:
        new_rows_df = pd.DataFrame(new_rows_list)
        dfy_extended = pd.concat([dfy_extended, new_rows_df], ignore_index=True)
        print(f"Added {len(new_rows_list)} new rows")
    else:
        print("No new rows to add")
    
    # 6b. Ensure ALL participants have exactly kmax timepoints up to tmax
    print(f"\n6b. Ensuring ALL participants have exactly {kmax} timepoints...")
    
    additional_rows_list = []
    for patno in dfy_extended['PATNO'].unique():
        current_count = len(dfy_extended[dfy_extended['PATNO'] == patno])
        if current_count < kmax:
            # This participant needs more rows
            additional_rows_needed = kmax - current_count
            baseline_row = dfy_extended[dfy_extended['PATNO'] == patno].iloc[0]
            
            # Get the current max time for this participant
            participant_data = dfy_extended[dfy_extended['PATNO'] == patno]
            current_max_time = participant_data[tvar[0]].max()
            
            for i in range(additional_rows_needed):
                new_row = baseline_row.copy()
                
                # Set time variables to NaN
                for col in ynum:
                    new_row[col] = np.nan
                
                # Set time variable to future timepoints with incremental tdelta
                future_time = current_max_time + (i + 1) * tdelta
                new_row[tvar[0]] = future_time
                
                additional_rows_list.append(new_row)
            
            print(f"  PATNO {patno}: Added {additional_rows_needed} rows (current: {current_count}, target: {kmax})")
    
    # Add the additional rows if any
    if additional_rows_list:
        additional_rows_df = pd.DataFrame(additional_rows_list)
        dfy_extended = pd.concat([dfy_extended, additional_rows_df], ignore_index=True)
        print(f"Added {len(additional_rows_list)} additional rows to ensure kmax consistency")
    
    # 7. Sort dataset
    print(f"\n7. Sorting dataset...")
    dfy_extended = dfy_extended.sort_values(['PATNO', tvar[0]]).reset_index(drop=True)
    
    # 8. Verify results
    print(f"\n8. Verifying results...")
    
    # Check final timepoint counts
    final_time_counts = dfy_extended.groupby('PATNO')[tvar[0]].count()
    
    print(f"Final verification:")
    print(f"  All PATNOs have exactly {kmax} timepoints: {all(final_time_counts == kmax)}")
    print(f"  Min final max time: {dfy_extended[tvar[0]].min():.2f}")
    print(f"  Max final max time: {dfy_extended[tvar[0]].max():.2f}")
    print(f"  Total observations: {len(dfy_extended):,}")
    print(f"  Total PATNOs: {dfy_extended['PATNO'].nunique()}")
    
    # Check if all targets achieved
    targets_achieved = all(final_time_counts == kmax)
    
    if targets_achieved:
        print(f"✅ All targets achieved:")
        print(f"   - All PATNOs have exactly {kmax} timepoints")
    else:
        print(f"⚠️  Some targets not achieved")
    
    # 9. Outcome variable summary
    print(f"\n9. Outcome variable summary:")
    for outcome in ynum:
        nan_count = dfy_extended[outcome].isna().sum()
        total_count = len(dfy_extended)
        print(f"  {outcome}: {nan_count:,} NaN values ({nan_count/total_count*100:.1f}%)")
    
    print(f"\n=== Testing Dataset Extension Complete ===")
    print(f"Original shape: {dfy_with_baseline.shape}")
    print(f"Extended shape: {dfy_extended.shape}")
    print(f"Rows added: {len(dfy_extended) - len(dfy_with_baseline):,}")
    print(f"Kmax: {kmax}")
    
    return dfy_extended


def impute_testing_data_with_mmrm_models(dfy_extended, ynum, ivar, Xt, fitted_models_path="./fittedModels"):
    """
    Load saved MMRM models and use them to impute missing ynum values in testing data.
    This function is similar to perform_mmrm_analysis but only loads models and imputes values.
    
    Parameters:
    -----------
    dfy_extended : pandas.DataFrame
        Extended testing dataset with missing ynum values
    ynum : list
        List of outcome variables to impute
    ivar : list
        List of identifier variables (e.g., ['PATNO'])
    Xt : list
        List of time-dependent variables (e.g., ['LEDD', 'year'])
    fitted_models_path : str
        Path to the fittedModels directory (default: "/Users/xiaolongluo/Documents/PPMI/fittedModels")
        
    Returns:
    --------
    dfy_with_imputed : pandas.DataFrame
        Testing dataset with imputed ynum values
    """
    print(f"\n=== Imputing Testing Data with Saved MMRM Models ===")
    print(f"Outcome variables: {ynum}")
    print(f"Identifier variable: {ivar}")
    print(f"Time-dependent variables: {Xt}")
    
    # Create a copy to avoid modifying original
    dfy_with_imputed = dfy_extended.copy()
    
    # Load saved MMRM models
    print("\nLoading saved MMRM models...")
    mmrm_models = {}
    
    for outcome in ynum:
        model_path = os.path.join(fitted_models_path, f'mmrm_model_{outcome}.pkl')
        if os.path.exists(model_path):
            mmrm_models[outcome] = joblib.load(model_path)
            print(f"  Loaded MMRM model for {outcome}")
            
            # Extract model information from the underlying model object
            if hasattr(mmrm_models[outcome], 'model'):
                underlying_model = mmrm_models[outcome].model
                
                # Get formula
                if hasattr(underlying_model, 'formula'):
                    print(f"    Formula: {underlying_model.formula}")
                
                # Get feature names (excluding Intercept)
                if hasattr(underlying_model, 'exog_names'):
                    feature_names = [name for name in underlying_model.exog_names if name != 'Intercept']
                    print(f"    Features: {feature_names}")
                
                # Get outcome names
                if hasattr(underlying_model, 'endog_names'):
                    print(f"    Outcome: {underlying_model.endog_names}")
                
                # Get model statistics
                if hasattr(mmrm_models[outcome], 'nobs'):
                    print(f"    Observations: {mmrm_models[outcome].nobs}")
                if hasattr(mmrm_models[outcome], 'aic'):
                    print(f"    AIC: {mmrm_models[outcome].aic:.2f}")
                if hasattr(mmrm_models[outcome], 'bic'):
                    print(f"    BIC: {mmrm_models[outcome].bic:.2f}")
            else:
                print(f"    Warning: Could not access underlying model object")

        else:
            print(f"  Warning: MMRM model for {outcome} not found at {model_path}")
            mmrm_models[outcome] = None
    
    # Check which models were successfully loaded
    loaded_models = {k: v for k, v in mmrm_models.items() if v is not None}
    print(f"\nSuccessfully loaded {len(loaded_models)} MMRM models")
    
    if not loaded_models:
        print("No MMRM models loaded. Cannot perform imputation.")
        return dfy_with_imputed
    
    # Prepare data for imputation
    print("\nPreparing data for imputation...")
    
    # Select key covariates for imputation (same as development data)
    # Note: Column names may have been renamed during development data preprocessing
    # We'll dynamically select available columns to avoid missing column errors
    base_covariates = ['LEDD', 'year', 'ENROLL_AGE', 'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT', 'SEX_10']
    cohort_covariates = ['COHORT_DEFINITION_encoded_Parkinsons_Disease',  # Without apostrophe (model expects this)
                         'COHORT_DEFINITION_encoded_Prodromal', 
                         'COHORT_DEFINITION_encoded_SWEDD']
    
    # Start with base covariates
    key_covariates = base_covariates.copy()
    
    # Add available cohort covariates
    for col in cohort_covariates:
        if col in dfy_with_imputed.columns:
            key_covariates.append(col)
            print(f"  Added cohort covariate: {col}")
        else:
            print(f"  Note: {col} not available in testing data")
    
    # Filter to only include columns that exist in dfy_with_imputed
    available_covariates = [col for col in key_covariates if col in dfy_with_imputed.columns]
    print(f"Available covariates for imputation: {available_covariates}")
    
    # Perform imputation for each outcome variable
    for outcome in ynum:
        if outcome not in loaded_models:
            print(f"\n--- Skipping {outcome} (model not loaded) ---")
            continue
            
        print(f"\n--- Imputing {outcome} ---")
        
        # Get the fitted model
        fitted_model = loaded_models[outcome]
        
        # Check if model has predict method
        if not hasattr(fitted_model, 'predict'):
            print(f"  Warning: Model for {outcome} does not have predict method")
            continue
        
        # Create NaN indicator column
        nan_col = f"{outcome}_is_nan"
        dfy_with_imputed[nan_col] = dfy_with_imputed[outcome].isna().astype(int)
        
        # Count missing values
        total_count = len(dfy_with_imputed)
        missing_count = dfy_with_imputed[outcome].isna().sum()
        print(f"  Missing values: {missing_count:,} ({missing_count/total_count*100:.1f}%)")
        
        if missing_count == 0:
            print(f"  No missing values to impute for {outcome}")
            continue
        
        # Prepare data for prediction
        try:
            # Select only rows with available covariates (excluding PATNO from covariates for prediction)
            prediction_covariates = [col for col in available_covariates if col != 'PATNO']
            analysis_data = dfy_with_imputed[prediction_covariates + [outcome]].dropna()
            
            if len(analysis_data) == 0:
                print(f"  No complete cases for {outcome} imputation")
                continue
            
            print(f"  Analysis sample size: {len(analysis_data):,} observations")
            print(f"  Prediction covariates: {prediction_covariates}")
            
            # Use the fitted model to predict for all data points (including those with NaN outcomes)
            # We need to handle the case where some covariates might be missing
            dfy_with_imputed_clean = dfy_with_imputed[prediction_covariates].fillna(0)  # Fill missing covariates with 0
            
            try:
                predictions = fitted_model.predict(dfy_with_imputed_clean)
                print(f"  Successfully generated predictions for {outcome}")
                
                # Create imputed outcome column
                imputed_col = f"{outcome}_imputed"
                dfy_with_imputed[imputed_col] = dfy_with_imputed[outcome].copy()
                
                # Fill NaN values with predictions
                nan_mask = dfy_with_imputed[outcome].isna()
                dfy_with_imputed.loc[nan_mask, imputed_col] = predictions[nan_mask]
                
                # Count imputed values
                imputed_count = nan_mask.sum()
                print(f"  Imputed values: {imputed_count:,} ({imputed_count/total_count*100:.1f}%)")
                
            except Exception as pred_error:
                print(f"  Prediction failed: {str(pred_error)}")
                # Create imputed column with original values
                imputed_col = f"{outcome}_imputed"
                dfy_with_imputed[imputed_col] = dfy_with_imputed[outcome].copy()
                
        except Exception as e:
            print(f"  Error preparing data for {outcome}: {str(e)}")
            # Create imputed column with original values
            imputed_col = f"{outcome}_imputed"
            dfy_with_imputed[imputed_col] = dfy_with_imputed[outcome].copy()
    
    # Summary of results
    print(f"\n=== MMRM Imputation Summary ===")
    print(f"Original dataset shape: {dfy_extended.shape}")
    print(f"Final dataset shape: {dfy_with_imputed.shape}")
    
    # Count new columns added
    new_columns = []
    for outcome in ynum:
        new_columns.extend([f"{outcome}_is_nan", f"{outcome}_imputed"])
    
    print(f"New columns added: {len(new_columns)}")
    print(f"New columns: {new_columns}")
    
    # Summary of imputed values
    print(f"\nImputation Summary:")
    for outcome in ynum:
        if f"{outcome}_imputed" in dfy_with_imputed.columns:
            original_nan = dfy_extended[outcome].isna().sum()
            final_nan = dfy_with_imputed[f"{outcome}_imputed"].isna().sum()
            imputed_count = original_nan - final_nan
            print(f"  {outcome}: {imputed_count:,} values imputed")
        else:
            print(f"  {outcome}: ERROR - imputed column not found!")
    
    # Debug: Check if imputed columns exist
    print(f"\nDebug - Checking imputed columns:")
    for outcome in ynum:
        imputed_col = f"{outcome}_imputed"
        if imputed_col in dfy_with_imputed.columns:
            print(f"  ✓ {imputed_col} exists with {dfy_with_imputed[imputed_col].notna().sum():,} non-null values")
        else:
            print(f"  ✗ {imputed_col} NOT FOUND")
    
    print(f"\n=== Final Dataset with MMRM Imputation ===")
    print(f"Final dataset shape: {dfy_with_imputed.shape}")
    print(f"Dataset ready for complete analysis")
    
    return dfy_with_imputed


def load_transformer_model(fitted_models_path="./fittedModels"):
    """
    Load the pre-trained transformer model from the fittedModels folder.
    
    Parameters:
    -----------
    fitted_models_path : str
        Path to the fittedModels directory (default: "/Users/xiaolongluo/Documents/PPMI/fittedModels")
    
    Returns:
        tf.keras.Model: The loaded transformer model
    """
    try:
        model_path = os.path.join(fitted_models_path, 'ppmi_transformer_model_final.keras')
        print(f"Loading transformer model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to load the model
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Successfully loaded transformer model")
            print(f"Model summary:")
            model.summary()
            return model
            
        except Exception as load_error:
            if "PPMIDTransformer" in str(load_error):
                if PPMIDTransformer is not None:
                    print(f"⚠️  PPMIDTransformer class is imported but model loading still failed")
                    print(f"Trying to load with custom_objects parameter...")
                    try:
                        # Try loading with custom_objects
                        model = tf.keras.models.load_model(
                            model_path, 
                            custom_objects={'PPMIDTransformer': PPMIDTransformer}
                        )
                        print(f"✅ Successfully loaded transformer model with custom_objects")
                        print(f"Model summary:")
                        model.summary()
                        return model
                    except Exception as e2:
                        print(f"❌ Failed to load even with custom_objects: {e2}")
                        return None
                else:
                    print(f"❌ Custom class 'PPMIDTransformer' not found in current environment")
                    print(f"This is a common issue when loading custom Keras models.")
                    print(f"\nTo resolve this issue, you need to:")
                    print(f"1. Import the PPMIDTransformer class definition")
                    print(f"2. Or register it with @keras.saving.register_keras_serializable()")
                    print(f"3. Or load the model in the same environment where it was trained")
                    print(f"\nAvailable transformer models in fittedModels:")
                    print(f"  - ppmi_transformer_model_final.keras")
                    print(f"  - ppmi_transformer_model_iteration_1.keras")
                    print(f"  - ppmi_transformer_model_iteration_2.keras")
                    print(f"  - ppmi_transformer_model_iteration_3.keras")
                    print(f"  - ppmi_transformer_model.keras")
                    return None
            else:
                raise load_error
        
    except Exception as e:
        print(f"❌ Error loading transformer model: {e}")
        return None


def calculate_r2_scores(df1_original_scale, ynum):
    """
    Calculate R² scores for each outcome variable using sklearn.metrics.r2_score.
    Filters to year > 0 and drops NaN values before calculation.
    
    Args:
        df1_original_scale: pandas.DataFrame with original scale predictions
        ynum: list of outcome variable names (e.g., ['NP1PTOT', 'NP2PTOT', 'NP3TOT'])
    
    Returns:
        dict: Dictionary with R² scores for each outcome variable
    """
    from sklearn.metrics import r2_score
    
    print(f"\n=== Calculating R² Scores ===")
    print(f"Outcome variables: {ynum}")
    print(f"Available columns: {list(df1_original_scale.columns)}")
    print(f"Dataframe shape: {df1_original_scale.shape}")
    
    r2_scores = {}
    
    # Filter to year > 0
    for y in ynum:
        df_sub = df1_original_scale.loc[df1_original_scale.year > 0, [y, y+'_pred']].dropna()
        
        if len(df_sub) == 0:
            print(f"⚠️  No valid data for {y} (year > 0)")
            r2_scores[y] = None
            continue
        
        # Extract arrays
        y_true = df_sub[y].values
        y_pred = df_sub[y+'_pred'].values
        
        # Debug: Print some statistics about the data
        print(f"Debug for {y}:")
        print(f"  y_true range: [{y_true.min():.2f}, {y_true.max():.2f}], mean: {y_true.mean():.2f}")
        print(f"  y_pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}], mean: {y_pred.mean():.2f}")
        print(f"  y_true std: {y_true.std():.2f}, y_pred std: {y_pred.std():.2f}")
        
        # R²
        r2 = r2_score(y_true, y_pred)
        r2_scores[y] = r2
        
        print(f"R² for {y}: {r2:.4f} (n={len(df_sub)})")
    
    # Print summary
    print(f"\n=== R² Summary ===")
    valid_scores = {k: v for k, v in r2_scores.items() if v is not None}
    if valid_scores:
        mean_r2 = sum(valid_scores.values()) / len(valid_scores)
        print(f"Mean R²: {mean_r2:.4f}")
        print(f"Valid scores: {len(valid_scores)}/{len(ynum)}")
    else:
        print("No valid R² scores calculated")
    
    return r2_scores

def plot_random_patno_predictions(df1_original_scale, ynum, Xt=['LEDD', 'year']):
    """
    Pick a random PATNO and plot year vs original y values and predictions with confidence intervals.
    
    Parameters:
    -----------
    df1_original_scale : pandas.DataFrame
        DataFrame with original scale predictions and confidence intervals
    ynum : list
        List of outcome variable names (e.g., ['NP1PTOT', 'NP2PTOT', 'NP3TOT'])
    Xt : list
        List of time-dependent variables (default: ['LEDD', 'year'])
        
    Returns:
    --------
    selected_patno : str
        The PATNO that was selected for plotting
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"\n=== Plotting Random PATNO Predictions ===")
    print(f"Outcome variables: {ynum}")
    print(f"Time variable: {Xt[-1]} (year)")
    
    # Get random PATNO
    available_patnos = df1_original_scale['PATNO'].unique()
    selected_patno = np.random.choice(available_patnos)
    
    print(f"Selected PATNO: {selected_patno}")
    
    # Filter data for selected PATNO
    patno_data = df1_original_scale[df1_original_scale['PATNO'] == selected_patno].copy()
    patno_data = patno_data.sort_values(Xt[-1])  # Sort by year
    
    print(f"Data points for PATNO {selected_patno}: {len(patno_data)}")
    print(f"Year range: {patno_data[Xt[-1]].min():.1f} to {patno_data[Xt[-1]].max():.1f}")
    
    # Create subplots
    n_outcomes = len(ynum)
    fig, axes = plt.subplots(1, n_outcomes, figsize=(6*n_outcomes, 5))
    
    # If only one outcome, make axes iterable
    if n_outcomes == 1:
        axes = [axes]
    
    # Get cohort information for title
    cohort = patno_data['COHORT_DEFINITION'].iloc[0] if 'COHORT_DEFINITION' in patno_data.columns else 'Unknown'
    
    # Plot each outcome variable
    for i, y in enumerate(ynum):
        ax = axes[i]
        
        # Get data for this outcome
        year_data = patno_data[Xt[-1]].values
        y_original = patno_data[y].values
        y_pred = patno_data[y + '_pred'].values
        y_lower = patno_data[y + '_lower'].values
        y_upper = patno_data[y + '_upper'].values
        
        # Plot original values (where not NaN)
        original_mask = ~np.isnan(y_original)
        if original_mask.any():
            ax.scatter(year_data[original_mask], y_original[original_mask], 
                      color='blue', alpha=0.7, s=60, label='Original', zorder=3)
            ax.plot(year_data[original_mask], y_original[original_mask], 
                   color='blue', alpha=0.5, linewidth=2, zorder=2)
        
        # Plot predictions
        pred_mask = ~np.isnan(y_pred)
        if pred_mask.any():
            ax.plot(year_data[pred_mask], y_pred[pred_mask], 
                   color='red', alpha=0.8, linewidth=2, label='Predicted', zorder=2)
        
        # Plot confidence intervals
        ci_mask = ~(np.isnan(y_lower) | np.isnan(y_upper))
        if ci_mask.any():
            ax.fill_between(year_data[ci_mask], y_lower[ci_mask], y_upper[ci_mask], 
                           color='red', alpha=0.2, label='95% CI', zorder=1)
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'{y} Score', fontsize=12)
        ax.set_title(f'{y} - PATNO {selected_patno}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        n_original = original_mask.sum()
        n_pred = pred_mask.sum()
        ax.text(0.02, 0.98, f'Original: {n_original}\nPredicted: {n_pred}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    # Overall title
    fig.suptitle(f'Predictions for PATNO {selected_patno} ({cohort})', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Summary for PATNO {selected_patno} ===")
    print(f"Cohort: {cohort}")
    print(f"Total observations: {len(patno_data)}")
    print(f"Year range: {patno_data[Xt[-1]].min():.1f} to {patno_data[Xt[-1]].max():.1f}")
    
    for y in ynum:
        original_count = patno_data[y].notna().sum()
        pred_count = patno_data[y + '_pred'].notna().sum()
        print(f"{y}: {original_count} original, {pred_count} predicted")
    
    return selected_patno


def reverse_scale_predictions(df1, fitted_models_path="./fittedModels"):
    """
    Load the baseline_Xnum_scaler.pkl and dfy_scaler.pkl from fittedModels and reverse scale 
    the data back to the original scale.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        DataFrame containing the predictions and confidence intervals
    fitted_models_path : str
        Path to the fittedModels directory (default: "/Users/xiaolongluo/Documents/PPMI/fittedModels")
        
    Returns:
    --------
    df1_reversed : pandas.DataFrame
        DataFrame with data reverse scaled back to original scale
    """
    import joblib
    import os
    import numpy as np
    
    print(f"\n=== Reverse Scaling Predictions to Original Scale ===")
    print(f"Fitted models path: {fitted_models_path}")
    print(f"Input dataframe shape: {df1.shape}")
    
    # Create a copy to avoid modifying original
    df1_reversed = df1.copy()
    
    # 1. Load baseline_Xnum_scaler and reverse scale Xnum columns
    print(f"\n1. Loading baseline_Xnum_scaler and reverse scaling Xnum columns...")
    
    baseline_scaler_path = os.path.join(fitted_models_path, 'baseline_Xnum_scaler.pkl')
    if os.path.exists(baseline_scaler_path):
        baseline_scaler = joblib.load(baseline_scaler_path)
        print(f"  ✅ Loaded baseline_Xnum_scaler")
        
        # Define Xnum columns that should be reverse scaled
        Xnum = ['ENROLL_AGE', 'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT']
        
        # Filter to only include columns that exist in the dataframe
        available_Xnum = [col for col in Xnum if col in df1_reversed.columns]
        
        if available_Xnum:
            print(f"  Reverse scaling columns: {available_Xnum}")
            # Apply inverse transform to reverse scale back to original scale
            df1_reversed[available_Xnum] = baseline_scaler.inverse_transform(df1_reversed[available_Xnum])
            print(f"  ✅ Successfully reverse scaled Xnum columns")
        else:
            print(f"  ⚠️  No Xnum columns found in dataframe")
    else:
        print(f"  ❌ baseline_Xnum_scaler.pkl not found at {baseline_scaler_path}")
    
    # 2. Load dfy_scaler and reverse scale ynum and prediction columns
    print(f"\n2. Loading dfy_scaler and reverse scaling ynum and prediction columns...")
    
    dfy_scaler_path = os.path.join(fitted_models_path, 'dfy_scaler.pkl')
    if os.path.exists(dfy_scaler_path):
        dfy_scaler = joblib.load(dfy_scaler_path)
        print(f"  ✅ Loaded dfy_scaler")
        print(f"  Scaler was fitted on: ynum + Xt[:-1] (4 columns total)")
        
        # Define column groups
        ynum = ['NP1PTOT', 'NP2PTOT', 'NP3TOT']
        Xt_minus_1 = ['LEDD']  # Xt[:-1] is just LEDD (excluding 'year')
        prediction_columns = [y + '_pred' for y in ynum]
        lower_columns = [y + '_lower' for y in ynum]
        upper_columns = [y + '_upper' for y in ynum]
        
        # 2a. First, scale back Xt[:-1] using the full scaler (4th component)
        print(f"\n  2a. Reverse scaling Xt[:-1] (LEDD) using full scaler...")
        available_Xt = [col for col in Xt_minus_1 if col in df1_reversed.columns]
        if available_Xt:
            # Create a temporary array with zeros for ynum columns and actual values for Xt[:-1]
            temp_array = np.zeros((len(df1_reversed), 4))  # 4 columns: 3 ynum + 1 Xt[:-1]
            temp_array[:, 3] = df1_reversed[available_Xt[0]].values  # Put LEDD in 4th position
            
            # Apply inverse transform
            temp_scaled = dfy_scaler.inverse_transform(temp_array)
            
            # Extract only the Xt[:-1] component (4th column)
            df1_reversed[available_Xt[0]] = temp_scaled[:, 3]
            print(f"    ✅ Successfully reverse scaled {available_Xt}")
        else:
            print(f"    ⚠️  No Xt[:-1] columns found in dataframe")
        
        # 2b. Scale back ynum and predictions using only the first 3 components of the scaler
        print(f"\n  2b. Reverse scaling ynum and predictions using first 3 scaler components...")
        
        # Create a partial scaler for ynum only (first 3 components)
        from sklearn.preprocessing import StandardScaler
        ynum_scaler = StandardScaler()
        ynum_scaler.mean_ = dfy_scaler.mean_[:3]  # First 3 means
        ynum_scaler.scale_ = dfy_scaler.scale_[:3]  # First 3 scales
        ynum_scaler.var_ = dfy_scaler.var_[:3]  # First 3 variances
        ynum_scaler.n_features_in_ = 3
        ynum_scaler.feature_names_in_ = dfy_scaler.feature_names_in_[:3] if hasattr(dfy_scaler, 'feature_names_in_') else None
        
        # Define column groups for ynum and predictions
        column_groups = [
            ('ynum (original)', ynum),
            ('predictions', prediction_columns),
            ('lower confidence intervals', lower_columns),
            ('upper confidence intervals', upper_columns)
        ]
        
        # Apply inverse transform to each group using the ynum-only scaler
        for group_name, columns in column_groups:
            # Filter to only include columns that exist in the dataframe
            available_columns = [col for col in columns if col in df1_reversed.columns]
            
            if available_columns:
                print(f"    Reverse scaling {group_name}: {available_columns}")
                
                # Apply inverse transform using the ynum-only scaler
                df1_reversed[available_columns] = ynum_scaler.inverse_transform(df1_reversed[available_columns])
                print(f"    ✅ Successfully reverse scaled {group_name}")
                
                # Print summary of reverse scaled values for this group
                print(f"      Summary of {group_name}:")
                for col in available_columns:
                    if col in df1_reversed.columns:
                        col_min = df1_reversed[col].min()
                        col_max = df1_reversed[col].max()
                        col_mean = df1_reversed[col].mean()
                        print(f"        {col}: min={col_min:.2f}, max={col_max:.2f}, mean={col_mean:.2f}")
            else:
                print(f"    ⚠️  No {group_name} columns found in dataframe")
    else:
        print(f"  ❌ dfy_scaler.pkl not found at {dfy_scaler_path}")
    
    # 3. Print final summary
    print(f"\n=== Reverse Scaling Summary ===")
    print(f"Original dataframe shape: {df1.shape}")
    print(f"Reverse scaled dataframe shape: {df1_reversed.shape}")
    print(f"✅ Data successfully reverse scaled back to original scale")
    
    return df1_reversed


def get_prediction(df, X_tensor, y_tensor, model, ynum):

    
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    
    
    print(f"Original dataframe shape: {df.shape}")
    print(f"X_tensor shape: {X_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")
    
    # Get unique PATNOs from the original dataframe
    unique_patnos = df['PATNO'].unique()
    print(f"Number of unique PATNOs: {len(unique_patnos)}")
    
    # Derive predictions and confidence limits as specified
    print("Deriving predictions and confidence limits...")
    
    # X_context = X_tensor[:, :-1, :] (exclude last timepoint)
    X_context = X_tensor[:, :-1, :]
    print(f"X_context shape: {X_context.shape}")
    
    # y_history = y_tensor[:, :-1, :] (exclude last timepoint)
    y_history = y_tensor[:, :-1, :]
    print(f"y_history shape: {y_history.shape}")
    
    # Get predictions from model
    mean_pred, lower, upper = model((X_context, y_history), training=False)
    print(f"mean_pred shape: {mean_pred.shape}")
    print(f"lower shape: {lower.shape}")
    print(f"upper shape: {upper.shape}")
    
    # Create y_tensor_pred by concatenating baseline with predictions
    y_tensor_pred = tf.concat([tf.expand_dims(y_tensor[:, 0, :], axis=1), mean_pred], axis=1)
    print(f"y_tensor_pred shape: {y_tensor_pred.shape}")
    
    # Create lower_pred and upper_pred by concatenating baseline with confidence limits
    # Note: We'll use the baseline values for the first timepoint since we don't have confidence intervals for baseline
    lower_pred = tf.concat([tf.expand_dims(y_tensor[:, 0, :], axis=1), lower], axis=1)
    upper_pred = tf.concat([tf.expand_dims(y_tensor[:, 0, :], axis=1), upper], axis=1)
    print(f"lower_pred shape: {lower_pred.shape}")
    print(f"upper_pred shape: {upper_pred.shape}")

    reshaped = tf.reshape(y_tensor_pred, [-1, y_tensor_pred.shape[-1]])
    df_pred=pd.DataFrame( reshaped, columns=[y+'_pred' for y in ynum])


    reshaped = tf.reshape(lower_pred, [-1, lower_pred.shape[-1]])
    lower_pred=pd.DataFrame( reshaped, columns=[y+'_lower' for y in ynum])


    reshaped = tf.reshape(upper_pred, [-1, upper_pred.shape[-1]])
    upper_pred=pd.DataFrame( reshaped, columns=[y+'_upper' for y in ynum])

    df1=pd.concat([df,df_pred,lower_pred,upper_pred],axis=1)
    return df1



def reverse_scale_predictions(df1, fitted_models_path="./fittedModels"):
    """
    Load the baseline_Xnum_scaler.pkl and dfy_scaler.pkl from fittedModels and reverse scale 
    the data back to the original scale.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        DataFrame containing the predictions and confidence intervals
    fitted_models_path : str
        Path to the fittedModels directory (default: "/Users/xiaolongluo/Documents/PPMI/fittedModels")
        
    Returns:
    --------
    df1_reversed : pandas.DataFrame
        DataFrame with data reverse scaled back to original scale
    """
    import joblib
    import os
    import numpy as np
    
    print(f"\n=== Reverse Scaling Predictions to Original Scale ===")
    print(f"Fitted models path: {fitted_models_path}")
    print(f"Input dataframe shape: {df1.shape}")
    
    # Create a copy to avoid modifying original
    df1_reversed = df1.copy()
    
    # 1. Load baseline_Xnum_scaler and reverse scale Xnum columns
    print(f"\n1. Loading baseline_Xnum_scaler and reverse scaling Xnum columns...")
    
    baseline_scaler_path = os.path.join(fitted_models_path, 'baseline_Xnum_scaler.pkl')
    if os.path.exists(baseline_scaler_path):
        baseline_scaler = joblib.load(baseline_scaler_path)
        print(f"  ✅ Loaded baseline_Xnum_scaler")
        
        # Define Xnum columns that should be reverse scaled
        Xnum = ['ENROLL_AGE', 'b_NP1PTOT', 'b_NP2PTOT', 'b_NP3TOT']
        
        # Filter to only include columns that exist in the dataframe
        available_Xnum = [col for col in Xnum if col in df1_reversed.columns]
        
        if available_Xnum:
            print(f"  Reverse scaling columns: {available_Xnum}")
            # Apply inverse transform to reverse scale back to original scale
            df1_reversed[available_Xnum] = baseline_scaler.inverse_transform(df1_reversed[available_Xnum])
            print(f"  ✅ Successfully reverse scaled Xnum columns")
        else:
            print(f"  ⚠️  No Xnum columns found in dataframe")
    else:
        print(f"  ❌ baseline_Xnum_scaler.pkl not found at {baseline_scaler_path}")
    
    # 2. Load dfy_scaler and reverse scale ynum and prediction columns
    print(f"\n2. Loading dfy_scaler and reverse scaling ynum and prediction columns...")
    
    dfy_scaler_path = os.path.join(fitted_models_path, 'dfy_scaler.pkl')
    if os.path.exists(dfy_scaler_path):
        dfy_scaler = joblib.load(dfy_scaler_path)
        print(f"  ✅ Loaded dfy_scaler")
        print(f"  Scaler was fitted on: ynum + Xt[:-1] (4 columns total)")
        
        # Define column groups
        ynum = ['NP1PTOT', 'NP2PTOT', 'NP3TOT']
        Xt_minus_1 = ['LEDD']  # Xt[:-1] is just LEDD (excluding 'year')
        prediction_columns = [y + '_pred' for y in ynum]
        lower_columns = [y + '_lower' for y in ynum]
        upper_columns = [y + '_upper' for y in ynum]
        
        # 2a. First, scale back Xt[:-1] using the full scaler (4th component)
        print(f"\n  2a. Reverse scaling Xt[:-1] (LEDD) using full scaler...")
        available_Xt = [col for col in Xt_minus_1 if col in df1_reversed.columns]
        if available_Xt:
            # Create a temporary array with zeros for ynum columns and actual values for Xt[:-1]
            temp_array = np.zeros((len(df1_reversed), 4))  # 4 columns: 3 ynum + 1 Xt[:-1]
            temp_array[:, 3] = df1_reversed[available_Xt[0]].values  # Put LEDD in 4th position
            
            # Apply inverse transform
            temp_scaled = dfy_scaler.inverse_transform(temp_array)
            
            # Extract only the Xt[:-1] component (4th column)
            df1_reversed[available_Xt[0]] = temp_scaled[:, 3]
            print(f"    ✅ Successfully reverse scaled {available_Xt}")
        else:
            print(f"    ⚠️  No Xt[:-1] columns found in dataframe")
        
        # 2b. Scale back ynum and predictions using only the first 3 components of the scaler
        print(f"\n  2b. Reverse scaling ynum and predictions using first 3 scaler components...")
        
        # Create a partial scaler for ynum only (first 3 components)
        from sklearn.preprocessing import StandardScaler
        ynum_scaler = StandardScaler()
        ynum_scaler.mean_ = dfy_scaler.mean_[:3]  # First 3 means
        ynum_scaler.scale_ = dfy_scaler.scale_[:3]  # First 3 scales
        ynum_scaler.var_ = dfy_scaler.var_[:3]  # First 3 variances
        ynum_scaler.n_features_in_ = 3
        ynum_scaler.feature_names_in_ = dfy_scaler.feature_names_in_[:3] if hasattr(dfy_scaler, 'feature_names_in_') else None
        
        # Define column groups for ynum and predictions
        column_groups = [
            ('ynum (original)', ynum),
            ('predictions', prediction_columns),
            ('lower confidence intervals', lower_columns),
            ('upper confidence intervals', upper_columns)
        ]
        
        # Apply inverse transform to each group using the ynum-only scaler
        for group_name, columns in column_groups:
            # Filter to only include columns that exist in the dataframe
            available_columns = [col for col in columns if col in df1_reversed.columns]
            
            if available_columns:
                print(f"    Reverse scaling {group_name}: {available_columns}")
                
                # Apply inverse transform using the ynum-only scaler
                df1_reversed[available_columns] = ynum_scaler.inverse_transform(df1_reversed[available_columns])
                print(f"    ✅ Successfully reverse scaled {group_name}")
                
                # Print summary of reverse scaled values for this group
                print(f"      Summary of {group_name}:")
                for col in available_columns:
                    if col in df1_reversed.columns:
                        col_min = df1_reversed[col].min()
                        col_max = df1_reversed[col].max()
                        col_mean = df1_reversed[col].mean()
                        print(f"        {col}: min={col_min:.2f}, max={col_max:.2f}, mean={col_mean:.2f}")
            else:
                print(f"    ⚠️  No {group_name} columns found in dataframe")
    else:
        print(f"  ❌ dfy_scaler.pkl not found at {dfy_scaler_path}")
    
    # 3. Print final summary
    print(f"\n=== Reverse Scaling Summary ===")
    print(f"Original dataframe shape: {df1.shape}")
    print(f"Reverse scaled dataframe shape: {df1_reversed.shape}")
    print(f"✅ Data successfully reverse scaled back to original scale")
    
    return df1_reversed

def calculate_r2_scores(df, ynum):
    """
    Calculate R² scores for model evaluation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with original and predicted values
    ynum : list
        List of outcome variable names
        
    Returns:
    --------
    dict
        Dictionary with R² scores for each outcome
    """
    print(f"\n=== Calculating R² Scores ===")
    
    from sklearn.metrics import r2_score
    
    r2_scores = {}
    
    for outcome in ynum:
        original_col = outcome
        pred_col = f"{outcome}_pred"
        
        if original_col in df.columns and pred_col in df.columns:
            # Get non-null values for comparison
            mask = df[original_col].notna() & df[pred_col].notna()
            
            if mask.sum() > 0:
                y_true = df.loc[mask, original_col]
                y_pred = df.loc[mask, pred_col]
                
                r2 = r2_score(y_true, y_pred)
                r2_scores[outcome] = r2
                
                print(f"{outcome}: R² = {r2:.4f} (n = {mask.sum()})")
            else:
                print(f"⚠️  No valid data for {outcome}")
                r2_scores[outcome] = None
        else:
            print(f"⚠️  Missing columns for {outcome}")
            r2_scores[outcome] = None
    
    print(f"✅ R² calculation complete")
    return r2_scores


def plot_random_patno_predictions(df, ynum, Xt, save_path=None, show_plot=False):
    """
    Plot predictions for a random PATNO.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with predictions
    ynum : list
        List of outcome variable names
    Xt : list
        List of time-dependent variables
    save_path : str, optional
        Path to save the plot
    show_plot : bool, default=False
        Whether to display the plot
        
    Returns:
    --------
    int
        The selected PATNO
    """
    print(f"\n=== Plotting Random PATNO Predictions ===")
    
    # Select a random PATNO
    available_patnos = df['PATNO'].unique()
    selected_patno = np.random.choice(available_patnos)
    
    print(f"Selected PATNO: {selected_patno}")
    
    # Filter data for selected PATNO
    patno_data = df[df['PATNO'] == selected_patno].copy()
    patno_data = patno_data.sort_values('year')
    
    # Create subplots
    n_outcomes = len(ynum)
    fig, axes = plt.subplots(1, n_outcomes, figsize=(5*n_outcomes, 6))
    if n_outcomes == 1:
        axes = [axes]
    
    fig.suptitle(f'Predictions vs Original Values for PATNO {selected_patno}', 
                 fontsize=16, fontweight='bold')
    
    for i, outcome in enumerate(ynum):
        ax = axes[i]
        
        # Plot original values
        if outcome in patno_data.columns:
            original_mask = patno_data[outcome].notna()
            if original_mask.any():
                ax.scatter(patno_data.loc[original_mask, 'year'], 
                          patno_data.loc[original_mask, outcome], 
                          color='blue', alpha=0.7, s=60, label='Original', zorder=3)
        
        # Plot predicted values
        pred_col = f"{outcome}_pred"
        if pred_col in patno_data.columns:
            ax.plot(patno_data['year'], patno_data[pred_col], 
                   color='red', alpha=0.8, linewidth=2, label='Predicted', zorder=2)
        
        # Customize plot
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'{outcome} Score', fontsize=12)
        ax.set_title(f'{outcome}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Predictions plot saved to: {save_path}")
    
    # Close the figure to prevent display and free memory
    plt.close(fig)
    
    print(f"✅ Plot created for PATNO {selected_patno}")
    return selected_patno

