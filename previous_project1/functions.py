############################################################################################################
#----------------------------------- Deep Learning Project- Group 13 ---------------------------------------
#--Members: ------------------------------------------------------------------------------------------------
#--1. Afonso Dias - 20211540 -------------------------------------------------------------------------------
#--2. Inês Araújo - 20240532 -------------------------------------------------------------------------------
#--3. Isabel Duarte - 20240545 -----------------------------------------------------------------------------
#--4. Leonor Mira - 20240658 -------------------------------------------------------------------------------
#--5. Rita Matos - 20211642 --------------------------------------------------------------------------------
############################################################################################################

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from PIL import Image
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import re
import math

############################################################################################################
#------------------------------------------- General Functions ---------------------------------------------
############################################################################################################

#--------------------------------- Load Data ---------------------------------------------------------------

# Load data from directory and encode labels
def load_data(directory, encoder=None):
    # Create a list of (file_path, label) pairs
    file_paths_labels = [
        (os.path.join(directory, class_name, filename), class_name)
        for class_name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, class_name))                 # Only consider directories
        for filename in os.listdir(os.path.join(directory, class_name))
        if filename.endswith('.jpg')
    ]

    # Unpack the list of tuples into separate lists for paths and labels
    file_paths, class_labels = zip(*file_paths_labels) if file_paths_labels else ([], [])
    
    # Convert class labels to numpy array and reshape
    class_labels = np.array(class_labels).reshape(-1, 1)

    # If no encoder was passed, create, fit and transform one
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_labels = encoder.fit_transform(class_labels)   # Fit the encoder and transform labels
        
    # If an encoder was passed, just transform the labels
    else:
        one_hot_labels = encoder.transform(class_labels)

    return file_paths, one_hot_labels, encoder

#--------------------------------- Data Augmentation Functions ---------------------------------------------

def data_generator(preprocessing_function=None, augment=False):
    if augment:
        if preprocessing_function is None:
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=45,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            return ImageDataGenerator(
                preprocessing_function=preprocessing_function,         # Preprocess images using EfficientNet's preprocessing function
                rotation_range=45,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,                                  # Randomly flip images horizontally
                brightness_range=[0.8, 1.2],                           # Randomly adjust brightness between 80% and 120%
                fill_mode='nearest'                                    # Fill in new pixels after rotation or shift
            )
    else:
        if preprocessing_function is None:
            return ImageDataGenerator(rescale=1./255)
        else:
            return ImageDataGenerator(preprocessing_function=preprocessing_function)
        
#--------------------------------- Model definition and compilation functions ------------------------------

# Callback definition
def callbacks(model_name):
    # Directory where model checkpoints will be saved
    base_dir = "./Callbacks_256x192"
    full_path = os.path.join(base_dir, model_name)
    os.makedirs(full_path, exist_ok=True)

    # Save model weights only when validation F1 score improves
    checkpoint = ModelCheckpoint(
        monitor='val_f1_score',                         
        filepath=os.path.join(full_path, 'f1_best_model.weights.h5'),  # Filepath to save weights
        save_weights_only=True,                         # Only save the weights (not the full model)
        mode='max',                                     # Save when the F1 score is maximized
        verbose=1,                                      # Print message when saving
        save_best_only=True                             # Only save if it's the best so far
    )

    # Stop training early if no improvement in validation F1 score
    early_stop = EarlyStopping(
        patience=3,                                     # Wait 3 epochs with no improvement before stopping
        monitor='val_f1_score',                         
        verbose=1,                                      # Print when training stops
        mode='max'                                      # Look for maximum F1 score
    )

    # Reduce learning rate when validation F1 score plateaus
    lr_adjust = ReduceLROnPlateau(
        patience=2,                                     # Wait 2 epochs before reducing learning rate
        factor=0.75,                                    # Reduce learning rate by 25%
        min_lr=1e-4,                                    # Do not reduce below this minimum
        monitor='val_f1_score',                         
        mode='max',                                     # Expect F1 score to increase
        verbose=1                                       # Print when learning rate is reduced
    )

    return [checkpoint, early_stop, lr_adjust]

# F1 Score Metric

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Per-class metric accumulators
        self.tp = self.add_weight(name="true_positives", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="false_positives", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="false_negatives", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot to class labels
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)

        for i in range(self.num_classes):
            # Build binary masks for the current class
            true_mask = tf.cast(tf.equal(y_true, i), self.dtype)
            pred_mask = tf.cast(tf.equal(y_pred, i), self.dtype)

            # Compute TP, FP, FN
            tp_i = tf.reduce_sum(true_mask * pred_mask)
            fp_i = tf.reduce_sum((1 - true_mask) * pred_mask)
            fn_i = tf.reduce_sum(true_mask * (1 - pred_mask))

            # Update metrics for current class
            self.tp.assign(tf.tensor_scatter_nd_add(self.tp, [[i]], [tp_i]))
            self.fp.assign(tf.tensor_scatter_nd_add(self.fp, [[i]], [fp_i]))
            self.fn.assign(tf.tensor_scatter_nd_add(self.fn, [[i]], [fn_i]))

    def result(self):
        # Calculate precision and recall
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f1)

    def reset_states(self):
        # Reset the internal state of the metric
        for var in [self.tp, self.fp, self.fn]:
            var.assign(tf.zeros_like(var))

# Visualization of the training history

def model_progress(history):
    # Get the history data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    f1_score = history.history['f1_score']
    val_f1_score = history.history['val_f1_score']

    # Define colors
    light_green = '#90ee90'     
    dark_green = '#006400'      

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss', color=light_green)
    plt.plot(val_loss, label='Validation Loss', color=dark_green)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy', color=light_green)
    plt.plot(val_accuracy, label='Validation Accuracy', color=dark_green)
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1 Score
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(f1_score, label='Training F1 Score', color=light_green)
    plt.plot(val_f1_score, label='Validation F1 Score', color=dark_green)
    plt.title('F1 Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.show()
    
def metrics(rec, metric='f1_score'):
    # Get the index of the best epoch based on the specified metric
    metric_values = rec.history[metric]
    idx = np.argmax(metric_values)

    accuracy = rec.history["accuracy"][idx]
    f1 = rec.history["f1_score"][idx]
    loss = rec.history["loss"][idx]
    val_accuracy = rec.history["val_accuracy"][idx]
    val_f1 = rec.history["val_f1_score"][idx]
    val_loss = rec.history["val_loss"][idx]

    return accuracy, f1, loss, val_accuracy, val_f1, val_loss

# Model compilation
def model_compiler(model_instance, optimizer='adam', num_classes=202):
    model_instance.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score(num_classes=num_classes)]
    )
    return model_instance

# Confusion Matrix

def get_most_confused_classes(y_true, y_pred, class_names, top_n=20):
    cm = confusion_matrix(y_true, y_pred)
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    most_confused_indices = np.argsort(errors_per_class)[-top_n:]
    cm_subset = cm[np.ix_(most_confused_indices, most_confused_indices)]
    subset_labels = [class_names[i] for i in most_confused_indices]
    return cm_subset, subset_labels

def plot_confusion_subset(cm_subset, labels, title="Top Confused Classes"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_subset, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

############################################################################################################
#---------------------------------Functions used in Pre-processing Notebook---------------------------------
############################################################################################################

#--------------------------------- Resizing ----------------------------------------------------------------
# Function to resize and save an image
def resize_and_save_image(image_path, output_path, dimensions):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, dimensions)
    cv2.imwrite(output_path, img_resized)

# Function to automatically generate paths from dimensions
def resize_paths(dimensions):
    dim_str = f"{dimensions[0]}x{dimensions[1]}"
    resized_dir = f"../../data/resized_data_{dim_str}"
    output_dir = f"../../data/resized_data_{dim_str}_split"

    # Create the directories if they don't exist
    Path(resized_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return resized_dir, output_dir, dimensions

# Function to resize and save all images according to the dimensions specified
def resize_and_save_all_images(source_dir, resized_dir, dimensions):
    # Loop through all images in all class directories
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):
            # Create a corresponding folder in the resized directory
            resized_class_path = os.path.join(resized_dir, class_folder)
            Path(resized_class_path).mkdir(parents=True, exist_ok=True)

            for image_path in glob(os.path.join(class_path, '*.jpg')):
                # Create output path keeping the original image name
                image_name = os.path.basename(image_path)
                output_path = os.path.join(resized_class_path, image_name)
                resize_and_save_image(image_path, output_path, dimensions)

# Function to split the dataset into train, validation, and test sets
def split_resized_dataset(resized_dir, output_dir):
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    for directory in [train_dir, val_dir, test_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    for folder_name in os.listdir(resized_dir):
        folder_path = os.path.join(resized_dir, folder_name)
        if os.path.isdir(folder_path):
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
            train_paths, test_val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
            val_paths, test_paths = train_test_split(test_val_paths, test_size=0.5, random_state=42)

            for path, dest_dir in zip([train_paths, val_paths, test_paths], [train_dir, val_dir, test_dir]):
                dest_folder = os.path.join(dest_dir, folder_name)
                Path(dest_folder).mkdir(parents=True, exist_ok=True)
                for image_path in path:
                    shutil.copy(image_path, dest_folder)

# Function to plot 4 random images from 4 random families in the resized dataset
def plot_random_family_images(resized_dir, dimensions):
    families = random.sample(os.listdir(resized_dir), 4)

    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    for ax, family in zip(axes, families):
        family_path = os.path.join(resized_dir, family)
        images = [f for f in os.listdir(family_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not images:
            continue
        img_path = os.path.join(family_path, random.choice(images))
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(family, fontsize=12) 
        ax.axis('off')

    fig.suptitle(f"Random Families - {dimensions[0]}x{dimensions[1]}", fontsize=20, y=0.95)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85) 
    plt.show()

# Function to plot one image example per phylum from resized dataset
def plot_images_by_phylum(resized_dir, dimensions):
    resized_phylum_image_paths = {}
    for folder_name in os.listdir(resized_dir):
        folder_path = os.path.join(resized_dir, folder_name)
        if os.path.isdir(folder_path):
            phylum_name = folder_name.split('_')[0]
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if image_files and phylum_name not in resized_phylum_image_paths:
                resized_phylum_image_paths[phylum_name] = image_files[0]

    fig_size = (15, 10)
    fig, axes = plt.subplots(1, len(resized_phylum_image_paths), figsize=fig_size)
    if len(resized_phylum_image_paths) == 1:
        axes = [axes]

    for ax, (phylum, image_path) in zip(axes, resized_phylum_image_paths.items()):
        img = Image.open(image_path)
        size = img.size
        channels = len(img.getbands())
        ax.imshow(img)
        ax.set_title(f"{phylum}\nSize: {size}")
        ax.axis('off')

    plt.suptitle(f"Resized Images by Phylum - {dimensions[0]}x{dimensions[1]}", fontsize=20, y=0.75)
    plt.tight_layout()
    plt.show()

############################################################################################################
#---------------------------------Functions used in Pre-trained Model Notebook------------------------------
############################################################################################################

#--------------------------------- Data Augmentation Functions ---------------------------------------------  

def generator(datagen, directory):
    return datagen.flow_from_directory(
        directory,                                     # Path to the images
        target_size=(224, 224),                        # Resize to match EfficientNet input
        batch_size=16,                                 # Number of images to load per batch
        class_mode="categorical",                      # Use categorical labels for multi-class classification
    )

# Visualize the data augmentation
def visualize_augmentations(image_path, datagen, target_size=(224, 224), num_augmented=5):
    
    # Load and preprocess the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size)

    # Prepare the image for the generator
    image_batch = np.expand_dims(image_resized, axis=0)

    # Create a temporary generator
    temp_generator = datagen.flow(
        image_batch,
        batch_size=1
    )

    # Plot original + augmented images
    fig, axes = plt.subplots(1, num_augmented + 1, figsize=(15, 5))
    fig.suptitle('Original and Augmented Images', fontsize=16, y = 0.85)

    # Original image
    axes[0].imshow(image_resized)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Augmented images
    for i in range(1, num_augmented + 1):
        augmented_img = next(temp_generator)[0]  
        augmented_img_rescaled = (augmented_img + 1) * 127.5  
        axes[i].imshow(augmented_img_rescaled.astype(np.uint8))
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

############################################################################################################
#---------------------------------Functions used in Custom Model Notebook-----------------------------------
############################################################################################################

# First version of data augmentation (more aggressive)
def data_generator_2(preprocessing_function=None, augment=False):
    if augment:
        if preprocessing_function is None:
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=180,
                width_shift_range=0.25,
                height_shift_range=0.25,
                shear_range=0.25,
                zoom_range=0.25,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            return ImageDataGenerator(
                preprocessing_function=preprocessing_function,         # Preprocess images using EfficientNet's preprocessing function
                rotation_range=180,
                width_shift_range=0.25,
                height_shift_range=0.25,
                shear_range=0.25,
                zoom_range=0.25,
                horizontal_flip=True,                                  # Randomly flip images horizontally        
                fill_mode='nearest'                                    # Fill in new pixels after rotation or shift
            )
    else:
        if preprocessing_function is None:
            return ImageDataGenerator(rescale=1./255)
        else:
            return ImageDataGenerator(preprocessing_function=preprocessing_function)

def visualize_augmentations_custom(image_path, datagen, target_size=(256, 256), num_augmented=5):
    # Load image and ensure correct color format
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)

    # Expand dimensions to simulate a batch
    img_batch = np.expand_dims(img_resized, axis=0)

    # Generate augmented samples from the image
    generator = datagen.flow(img_batch, batch_size=1)

    # Set up the plot
    fig, axs = plt.subplots(1, num_augmented + 1, figsize=(16, 5))
    fig.suptitle("Image Augmentation Preview", fontsize=16, y=0.9)

    # Show original image
    axs[0].imshow(img_resized)
    axs[0].set_title("Original")
    axs[0].axis("off")

    # Loop through and display augmented images
    for idx in range(1, num_augmented + 1):
        aug_img = next(generator)[0]
        aug_img_display = np.clip(aug_img * 255, 0, 255).astype(np.uint8)
        axs[idx].imshow(aug_img_display)
        axs[idx].set_title(f"Variant {idx}")
        axs[idx].axis("off")

    plt.tight_layout()
    plt.show()

def generator_custom(datagen, directory):
    return datagen.flow_from_directory(
        directory,                                     # Path to the images
        target_size=(256, 256),                        # Resize to match EfficientNet input
        batch_size=16,                                 # Number of images to load per batch
        class_mode="categorical",                      # Use categorical labels for multi-class classification
    )

############################################################################################################
#---------------------------------Functions used in Best Pre-trained Model Notebook-------------------------
############################################################################################################

#Load data from directory 
def extract_image_data(directory, phylum_encoder=None, family_encoder=None):
    img_paths = []
    fam_lbls = []
    phy_lbls = []
    unique_phylums = set()
 # Iterate through subdirectories in the dataset directory
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if '_' not in class_name or not os.path.isdir(class_path):
            continue
# Split the directory name into phylum and family labels
        phylum, family = class_name.split('_', 1)
        unique_phylums.add(phylum)

        img_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        img_paths.extend([os.path.join(class_path, f) for f in img_files])
        fam_lbls.extend([family] * len(img_files))
        phy_lbls.extend([phylum] * len(img_files))
# Convert family and phylum labels to numpy arrays and reshape them
    fam_lbls = np.array(fam_lbls).reshape(-1, 1)
    phy_lbls = np.array(phy_lbls).reshape(-1, 1)

  # Helper function to encode labels using OneHotEncoder
    def encode_labels(labels, encoder):
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False)
            labels_encoded = encoder.fit_transform(labels)
        else:
            labels_encoded = encoder.transform(labels)
        return labels_encoded, encoder
    
 # Encode family and phylum labels
    fam_onehot, family_encoder = encode_labels(fam_lbls, family_encoder)
    phy_onehot, phylum_encoder = encode_labels(phy_lbls, phylum_encoder)

    return img_paths, fam_onehot, phy_onehot, family_encoder, phylum_encoder, list(unique_phylums)

#Compile Model
def compile_model(model):
    model.compile(
        optimizer= Adam(learning_rate=1e-3),     # Adam optimizer 
        # Define separate loss functions for each output
        loss={'family_output': 'categorical_crossentropy', 'phylum_output': 'categorical_crossentropy'},
        loss_weights={'family_output': 1.0, 'phylum_output': 0.5},
        metrics={'family_output': [F1Score(num_classes=202, name='family_f1_score'), 'accuracy', tf.keras.metrics.CategoricalCrossentropy(name='family_loss')],
                 'phylum_output': ['accuracy', tf.keras.metrics.CategoricalCrossentropy(name='phylum_loss')]})
    return model

# Multi Output Generator
def data_flow_gen(data_aug, dir_path, target_size=(224, 224), batch_size=16, class_mode="categorical", shuffle=True):
    return data_aug.flow_from_directory(
        dir_path,
        target_size=target_size,    
        batch_size=batch_size, 
        class_mode=class_mode,
        shuffle=shuffle # Shuffle the data if True
    )

# Custom generator for multi-task learning models
def multi_task_gen(fam_gen, phylum_lbls):
    # Extract batch size from the family generator  
    batch_size = fam_gen.batch_size  
    num_samples = len(fam_gen.filenames)  
    # Calculate the total number of batches
    num_batches = (num_samples + batch_size - 1) // batch_size  
    while True:  
        for i in range(num_batches):
            images, family_batch_labels = fam_gen[i]
            # Calculate start and end indices for the phylum labels
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            # Extract phylum labels for the current batch  
            phylum_batch_labels = phylum_lbls[start_idx:end_idx]
            yield images, (family_batch_labels, phylum_batch_labels)

# Callbacks
def build_callbacks(
    monitor_key="val_family_output_family_f1_score",
    checkpoint_dir="./Callbacks_mtl",
    model_name="history_mtl_callbacks",
    stop_patience=5,
    patience_reduce_lr=3,
    min_lr=1e-5
):
    # Create the directory for saving checkpoints
    save_path = os.path.join(checkpoint_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    return [
        # Save the best model based on the monitored metric
        ModelCheckpoint(
            filepath=os.path.join(save_path, 'best_model.weights.h5'),
            save_best_only=True,
            save_weights_only=True,
            monitor=monitor_key,
            mode='max',
            verbose=1
        ),
        # Stop training early if the monitored metric does not improve
        EarlyStopping(
            monitor=monitor_key,
            patience=stop_patience,
            mode='max',
            verbose=1
        ),
        # Reduce the learning rate if the monitored metric plateaus
        ReduceLROnPlateau(
            monitor=monitor_key,
            factor=0.75,
            patience=patience_reduce_lr,
            mode='max',
            verbose=1,
            min_lr=min_lr
        )
    ]

# Plotting the training history
def visualize_training_progress(history):
    # Create a figure for family metrics
    plt.figure(figsize=(15, 5))

    # Plot family accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['family_output_accuracy'], label='Train Accuracy (Family)')
    plt.plot(history.history['val_family_output_accuracy'], label='Validation Accuracy (Family)')
    plt.title('Family Classification Accuracy')
    plt.xlabel('Epoch')     # Label for the x-axis
    plt.ylabel('Accuracy')  # Label for the y-axis
    plt.legend()

    # Plot family F1-score
    plt.subplot(1, 2, 2)
    plt.plot(history.history['family_output_family_f1_score'], label='Train F1 Score (Family)')
    plt.plot(history.history['val_family_output_family_f1_score'], label='Validation F1 Score (Family)')
    plt.title('Family Classification F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    # Adjust layout and display the family metrics plot
    plt.tight_layout()
    plt.show()

    # Create a figure for phylum metrics and losses
    plt.figure(figsize=(15, 5))

    # Plot phylum accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['phylum_output_accuracy'], label='Train Accuracy (Phylum)')
    plt.plot(history.history['val_phylum_output_accuracy'], label='Validation Accuracy (Phylum)')
    plt.title('Phylum Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Total Train Loss')
    plt.plot(history.history['val_loss'], label='Total Validation Loss')
    plt.plot(history.history['family_output_loss'], label='Train Loss (Family)')
    plt.plot(history.history['val_family_output_loss'], label='Validation Loss (Family)')
    plt.plot(history.history['phylum_output_loss'], label='Train Loss (Phylum)')
    plt.plot(history.history['val_phylum_output_loss'], label='Validation Loss (Phylum)')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()