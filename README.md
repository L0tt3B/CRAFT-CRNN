# CRAFT-CRNN Text Detection & Recognition

This repository contains code extracted from a **JupyterLab project** where I implemented a complete text detection and extraction pipeline using CRAFT for text detection and CRNN for text recognition.

## What This Repo Contains

### Model Architectures (`models/`)
- **CRAFT** (Character-Region Awareness For Text detection) - Complete implementation
- **CRNN** (Convolutional Recurrent Neural Network) - Sequence recognition model  
- **VGG16-BN Backbone** - Custom feature extraction backbone

### Algorithms & Utilities (`utils/`)
- **Text detection algorithms** - Character affinity processing, bounding box generation
- **Text recognition utilities** - CTC decoding, label conversion, data processing
- **Dataset handling** - LMDB data loading, image transformations
- **Visualization tools** - Result plotting and output generation

### Sample Data (`examples/`)
- Example images to provide context for the algorithms

## What's Not Included

The following components from the original JupyterLab are **not included**:

- **Trained model weights** (.pth files)
- **Complete end-to-end inference pipeline** 
- **Private datasets and training data**
- **Generated output files and results**
- **The original Jupyter notebook** with execution cells

## Technical Implementation

### Text Detection (CRAFT)
- Character-level region and affinity detection
- Fixed-padding bounding box expansion  
- Reading-order sorting (top-left to bottom-right)
- Connected components analysis with OpenCV

### Text Recognition (CRNN)
- CNN feature extraction + Bidirectional LSTM
- Connectionist Temporal Classification (CTC)
- Sequence-to-text decoding
- Confidence-based recognition
