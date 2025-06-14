# Movie Genre Detection from YouTube Trailers: Experimental Design

## Introduction

This document outlines the experimental design for building a movie genre detection model using YouTube trailers as input. The model uses only the video signal (visual and audio content) without any associated metadata. The design leverages the MovieLens 20M dataset with YouTube trailer IDs for model training and evaluation.

## Methodology for Determining Movie Genre Label Codebook

The genre label codebook is determined through a comprehensive analysis of the MovieLens dataset's existing genre taxonomy, which is widely accepted in the movie industry and research community.

### Label Codebook Strategy:

1. **Existing Genre Taxonomy**: We use the MovieLens dataset's genre taxonomy, which includes:
   - Action
   - Adventure
   - Animation
   - Children's
   - Comedy
   - Crime
   - Documentary
   - Drama
   - Fantasy
   - Film-Noir
   - Horror
   - Musical
   - Mystery
   - Romance
   - Sci-Fi
   - Thriller
   - War
   - Western

2. **Genre Distribution Analysis**: We perform an analysis to understand the distribution of genres within the dataset, identifying potentially underrepresented genres.

3. **Multi-Label Approach**: Since movies often belong to multiple genres, we design our codebook to handle multi-label classification rather than forcing each movie into a single category.

4. **Genre Validation**: The genre assignments in MovieLens come from a combination of expert curation (originally from IMDB) and user validation, providing a reliable foundation.

## Methodology for Labelling of a Large Scale Dataset

Our approach for labeling the large-scale dataset leverages the existing MovieLens annotations while ensuring consistency and handling of multi-label cases.

### Labeling Methodology:

1. **Mapping MovieLens IDs to YouTube Trailers**: 
   - Use the links.csv file to connect MovieLens movie IDs with their corresponding IMDB IDs
   - Map these to YouTube trailer IDs either through an API or pre-existing mapping
   - Download trailers using yt-dlp for processing

2. **Multi-Label Encoding**:
   - Convert genre lists into multi-hot encoded vectors
   - Use the MultiLabelBinarizer from scikit-learn to create binary indicators for each genre
   - This approach preserves multi-genre assignments while creating ML-ready labels

3. **Data Quality Assurance**:
   - Verify that each trailer is correctly mapped to its corresponding movie
   - Validate that trailer content is representative of the movie
   - Remove corrupt or inappropriate trailers

4. **Genre Imbalance Handling**:
   - Analyze distribution of genres across the dataset
   - Apply class weights during model training to account for imbalance
   - Consider data augmentation techniques for underrepresented genres

## Partitioning Methodology

Our dataset partitioning strategy ensures that the model is properly trained, validated, and tested while maintaining the distribution of genres across all partitions.

### Partitioning Strategy:

1. **Train/Validation/Test Split**:
   - Training set: 70% of the data
   - Validation set: 15% of the data
   - Test set: 15% of the data

2. **Stratified Sampling**:
   - Use stratified sampling based on genre distribution
   - For multi-label data, we use an iterative stratification approach that ensures each partition maintains similar genre distributions
   - This is crucial for handling imbalanced classes and multi-label scenarios

3. **Cross-Validation**:
   - Apply k-fold cross-validation (k=5) on the training data for hyperparameter tuning
   - Maintain stratification across folds to ensure representative distributions

4. **Data Independence**:
   - Ensure no leakage between partitions
   - Keep all trailers from the same movie in the same partition

## Modelling Approach

Our multimodal deep learning approach integrates visual and audio content from trailers to predict movie genres.

### Model Architecture:

1. **Visual Content Processing**:
   - Extract frames from trailers at a consistent rate (e.g., 1 frame per second)
   - Process frames through a pre-trained CNN (e.g., ResNet-50)
   - Extract high-level visual features capturing scenes, colors, and visual style

2. **Audio Content Processing**:
   - Extract audio from trailers
   - Apply Automatic Speech Recognition (ASR) using Whisper model
   - Process the transcribed text through a language model (e.g., BERT) to extract semantic features

3. **Multimodal Integration**:
   - Design a fusion module that combines visual and text features
   - Use attention mechanisms to weigh different modalities
   - Learn joint representations capturing the correlation between visual and audio elements

4. **Classification Head**:
   - Multi-label classification layer with sigmoid activations
   - Output probabilities for each genre

### Training Strategy:

1. **Loss Function**: Binary Cross-Entropy Loss (appropriate for multi-label classification)
2. **Optimization**: Adam optimizer with learning rate scheduling
3. **Regularization**: Dropout, weight decay, and early stopping
4. **Handling Class Imbalance**: Class weights in the loss function

## Experimental Protocol and Performance Metric Calculation

Our experimental protocol follows a rigorous methodology to ensure reliable and reproducible results.

### Experimental Protocol:

1. **Baseline Models**:
   - Visual-only model (CNN)
   - Text-only model (from ASR)
   - Simple concatenation of features

2. **Proposed Models**:
   - Multimodal fusion model
   - Attention-based fusion model
   - Temporal modeling variation

3. **Ablation Studies**:
   - Effect of different visual backbones
   - Effect of different ASR models
   - Impact of temporal modeling components

4. **Hyperparameter Optimization**:
   - Learning rate
   - Batch size
   - Network architecture parameters
   - Fusion strategy parameters

### Performance Metrics:

1. **Primary Metrics**:
   - F1-Score (micro and macro averaged)
   - Precision and Recall
   - ROC-AUC for each genre

2. **Secondary Metrics**:
   - Hamming Loss
   - Subset Accuracy
   - Example-based F1 Score

3. **Analysis Methods**:
   - Per-genre performance analysis
   - Confusion matrix analysis
   - Error case studies
   - Visualization of learned representations

4. **Statistical Significance**:
   - Perform statistical tests to ensure improvements are significant
   - Report confidence intervals for key metrics

## References

1. Simou, E., et al. (2018). "Multimodal and Temporal Video Content Analysis for Automatic Genre Categorization." In *Proceedings of the 2018 ACM International Conference on Multimedia Retrieval*.

2. Wehrmann, J., & Barros, R. C. (2017). "Movie Genre Classification: A Multi-Label Approach Based on Convolutions Through Time." *Applied Soft Computing*, 61, 973-982.

3. Zhou, H., et al. (2018). "Temporal Modeling Approaches for Large-scale YouTube-8M Video Understanding." *arXiv preprint arXiv:1707.04555*.

4. Yang, C., et al. (2019). "Multi-label Classification for Video Genre Detection Using Audio-Visual Features." In *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

5. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In *Proceedings of NAACL-HLT 2019*.

6. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

7. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Blog*, 1(8), 9.

8. Harper, F. M., & Konstan, J. A. (2015). "The MovieLens Datasets: History and Context." *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 19.

9. Kay, W., et al. (2017). "The Kinetics Human Action Video Dataset." *arXiv preprint arXiv:1705.06950*.

10. Snoek, C. G., et al. (2019). "The MediaEval 2019 Emotional Impact of Movies Task." In *MediaEval 2019 Workshop*.
