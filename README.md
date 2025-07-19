# transformer_for_metasurface
This project  is a Transformer-based model trained for metasurface inverse design . It includes spectral data preprocessing, an encoder-decoder architecture, and the entire workflow for training and evaluation


## Introduction
(Details to be added)\
**You can compose a more comprehensive jupyter notebook to show how to run from the beginning to the end.
Or you might give detailed instructions in readme**

Image: Model Architecture

## File Descriptions
**todo: you need include data-related codes in data/, e.g., python code to generate structural parameters, fdtd scripts,...
so that users can follow your steps and generate their own dataset for further usage**  
- `data/`: Directory containing datasets.
- `config.py`: Configuration file.
- `dataset.py, dataloader.py`: Scripts for handling datasets.
- `loss_func.py`: Defines loss functions.
- `model.py`: Model architecture implementation.
- **file names below are confused. evalue.py looks like evaluation&predict of inverse model. model_evalue seems like
- evaluation mode of forward network. model_evalue_plot.py has some visualization codes. Anyway, you need to reorganize 
- these three files.**
- `model_evalue.py, model_evalue_plot.py, evalue.py`: Model evaluation scripts.
- **is train.py using the real data + augmented data and train_finetune only using real_data? if that's the case, 
- you need clear clarifications and explaining the rationale behind it.**
- `train.py`: Model training script.The data used is a combination of FDTD simulated data and augmented data, with the order shuffled for training. The accuracy of the augmented data can reach over 90%, so the performance on the test set using only FDTD simulated data is still good.
- `train_finetune.py`: Unlike train.py, train_finetune first performs pre-training using data augmented data, and then fine-tunes the simulated data with FDTD simulation data. Due to the accuracy of the data augmentation, the overall difference is not significant compared to direct training, but it should be very useful for adding new shapes in the future.
- `tokenlizer.py`: Tokenizer for data processing.
- `utils.py`: Utility functions.

## Model Evaluation
(Details to be added)

## Usage
1. Install the required dependencies.
2. Prepare the data and store it in .pt format.
3. Adjust the parameters in config.py as needed.
4. Run train.py to start training the model.
5. Use evaluate.ipynb for model evaluation and analysis.

## Results Presentation

We provide a Jupyter Notebook to demonstrate the model's performance and evaluation process. You can view the detailed results and analysis through the following link:

[Results Presentation - results_presentation.ipynb](results_presentation.ipynb)

## Todo-List

### · Data Processing
- [X] Data Augmentation
- [X] Data Validation
- [X] Data Normalization Attempts
#### · Future Plan
- [ ] Pretrain on the augmented dataset and SFT on the real dataset

### · Encoder Model Evaluation
- [X] Transformer Encoder Model 
- [X] ViT Encoder Model
- [X] CNN Encoder Model
- [X] Swin Transformer Encoder Model

### · Model Debugging
#### · Parameters
- [ ] Batch Size
- [ ] Learning Rates
- [ ] Number of Encoder & Decoder Layers 
- [X] Dimension of Hidden Layer
- [X] Dimension of Feedforward Layer
- [X] Number of Attention Heads
- [ ] Window Size of ViT
#### · Loss Functions
- [X] Cross-Entropy Loss
- [X] Critic Loss + Cross-Entropy Loss
- [X] Token Loss + Cross-Entropy Loss

#### · Model Prediction
- [ ] Top-K value
- [ ] Temperature

### · Case Study
(Details to be added)


