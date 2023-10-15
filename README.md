# Speech Emotion Recognition

This project aims to recognize emotions in speech using machine learning techniques. We use the Toronto Emotional Speech Set (TESS) dataset for training and evaluating our models.

## Dataset

The TESS dataset contains emotional speech samples. The dataset was loaded and analyzed to understand the distribution of emotions.

Download Dataset --> [click here](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

![Emotion Distribution](https://drive.google.com/file/d/1rqIow3-GXCWZ8d_lwIQgw0v0woiaq4FT/view?usp=drive_link](https://drive.google.com/drive/folders/1DdoFv42VN6tadS2RrodXxe5OCE6V0BT_)


## Data Preprocessing

1. Loaded the dataset and created a DataFrame.
2. Extracted Mel-frequency cepstral coefficients (MFCCs) as audio features.
3. One-hot encoded the labels.
4. Split the dataset into training and testing sets.

## Model Architecture

The neural network model used for this project has the following architecture:

- LSTM layer with 123 units
- Dense layer with 64 units and ReLU activation
- Dropout layer (dropout rate: 22%)
- Dense layer with 32 units and ReLU activation
- Dropout layer (dropout rate: 12%)
- Output layer with 7 units and softmax activation

Model Summary:

```plaintext
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 123)               51084
_________________________________________________________________
dense (Dense)                (None, 64)                7936
_________________________________________________________________
dropout (Dropout)            (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 7)                 231
=================================================================
Total params: 61,331
Trainable params: 61,331
Non-trainable params: 0
```

## Training and Early Stopping

The model was trained with early stopping, monitoring validation loss, and patience set to 10 epochs. Training stopped when the validation loss no longer improved.

![Training vs Validation Accuracy](https://drive.google.com/drive/folders/1DdoFv42VN6tadS2RrodXxe5OCE6V0BT_)

## Visualization

The `waveplot` function was used to visualize audio waveforms, and the training process's accuracy and validation accuracy were visualized over epochs.

## Future Improvements

- Experiment with different model architectures.
- Fine-tune hyperparameters.
- Use transfer learning from pre-trained models.
- Augment the dataset for better model generalization.

## Conclusion

This project demonstrates the process of building a speech emotion recognition model using MFCC features and LSTM neural networks. The model achieved an accuracy of **98.26%** on the test data.

Feel free to contribute and further improve this project !

If you Really liked this Project , Don't forget to Give a Star :-) 
