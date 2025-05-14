# Handwritten Occupation Code Transcription Model
This project involves the development and use of a machine learning model to automatically transcribe handwritten 3-digit occupation codes from the Norwegian population census of 1950.

## Features
- Model architecture: CNN-RNN with a CTC end layer.
- Accuracy: 97% on the provided [training dataset](https://doi.org/10.18710/OYIH83).
- Supported labels: Single digits (0-9), 't' (for text), and 'b' (for blank cells).
- Training dataset details: 30,000 manually labeled images, 264 classes, highly imbalanced distribution.

## Usage
### Running the model
In order to run the model you need to have a database of images. Update the variables in Testing/inference_runner.py and Testing/inference.py with your database and table information, as well as the path to the model. These areas where an update is required is marked in the python scripts. 
After that, run the inference_runner.py script.

### Training your own model
The script to train your own model can be found in Training/ctc_training.py. When providing training images to the model, the default way is to add the path to a folder containing your training set images. This is done to reflect the directory hierarchy of the provided training dataset. 
But this way of fetching training images can be altered to a database solution as well. The important part is that the script generates lists of images and 3-digit string labels.
Once the script ctc_training.py has been updated with the path to the training set images, run the script.

## Requirements
Please note that this model was trained using version 2.13 of tensorflow, and will require the same version if you wish to retrain the model.

## Acknowledgements
The training dataset used in the project was manually labeled by our team at HistLab using a custom GUI (details available in the [accompanying paper](https://doi.org/10.51964/hlcs11331).)

## Contact
For questions or feedback, please reach out to the project maintainers at bjorn-richard.pedersen@uit.no

## References
For more details about the project and the custom GUI used for labeling, as well as general lessons learned for creating a ML transcription pipeline, please refer to the [accompanying paper](https://doi.org/10.51964/hlcs11331).

More information about the custom GUI as well as the manual work done to validate the model's outputs can be found in our [follow-up paper](https://doi.org/10.51964/hlcs15456).
