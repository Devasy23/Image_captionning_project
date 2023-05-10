# Image_captionning_project
Our repo for NLP innovative


**Image Captioning with VGG16 and LSTM(model 1)**
This code implements a deep learning model for image captioning using transfer learning with VGG16 and LSTM networks.

**Dependencies:**
Tensorflow
Keras
Numpy

**Dataset:**
The model requires a dataset of images and their corresponding captions for training. The dataset should be preprocessed to create two numpy arrays:

Ximage_train: A 4D numpy array of shape (num_images, image_height, image_width, num_channels) containing the images.
Xtext_train: A 2D numpy array of shape (num_images, max_caption_length) containing the tokenized captions.

**Usage:**
1. Load the required dependencies and preprocess the dataset.
2. Load the VGG16 model and its weights from the keras library or locally saved weights.
3. Pop the last layer from the VGG16 model and set it as the output layer for feature extraction.
4. Create a model for the encoder by removing the last two layers from the VGG16 model.
5. Create a model for the decoder by adding a LSTM layer and a Dense layer to the encoder model.
6. Compile the model using categorical_crossentropy as the loss function and adam as the optimizer.
7. Train the model on the preprocessed dataset
