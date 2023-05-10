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

**Image Captioning using Transfer Learning(model 2):**
This project aims to generate captions for images using transfer learning. We use pre-trained models such as VGG16 and ResNet50 to extract features from images, and LSTM to generate captions.

**Prerequisites:**
Tensorflow
Keras
Numpy
Pandas
Matplotlib
NLTK

**Model Description:**
We use pre-trained models VGG16 and ResNet50 to extract features from images. The last layer of these models is removed as it corresponds to the classification layer. The output from the last convolutional layer is fed to the LSTM layer along with the embedded text input. The LSTM layer learns the sequence of text and generates captions.

**VGG16 model**
The VGG16 model is loaded from Keras and the last layer is removed. The model is then restructured using Keras' Model API. We then create a new model with inputs as image and output as the output of the second last layer of VGG16.

**ResNet50 model**
The ResNet50 model is loaded from Keras with pre-trained weights from ImageNet. The last layer is removed and the model is restructured using Keras' Model API. We then create a new model with inputs as image and output as the output of the second last layer of ResNet50.

**LSTM model**
The text input is first pre-processed using NLTK. We create a vocabulary of words and use it to convert each word to an integer. The text is then embedded using Keras' Embedding layer. The embedded text is then fed to the LSTM layer which generates captions.

**Decoder model**
The output from the LSTM layer and the output from the pre-trained model are combined and fed to the decoder model. The decoder model is a fully connected neural network with two hidden layers and an output layer. The output layer is a softmax layer which predicts the probability of each word in the vocabulary.
