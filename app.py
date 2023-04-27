import streamlit as st
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array



# Load the pre-trained model and tokenizer
loaded_Model = tf.keras.models.load_model('my_model.h5', compile=False)
loaded_Model.compile(loss='categorical_crossentropy', optimizer='adam')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
index_word = dict([(index, word)
                  for word, index in tokenizer.word_index.items()])




def preprocess_image(image_path):
    
    # Load VGG16 model with desired output layer
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # model = load_model('vgg16.h5')
    
    # Load and preprocess image
    npix = 224
    target_size = (npix, npix, 3)
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    
    # Make a prediction using VGG16 model
    prediction = model.predict(image.reshape((1,) + image.shape[:3]))
    
    return prediction.flatten()

# x = preprocess_image('image.jpg')
# x = x.reshape(1, -1)

# Define the function for generating a caption for the uploaded image
def predict_caption(image, tokenizer, model, maxlen):
    '''
    image.shape = (1, 4462)
    '''
    in_text = 'startseq'


    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word[yhat]
        in_text += " " + newword
        if newword == "endseq":
            break
    return (in_text)

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Image Captioning', page_icon=':camera:', layout='wide')
    st.title('Image Captioning')
    
    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Generate and display the caption
        x= preprocess_image(uploaded_file)
        x = x.reshape(1, -1)
        caption = predict_caption(x, tokenizer, loaded_Model, 30)
        st.header('Caption:')
        st.write(caption)

if __name__ == '__main__':
    app()
