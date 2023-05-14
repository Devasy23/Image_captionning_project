# import streamlit as st
# from PIL import Image
# import numpy as np
# import pickle
# import tensorflow as tf
# import os
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array



# # Load the pre-trained model and tokenizer


# def load():
#     global loaded_Model, tokenizer, index_word
#     loaded_Model = tf.keras.models.load_model('best_model.h5', compile=False)
#     loaded_Model.compile(loss='categorical_crossentropy', optimizer='adam')
#     tokenizer = pickle.load(open('tokenizer_best.pkl', 'rb'))
#     index_word = dict([(index, word)
#                     for word, index in tokenizer.word_index.items()])




# def preprocess_image(image_path):
    
#     # Load VGG16 model with desired output layer
#     model = VGG16()
#     model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
#     # model = load_model('vgg16.h5')
    
#     # Load and preprocess image
#     npix = 224
#     target_size = (npix, npix, 3)
#     image = load_img(image_path, target_size=target_size)
#     image = img_to_array(image)
#     image = preprocess_input(image)
    
#     # Make a prediction using VGG16 model
#     prediction = model.predict(image.reshape((1,) + image.shape[:3]))
    
#     return prediction.flatten()

# # x = preprocess_image('image.jpg')
# # x = x.reshape(1, -1)

# # Define the function for generating a caption for the uploaded image
# def predict_caption(image, tokenizer, model, maxlen, index_word):
#     '''
#     image.shape = (1, 4462)
#     '''
#     in_text = 'startseq'


#     for iword in range(maxlen):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen)
#         yhat = model.predict([image, sequence], verbose=0)
#         yhat = np.argmax(yhat)
#         newword = index_word[yhat]
#         in_text += " " + newword
#         if newword == "endseq":
#             break
#     return (in_text)

# # Define the Streamlit app
# def format_caption(caption):
#     '''
#     caption: string
#     '''
#     caption = caption.split()
#     caption = caption[1:-1]
#     caption = ' '.join(caption)
#     return caption

# def app():
#     st.set_page_config(page_title='Image Captioning', page_icon=':camera:', layout='wide')
#     st.title('Image Captioning')
    
#     # Allow the user to upload an image
#     uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, width=300, caption='Uploaded Image' )
        
        
#         # give an option to choose between 2 models
#         model = st.selectbox('Choose a model', ['Model 1', 'Model 2'])
#         if model == 'Model 1':
#             st.info('Chosen model : ResNet50 model-20 epochs')
        
        
#         # Generate and display the caption
#             x= preprocess_image(uploaded_file)
#             x = x.reshape(1, -1)
#             tokenizer = pickle.load(open('tokenizer_best.pkl', 'rb'))
#             loaded_Model = tf.keras.models.load_model('best_model.h5', compile=False)
#             loaded_Model.compile(loss='categorical_crossentropy', optimizer='adam')
             
#             index_word = dict([(index, word)
#                 for word, index in tokenizer.word_index.items()]) 
#             caption = predict_caption(x, tokenizer, loaded_Model, 35, index_word)
#             # st.header('Caption:')
#             caption = format_caption(caption)
#             st.header('Caption : '+ caption)
#             # st.write(caption)
            
#         elif model == 'Model 2':
#             loaded_Model = tf.keras.models.load_model('my_model.h5', compile=False)
#             loaded_Model.compile(loss='categorical_crossentropy', optimizer='adam')
            
#             tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
#             index_word = dict([(index, word)
#                 for word, index in tokenizer.word_index.items()])
            
#             st.info('Chosen Model : VGG16 model-20 epochs')
#             x= preprocess_image(uploaded_file)
#             x = x.reshape(1, -1)
#             caption = predict_caption(x, tokenizer, loaded_Model, 30, index_word)
#             caption = format_caption(caption)
#             st.header('Caption : '+ caption)
#             # st.write(caption)

# if __name__ == '__main__':
#     app()
    
    
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

def preprocess_image(image):
    # Load the image and resize it to (224, 224)
    img = Image.open(image)
    img = img.resize((224, 224))
    
    # Convert the image to a numpy array and normalize the pixel values
    x = np.array(img)
    x = x / 255.0
    
    # Add an extra dimension to the array to represent the batch size
    x = np.expand_dims(x, axis=0)
    
    return x

def predict_caption(image, tokenizer, model, max_len, index_word):
    # Generate the initial state of the decoder using the encoder output
    encoder_output = model.layers[0](image)
    decoder_state_h = model.layers[1](encoder_output)
    decoder_state_c = model.layers[2](encoder_output)
    decoder_state = [decoder_state_h, decoder_state_c]
    
    # Initialize the caption with the start token
    caption = '<start>'
    
    # Generate the caption word by word
    for i in range(max_len):
        # Convert the caption to a sequence of integers
        sequence = tokenizer.texts_to_sequences([caption])[0]
        
        # Pad the sequence to the maximum length
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_len)
        
        # Predict the next word using the decoder and attention mechanism
        output, attention_weights = model.layers[3]([encoder_output, decoder_state])
        output = model.layers[4](output)
        output = tf.argmax(output, axis=-1)
        
        # Convert the predicted word index to a word
        word = index_word.get(output.numpy()[0][0], '<unk>')
        
        # Stop generating the caption if the end token is reached
        if word == '<end>':
            break
        
        # Add the predicted word to the caption
        caption += ' ' + word
        
        # Update the decoder state using the predicted word and attention weights
        decoder_state, decoder_hidden, attention_weights = model.layers[5](
            [output, decoder_state, encoder_output])
    
    return caption

def format_caption(caption):
    # Remove the start and end tokens from the caption
    caption = caption.replace('<start>', '').replace('<end>', '')
    
    # Remove any extra spaces from the caption
    caption = caption.strip()
    
    # Capitalize the first letter of the caption
    caption = caption.capitalize()
    
    return caption

def app():
    st.set_page_config(page_title='Image Captioning', page_icon=':camera:', layout='wide')
    st.title('Image Captioning')
    
    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        st.image(image[0], width=300, caption='Uploaded Image' )
        
        # Give an option to choose between 2 models
        model = st.selectbox('Choose a model', ['Model 1', 'Model 2'])
        if model == 'Model 1':
            st.info('Chosen model : ResNet50 model-20 epochs')
            
            # Load the tokenizer and model
            tokenizer = pickle.load(open('tokenizer_best.pkl', 'rb'))
            model = tf.keras.models.load_model('best_model.h5', compile=False)
            
            # Generate and display the caption
            caption = predict_caption(image, tokenizer, model, 35, tokenizer.index_word)
            caption = format_caption(caption)
            st.header('Caption : '+ caption)
            
        elif model == 'Model 2':
            st.info('Chosen model : Custom model')
            
            # Load the model
            model = tf.keras.models.load_model('my_model.h5', compile=False)
            model.compile(loss='categorical_crossentropy', optimizer='adam')
            
            tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
            index_word = dict([(index, word)
                for word, index in tokenizer.word_index.items()])
             
            
            # Generate and display the caption
            caption = predict_caption(image, tokenizer, model, 35, tokenizer.index_word)
            caption = format_caption(caption)
            st.header('Caption : '+ caption)
