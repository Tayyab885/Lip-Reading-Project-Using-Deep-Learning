import streamlit as st
import os
import tensorflow as tf
import moviepy.editor as moviepy


from utils import load_data, num_to_char
from model import load_model


st.set_page_config(layout="wide", page_title="Lip Reading", page_icon="👄")

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip Reader')
    st.info('This is a lip reading application that uses a deep learning model to predict the words being spoken in a video.')

st.title('Lip Reading Application')
options = os.listdir(os.path.join('..','data','s1'))    
video_path = st.selectbox('Select a video', options)

col1, col2 = st.columns(2)

if options:
    # Read the video
    with col1:
        st.subheader('Video')
        st.info('The Video below display the converted video in mp4 format')
        file_path = os.path.join('..','data','s1',video_path)
        clip = moviepy.VideoFileClip(file_path)
        clip.write_videofile('test_video.mp4')
        gray_clip = clip.fx(moviepy.vfx.blackwhite)
        gray_clip.write_gif('animation.gif')
        
        # Display the video
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
        
    
    with col2:
        st.subheader('GIF')
        st.info('This is all the machine learning model sees when making a prediction')
        
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 
        
        
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat,[75],greedy=True)[0][0].numpy()
        st.text(decoder)
        
        ## Convert the tokens to words
        st.info('This is the output of the machine learning model as words')
        converted = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')   
        st.text(converted) 
        
        
