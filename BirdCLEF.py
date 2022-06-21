import time
from urllib.request import urlopen

import pandas as pd
import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
from bs4 import BeautifulSoup
from codecarbon import EmissionsTracker
from matplotlib import pyplot as plt
from birdnet.BirdNET import analyze_NCH
from birdnet.BirdNET import config
import re
import random


def load_audio(file):
    y, sr = librosa.load(file, sr=22050)
    return y, sr


def draw_spectrogram(file):
    y, sr = load_audio(file)
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(figsize=(4, 2))
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="linear", ax=ax)
    # ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set(title='Frequency spectrogram')
    st.pyplot(fig)


def draw_mel_spectrogram(file):
    y, sr = load_audio(file)
    D = np.abs(librosa.stft(y)) ** 2
    S = librosa.feature.melspectrogram(S=D, sr=sr)

    fig, ax = plt.subplots(figsize=(4, 2))
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    st.pyplot(fig)


def update_config(duration, metadata):
    config.WORKING_PATH = r'birdnet'
    config.CODES_FILE = os.path.join(config.WORKING_PATH, 'BirdNET', 'eBird_taxonomy_codes_2021E.json')
    config.LABELS_FILE = os.path.join(config.WORKING_PATH, 'BirdNET', 'BirdNET_GLOBAL_2K_V2.1_Labels.txt')
    config.LABELS_TRANS_FILE = os.path.join(config.WORKING_PATH, 'BirdNET', 'ebird.json')
    config.CODES = analyze_NCH.loadCodes()
    config.LABELS = analyze_NCH.loadLabels(config.LABELS_FILE)
    config.SPECIES_LIST_FILE = os.path.join(config.WORKING_PATH, 'BirdNET', 'labels_birdclef.txt')
    config.SPECIES_LIST = analyze_NCH.loadSpeciesList(config.SPECIES_LIST_FILE)
    config.SIG_LENGTH = duration
    config.LATITUDE = metadata.latitude.iloc[0]
    config.LONGITUDE = metadata.longitude.iloc[0]


def get_photo_url(name):
    if name == "Japanese Bush Warbler":
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Cettia_diphone_%28crying%29.JPG/1280px-" \
               "Cettia_diphone_%28crying%29.JPG"
    elif name == "Omao":
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Oma%27o_%289-4-2017%29_Pu%27u_O%27o_trail%2C_" \
               "Kipuka_Ainahou_section%2C_Hawai%27i_co%2C_Hawaii_-04jpg_%2837545163396%29.jpg/" \
               "800px-Oma%27o_%289-4-2017%29_Pu%27u_O%27o_trail%2C_Kipuka_Ainahou_section%2C_Hawai%27i_co%2" \
               "C_Hawaii_-04jpg_%2837545163396%29.jpg"
    html = urlopen('https://en.wikipedia.org/wiki/' + name.replace(' ', '_'))
    bs = BeautifulSoup(html, 'html.parser')
    image = bs.find_all('img', {'src': re.compile('.jpg')})[0]['src'].split('.jpg')[0] + '.jpg'
    return image.replace('//upload.wikimedia.org/wikipedia/commons/thumb/',
                         'https://upload.wikimedia.org/wikipedia/commons/')


@st.experimental_memo
def predict(file, col_names):
    return pd.DataFrame(analyze_NCH.predictFile((file, config.getConfig())), columns=col_names)


def get_list_of_audio():
    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk("audio"):
        list_of_files += [os.path.join(dirpath, file) for file in filenames]
    if r"audio\train_metadata.csv" in list_of_files:
        list_of_files.remove(r"audio\train_metadata.csv")
    return list_of_files


def kwh_to_g_co2(emmissions):
    emmissions = emmissions / 80
    return round(emmissions, 8)


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_icon="image/favicon_blanc.png")
    st.sidebar.image("image/logo_aqsone_noir_Xsmall.png")
    df_code = pd.read_csv(r"birdnet/BirdNET/code_to_name.csv")
    df_rate = pd.read_csv("audio/train_metadata.csv")[["filename", "latitude", "longitude", "rating"]]

    list_of_files = get_list_of_audio()

    code_random_bird = random.choice(df_code.Code)
    audio_for_this_bird = [audio for audio in list_of_files if code_random_bird in audio]
    selected_audio = random.choice(audio_for_this_bird)
    code_and_file = selected_audio[6:].replace("\\", "/")

    duration = librosa.get_duration(filename=r"audio/" + code_and_file)
    col1, col2, col3 = st.columns([3, 1, 3])
    metadata = df_rate.loc[df_rate.filename == code_and_file]

    with col1:
        st.header("Audio and spectrogram")
        audio = st.audio(selected_audio)
        draw_mel_spectrogram(selected_audio)
        quality = metadata.rating.iloc[0]
        if quality == 0:
            quality = "Unknown"
        else:
            quality = str(quality)
        st.metric("Audio Quality (out of 5):", quality)
        st.map(metadata, zoom=4)
    update_config(duration, metadata)
    start = time.time()
    labels_fr = pd.read_json(config.LABELS_TRANS_FILE)
    time_compute = round(time.time() - start, 2)
    st.sidebar.write("File selected:", code_and_file)
    scientific_name_selected_bird = df_code.loc[df_code.Code == code_random_bird].sciName.iloc[0]
    selected_bird = df_code.loc[df_code.Code == code_random_bird].comNames_en.iloc[0]
    selected_bird_fr = labels_fr.loc[labels_fr.speciesCode == code_random_bird].comName.iloc[0]
    st.sidebar.write("Code:", code_random_bird)
    st.sidebar.write("English name: ", selected_bird)
    st.sidebar.write("Scientific name:", scientific_name_selected_bird)
    st.sidebar.write("French name:", selected_bird_fr)

    with col3:
        st.header("Results")
        col_names = ['ts_start', 'ts_stop', 'sciName', 'comName_en', 'confidence']
        
        tracker = EmissionsTracker()
        tracker.start()
        df = predict(r"audio/" + code_and_file, col_names)
        emissions = tracker.stop()
        emissions = kwh_to_g_co2(emissions)
        df = df.merge(labels_fr, on='sciName', how='left')

        if selected_bird in list(df.comName_en):
            st.success(f"The {selected_bird} has been detected in {time_compute} seconds. This prediction consumed {emissions}gCO₂eq/kWh")
        else:
            st.error(f"The {selected_bird} has not been detected. The computation tooked {time_compute} seconds. This prediction consumed {emissions}gCO₂eq/kWh")

        if df.shape[0] > 0:
            df.loc[:, "url_photo"] = df.comName_en.apply(get_photo_url)
            if df.shape[0] > 1:
                bird_found = st.selectbox("Details on the birds that have been found:", df.comName_en)
            else:
                bird_found = df.comName_en.iloc[0]
                st.subheader(f"The AI has detected: {bird_found}")
            df_bird = df.loc[df.comName_en == bird_found]

            st.image(df_bird.url_photo.iloc[0])
            st.metric("Confidence", str(round(df.confidence.iloc[0] * 100, 2)) + "%")
        else:
            st.error("No bird detected")
