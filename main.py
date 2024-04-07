from neurosity import NeurositySDK
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import pygame
import tempfile
import re
import clip
import io
from PIL import Image
import requests

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32",device=device,jit=False)
checkpoint = torch.load("model/Neurosity_ViT-b32.pt")
model.load_state_dict(checkpoint['model_state_dict'])

load_dotenv()

data_list = []
incremental = 1
openai_key = os.getenv("OPENAI_KEY")
playht_user = os.getenv("PLAYHT_USER")
playht_key =  os.getenv("PLAYHT_KEY")

file_path = 'context.txt'

# Open the file and read its content
with open(file_path, 'r') as file:
    context = file.read()

#  I'm in front of an audience, greeting them, talking to people.
message_history = [
            {"role": "system", "content": f"Your job is a mute people helper. You will be provided by words and you will construct a sentence from those words as like you were talking on behaf of the user. The sentence should be short with meaning given the provided context. Here is more information about what the context of the user is : {context} "}
        ]
last_response = ""

# Your existing word list
word_list = [
    "I", "You", "That", "Here", "There", "Go", "See", "Use", "Make", "Want",
    "Give", "Say", "Thing", "Food", "Water", "Place", "Time", "Day", "Night",
    "Person", "Good", "Bad", "Star", "Animal", "Friend", "Enemy", "Family",
    "Love", "Fear", "Happy", "Sad", "Help", "Yes", "No"
]

# Global index to keep track of the current word in the list
current_word_index = 0

# Your Neurosity SDK initialization
neurosity = NeurositySDK({
    "device_id": os.getenv("NEUROSITY_DEVICE_ID"),
})

# Login to Neurosity
neurosity.login({
    "email": os.getenv("NEUROSITY_EMAIL"),
    "password": os.getenv("NEUROSITY_PASSWORD")
})

prev_probs = None
last_words = []

def create_embedding(data):
    global data_list, incremental, prev_probs, last_words
    incremental += 1
    data_list = []
    if not data:
        print("No data to plot.")
        return

    concatenated_data = np.concatenate([np.array(data_dict['data']) for data_dict in data], axis=1)
    channel_names = data[0]['info']['channelNames']
    sampling_rate = data[0]['info']['samplingRate']
    time_points = np.linspace(0, concatenated_data.shape[1] / sampling_rate, num=concatenated_data.shape[1])

    plt.figure(figsize=(20, 10))
    for i, _ in enumerate(channel_names):
        plt.plot(time_points, concatenated_data[i, :], label='_nolegend_')

    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)

    img_buffer.seek(0)

    plot_image = preprocess(Image.open(img_buffer)).unsqueeze(0).to(device)
    plt.close()
    text = clip.tokenize(word_list).to(device)

    with torch.no_grad():
        print(f"Brainwave embeddings from image")
        image_features = model.encode_image(plot_image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(plot_image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    if prev_probs is None:
        prev_probs = probs


    probs_diff = probs - prev_probs
    top_1_index = np.argsort(probs_diff[0])[-1]
    top_1_word = word_list[top_1_index]

    last_words.append(top_1_word)

    if (len(last_words) == 3):
        get_constructed_sentence(last_words)
        last_words = []

    prev_probs = probs

    plt.close()

def text_to_speech(text):

    print("Text-to-speech from cloned voice")
    url = "https://play.ht/api/v2/tts"
    payload = {
        "text": text,
        "voice": "s3://voice-cloning-zero-shot/0880bbd5-9707-4f00-8969-a6f5634f29f0/original/manifest.json",
        "voice_engine": "PlayHT2.0"
    }
    headers = {
        "Authorization": f"Bearer {playht_key}",
        "X-USER-ID": f"{playht_user}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    url_match = re.search(r'"url":"([^"]+)"', response.text)

    if url_match:
        audio_url = url_match.group(1)

        # Download the MP3 file to a temporary file
        response = requests.get(audio_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(response.content)
            audio_file_path = tmp_file.name

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file and play it
        pygame.mixer.music.load(audio_file_path)
        pygame.mixer.music.play()

        # Keep the script running until the audio is finished playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Clean up the temporary file
        """ os.remove(audio_file_path) """
    else:
        print("Audio URL not found in the response.")


def get_constructed_sentence(last_words):
    global message_history
    print("Reconstruction from words")
    url = "https://api.openai.com/v1/chat/completions"

    # Headers including the Content-Type and Authorization with the Bearer token
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }

    message_history.append({"role": "user", "content": f'{last_words}'})

    # JSON payload as a Python dictionary
    data = {
        "model": "gpt-3.5-turbo-0125",
        "temperature":0.6,
        "messages": message_history
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=data)
    last_llm_response = response.json()["choices"][0]["message"]["content"]

    message_history.append({"role": "assistant", "content": last_llm_response})

    text_to_speech(last_llm_response)

def callback(data):
    global current_word_index

    data_list.append(data)

    if len(data_list) == 10:
        """ current_word_index = 0 """
        filename = word_list[current_word_index]
        create_embedding(data_list)
        current_word_index += 1
        if current_word_index == 34 :
            current_word_index = 0

unsubscribe = neurosity.brainwaves_raw(callback)


