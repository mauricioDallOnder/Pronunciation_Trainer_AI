import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Verifica se o g2pk e unidecode estão instalados, caso contrário, instala
try:
    import g2pk
except ImportError:
    install('g2pk')

try:
    from unidecode import unidecode
except ImportError:
    install('unidecode')

from flask import Flask, request, render_template, jsonify, send_file
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import epitran
import re
import os
import tempfile
import wave
import numpy as np
import pickle
import random
import pandas as pd
from gtts import gTTS
from WordMetrics import edit_distance_python2
from WordMatching import get_best_mapped_words

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# Load the French SST Model
model_name = "facebook/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Initialize Epitran for French
epi = epitran.Epitran('fra-Latn')

def get_pronunciation(word):
    return epi.transliterate(word)

# Mapeamento ajustado de fonemas franceses para português
french_to_portuguese_phonemes = {
    'ɑ̃': 'an',   # como em 'pão'
    'ɛ̃': 'en',   # como em 'vem'
    'ɔ̃': 'on',   # como em 'som'
    'œ̃': 'an',   # como em 'um'
    'ʃ': 'ch',    # como em 'chave'
    'ʒ': 'j',     # como em 'jogo'
    'ʀ': 'r',     # como em 'carro'
    'ɥ': 'u',     # como em 'qu'
    'ø': 'eu',    # como em 'seu'
    'œ': 'eu',    # como em 'mel'
    'ə': 'e',     # como em 'ele'
    'ɑ': 'a',     # como em 'pá'
    'ɔ': 'o',     # como em 'só'
    'e': 'e',     # como em 'ele'
    'ɛ': 'é',     # como em 'pé'
    'i': 'i',     # como em 'si'
    'o': 'ô',     # como em 'avô'
    'u': 'u',     # como em 'luz'
    'j': 'i',     # como em 'mais'
    'w': 'u',     # como em 'qu'
    'm': 'm',     # como em 'mão'
    'n': 'n',     # como em 'nó'
    'p': 'p',     # como em 'pá'
    'b': 'b',     # como em 'boca'
    'd': 'd',     # como em 'dado'
    'f': 'f',     # como em 'foca'
    'v': 'v',     # como em 'vó'
    's': 's',     # como em 'sapo'
    'z': 'z',     # como em 'zebra'
    't': 't',     # como em 'tá'
    'k': 'c',     # como em 'casa'
    'g': 'g',     # como em 'gato'
    'l': 'l',     # como em 'lago'
    'ʁ': 'r',     # como em 'carro'
}

def convert_pronunciation_to_portuguese(pronunciation):
    words = pronunciation.split()
    pronunciation_mapped = []
    for word in words:
        mapped_word = []
        i = 0
        while i < len(word):
            match = None
            for length in range(3, 0, -1):  # Verifica fonemas de 3 caracteres, 2 caracteres e 1 caractere
                phoneme = word[i:i + length]
                if phoneme in french_to_portuguese_phonemes:
                    match = french_to_portuguese_phonemes[phoneme]
                    mapped_word.append(match)
                    i += length
                    break
            if not match:
                mapped_word.append(unidecode(word[i]))
                i += 1
        pronunciation_mapped.append(''.join(mapped_word))
    return ' || '.join(pronunciation_mapped)

def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def transliterate_and_convert(word):
    pronunciation = get_pronunciation(word)
    pronunciation_pt = convert_pronunciation_to_portuguese(pronunciation)
    return pronunciation_pt

# Load sentences from pickle file and categorize them
with open('data_de_en_fr.pickle', 'rb') as f:
    sentences_df = pickle.load(f)

# Convert DataFrame to list of dictionaries
sentences = sentences_df.to_dict(orient='records')

# Ensure the sentences are in the expected format (list of dictionaries)
if isinstance(sentences, list) and all(isinstance(s, dict) for s in sentences):
    easy_sentences = [s for s in sentences if len(s['fr_sentence'].split()) <= 5]
    medium_sentences = [s for s in sentences if 5 < len(s['fr_sentence'].split()) <= 10]
    hard_sentences = [s for s in sentences if len(s['fr_sentence'].split()) > 10]
else:
    raise ValueError("Unexpected data format in the pickle file. Expected a list of dictionaries.")

def remove_punctuation_end(sentence):
    return sentence.rstrip('.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sentence', methods=['POST'])
def get_sentence():
    difficulty = request.form['difficulty']
    if difficulty == 'easy':
        sentence = random.choice(easy_sentences)
    elif difficulty == 'medium':
        sentence = random.choice(medium_sentences)
    else:
        sentence = random.choice(hard_sentences)
    sentence['fr_sentence'] = remove_punctuation_end(sentence['fr_sentence'])
    return jsonify(sentence)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio']
    text = request.form['text']

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    try:
        # Read the audio file using wave module
        with wave.open(tmp_file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            waveform = wav_file.readframes(num_frames)
            waveform = np.frombuffer(waveform, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize audio
            waveform = torch.tensor(waveform).unsqueeze(0)  # Add batch dimension

        # Resample if necessary
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        waveform = waveform.squeeze(0)  # Remove the batch dimension for a single example
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    finally:
        os.remove(tmp_file_path)  # Clean up the temporary file

    # Normalize texts
    normalized_transcription = normalize_text(transcription)
    normalized_text = normalize_text(text)

    # Calculate word mappings
    words_estimated = normalized_transcription.split()
    words_real = normalized_text.split()
    if len(words_real) == 0:
        ratio = 0
    else:
        mapped_words, mapped_words_indices = get_best_mapped_words(words_estimated, words_real)
        ratio = len([i for i in range(len(words_real)) if i < len(mapped_words) and words_real[i] == mapped_words[i]]) / len(words_real)
    
    # Generate HTML with color-coded words
    diff_html = []
    pronunciations = {}
    feedback = {}
    for real_word, mapped_word in zip(words_real, mapped_words):
        correct_pronunciation = transliterate_and_convert(real_word)
        user_pronunciation = transliterate_and_convert(mapped_word)
        if real_word == mapped_word:
            diff_html.append(f'<span class="word correct" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
        else:
            diff_html.append(f'<span class="word incorrect" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
            feedback[real_word] = {
                'correct': correct_pronunciation,
                'user': user_pronunciation,
                'suggestion': f"Tente pronunciar '{real_word}' como '{correct_pronunciation}'"
            }
        pronunciations[real_word] = {
            'correct': correct_pronunciation,
            'user': user_pronunciation
        }
    diff_html = ' '.join(diff_html)
    
    # Calculate edit distance
    edit_dist = edit_distance_python2(normalized_transcription, normalized_text)

    return jsonify({'ratio': ratio, 'diff_html': diff_html, 'pronunciations': pronunciations, 'feedback': feedback, 'edit_distance': edit_dist})

@app.route('/pronounce', methods=['POST'])
def pronounce():
    text = request.form['text']
    words = text.split()
    pronunciations = [transliterate_and_convert(word) for word in words]
    return jsonify({'pronunciations': ' '.join(pronunciations)})

@app.route('/speak', methods=['POST'])
def speak():
    text = request.form['text']
    tts = gTTS(text=text, lang='fr')
    file_path = tempfile.mktemp(suffix=".mp3")
    tts.save(file_path)
    return send_file(file_path, as_attachment=True, mimetype='audio/mp3')

if __name__ == '__main__':
    app.run(port=5000)
