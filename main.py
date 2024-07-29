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
from WordMatching import dtw  # Importando a função dtw
from Levenshtein import distance as levenshtein_distance
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

def compare_pronunciations(correct_pronunciation, user_pronunciation, threshold=2):
    distance = edit_distance_python2(correct_pronunciation, user_pronunciation)
    return distance <= threshold

def remove_punctuation_end(sentence):
    return sentence.rstrip('.')
#---------------------------------------------------------------------------

# Load sentences for random selection from data_de_en_fr.pickle
try:
    with open('data_de_en_fr.pickle', 'rb') as f:
        random_sentences_df = pickle.load(f)
    # Verificar se é um DataFrame e converter para lista de dicionários
    if isinstance(random_sentences_df, pd.DataFrame):
        random_sentences = random_sentences_df.to_dict(orient='records')
    else:
        random_sentences = random_sentences_df
except Exception as e:
    print(f"Erro ao carregar data_de_en_fr.pickle: {e}")
    random_sentences = []

# Load categorized sentences from frases_categorias.pickle
try:
    with open('frases_categorias.pickle', 'rb') as f:
        categorized_sentences = pickle.load(f)
except Exception as e:
    print(f"Erro ao carregar frases_categorias.pickle: {e}")
    categorized_sentences = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sentence', methods=['POST'])
def get_sentence():
    try:
        category = request.form.get('category', 'random')
        
        if category == 'random':
            if random_sentences:
                sentence = random.choice(random_sentences)
                sentence_text = remove_punctuation_end(sentence.get('fr_sentence', "Frase não encontrada."))
            else:
                return jsonify({"error": "Nenhuma frase disponível para seleção aleatória."}), 500
        else:
            if category in categorized_sentences:
                sentence_text = random.choice(categorized_sentences[category])
                sentence_text = remove_punctuation_end(sentence_text)
            else:
                return jsonify({"error": "Categoria não encontrada."}), 400

        return jsonify({'fr_sentence': sentence_text})
    
    except Exception as e:
        print(f"Erro no endpoint /get_sentence: {e}")
        return jsonify({"error": "Erro interno no servidor."}), 500

#---------------------------------------------------------------------------

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio']
    text = request.form['text']

    if not file:
        return jsonify({'ratio': 0, 'diff_html': '', 'pronunciations': {}, 'feedback': {}})

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

    # Estimativas e palavras reais
    words_estimated = normalized_transcription.split()
    words_real = normalized_text.split()

    # Calcular a matriz de alinhamento usando DTW
    alignment_matrix = dtw(words_real, words_estimated)
    
    # Extraia o caminho de alinhamento
    n, m = len(words_real), len(words_estimated)
    i, j = n, m
    aligned_pairs = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if alignment_matrix[i, j] == alignment_matrix[i-1, j-1] + levenshtein_distance(words_real[i-1], words_estimated[j-1]):
                aligned_pairs.append((words_real[i-1], words_estimated[j-1]))
                i, j = i - 1, j - 1
            elif alignment_matrix[i, j] == alignment_matrix[i-1, j] + levenshtein_distance(words_real[i-1], ""):
                aligned_pairs.append((words_real[i-1], ""))
                i -= 1
            else:
                aligned_pairs.append(("", words_estimated[j-1]))
                j -= 1
        elif i > 0:
            aligned_pairs.append((words_real[i-1], ""))
            i -= 1
        elif j > 0:
            aligned_pairs.append(("", words_estimated[j-1]))
            j -= 1

    aligned_pairs.reverse()

    # Generate HTML with color-coded words and feedback
    diff_html = []
    pronunciations = {}
    feedback = {}
    correct_count = 0
    incorrect_count = 0

    for real_word, mapped_word in aligned_pairs:
        correct_pronunciation = transliterate_and_convert(real_word)
        user_pronunciation = transliterate_and_convert(mapped_word)
        # Aqui, podemos ajustar o threshold para tornar a comparação mais rigorosa
        if compare_pronunciations(real_word, mapped_word, threshold=1):
            diff_html.append(f'<span class="word correct" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
            correct_count += 1
        else:
            diff_html.append(f'<span class="word incorrect" onclick="showPronunciation(\'{real_word}\')">{real_word}</span>')
            incorrect_count += 1
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

    # Calcula a taxa de acerto
    total_words = correct_count + incorrect_count
    ratio = (correct_count / total_words) * 100 if total_words > 0 else 0

    # Logging para depuração
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Total: {total_words}, Ratio: {ratio}")
    formatted_ratio = "{:.2f}".format(ratio)
    return jsonify({'ratio': formatted_ratio, 'diff_html': diff_html, 'pronunciations': pronunciations, 'feedback': feedback})







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
    app.run(debug=True)
