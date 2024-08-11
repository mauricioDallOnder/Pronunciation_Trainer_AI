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

# Instalação de pacotes adicionais
try:
    import epitran
except ImportError:
    install('epitran')

try:
    from flask import Flask, request, render_template, jsonify, send_file
except ImportError:
    install('flask')

try:
    import torch
    import torchaudio
except ImportError:
    install('torch')
    install('torchaudio')

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except ImportError:
    install('transformers')

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Configure o backend para não-GUI
except ImportError:
    install('matplotlib')
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Configure o backend para não-GUI

try:
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
    from WordMatching import get_best_mapped_words, dtw
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    install('gtts')
    install('python-Levenshtein')

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# Load the French SST Model
model_name = "facebook/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load or initialize performance data
performance_file = 'performance_data.pkl'

def load_performance_data():
    if os.path.exists(performance_file):
        with open(performance_file, 'rb') as f:
            return pickle.load(f)
    else:
        return []

def save_performance_data(data):
    with open(performance_file, 'wb') as f:
        pickle.dump(data, f)

performance_data = load_performance_data()

# Initialize Epitran for French
epi = epitran.Epitran('fra-Latn')

def get_pronunciation(word):
    return epi.transliterate(word)

# Funções para melhorar a transcrição fonética

def omit_schwa(pronunciation):
    # Verifica a presença de sequências que costumam omitir o 'e'
    if "ement" in pronunciation:
        return pronunciation.replace("ement", "mã")
    return pronunciation

def normalize_vowels(pronunciation):
    # Normaliza vogais para consistência
    pronunciation = pronunciation.replace("ó", "ô")  # Exemplo de ajuste
    return pronunciation

def handle_special_cases(pronunciation):
    # Regras especiais para contextos específicos
    if "informations" in pronunciation:
        pronunciation = pronunciation.replace("ʒ", "j")
    return pronunciation

def convert_pronunciation_to_portuguese(pronunciation):
    pronunciation = omit_schwa(pronunciation)
    pronunciation = normalize_vowels(pronunciation)
    pronunciation = handle_special_cases(pronunciation)
    words = pronunciation.split()
    pronunciation_mapped = []
    for word in words:
        mapped_word = []
        i = 0
        while i < len(word):
            match = None
            for length in range(3, 0, -1):
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

# Mapeamento ajustado de fonemas franceses para português
french_to_portuguese_phonemes = {
    'ɑ̃': 'ã',   # Como em "mãe"
    'ɛ̃': 'ẽ',   # Como em "tem"
    'ɔ̃': 'õ',   # Como em "bom"
    'œ̃': 'ã',   # Similar a "mãe" (mais próximo do som nasal)
    'ʃ': 'ch',   # Como em "chá"
    'ʒ': 'j',    # Como em "jeito"
    'ʀ': 'rr',   # Como em "carro" (r uvular, som padrão em Portugal e comum em algumas regiões do Brasil)
    'ɥ': 'ui',   # Semivogal, como em "huit" (aproximado ao som de "ui" no português, ex: "fui")
    'ø': 'ê',    # Como em "peur" (intermediário entre "é" e "ê")
    'œ': 'é',    # Como em "fleur" (intermediário entre "é" e "ê")
    'ə': 'e',    # Como em "le" (vogal neutra, semelhante ao "e" átono em "cabeça")
    'ɑ': 'a',    # Como em "pá" (som mais aberto)
    'ɔ': 'ó',    # Como em "só"
    'e': 'ê',    # Como em "vê"
    'ɛ': 'é',    # Como em "pé"
    'i': 'i',    # Como em "vi"
    'o': 'ô',    # Como em "pôr"
    'u': 'u',    # Como em "tu"
    'j': 'i',    # Como em "sim" (semivogal, ex: "pai")
    'w': 'u',    # Como em "quase" (semivogal, ex: "quarto")
    'm': 'm',    # Como em "mão"
    'n': 'n',    # Como em "não"
    'p': 'p',    # Como em "pato"
    'b': 'b',    # Como em "boca"
    'd': 'd',    # Como em "dado"
    'f': 'f',    # Como em "faca"
    'v': 'v',    # Como em "vaca"
    's': 's',    # Como em "sapo" (s surdo)
    'z': 'z',    # Como em "zero" (s sonoro)
    't': 't',    # Como em "taco"
    'k': 'k',    # Como em "casa" (som de /k/)
    'g': 'g',    # Como em "gato"
    'l': 'l',    # Como em "lado" (l claro)
    'ʁ': 'rr',   # Como em "carro" (r uvular)
}

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

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
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
            waveform = np.frombuffer(waveform, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.tensor(waveform).unsqueeze(0)

        # Resample if necessary
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        waveform = waveform.squeeze(0)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    finally:
        os.remove(tmp_file_path)

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
        if compare_pronunciations(real_word, mapped_word):
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

    # Calcula a taxa de acerto e completude
    total_words = correct_count + incorrect_count
    ratio = (correct_count / total_words) * 100 if total_words > 0 else 0
    completeness_score = (len(words_estimated) / len(words_real)) * 100

    # Armazena os resultados diários
    performance_data.append({
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'correct': correct_count,
        'incorrect': incorrect_count,
        'ratio': ratio,
        'completeness_score': completeness_score,
        'sentence': text
    })
    save_performance_data(performance_data)

    # Logging para depuração
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Total: {total_words}, Ratio: {ratio}")
    formatted_ratio = "{:.2f}".format(ratio)
    formatted_completeness = "{:.2f}".format(completeness_score)

    return jsonify({
        'ratio': formatted_ratio,
        'diff_html': diff_html,
        'pronunciations': pronunciations,
        'feedback': feedback,
        'completeness_score': formatted_completeness
    })

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

@app.route('/performance', methods=['GET'])
def performance():
    # Agrupar dados por data
    df = pd.DataFrame(performance_data)
    grouped = df.groupby('date').agg({
        'correct': 'sum',
        'incorrect': 'sum',
        'ratio': 'mean'
    }).reset_index()

    dates = grouped['date']
    corrects = grouped['correct']
    incorrects = grouped['incorrect']
    ratios = grouped['ratio']

    x = np.arange(len(dates))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width / 2, corrects, width, label='Acertos', color='green')
    rects2 = ax.bar(x + width / 2, incorrects, width, label='Erros', color='red')

    ax.set_xlabel('Data')
    ax.set_ylabel('Percentagem')
    ax.set_title('Desempenho Diário')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45)  # Rotaciona os labels do eixo X para melhor leitura
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()

    graph_path = 'static/performance_graph.png'
    plt.savefig(graph_path, bbox_inches='tight')  # Ajusta o gráfico para que tudo fique visível
    plt.close()

    return send_file(graph_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
