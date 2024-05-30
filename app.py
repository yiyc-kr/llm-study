from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from kiwipiepy import Kiwi
kiwi = Kiwi()
import nltk

app = Flask(__name__)

# 모델 디렉토리 설정
model_dir_ko = "ko_en_checkpoint"
model_dir_en = "en_ko_checkpoint-56250"

# 한국어 및 영어 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir_ko, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir_ko)
# tokenizer_en = AutoTokenizer.from_pretrained(model_dir_en, use_fast=False)
# model_en = AutoModelForSeq2SeqLM.from_pretrained(model_dir_en)
# tokenizer = tokenizer_ko
# model = model_ko

@app.route('/translate/ko', methods=['POST'])
def translate():
    data = request.json
    print("DATA:", data)
    print("JSON", request.json)
    text = data.get('text')
    # language = data.get('language')  # "ko" 또는 "en"

    if not text:
        return jsonify({"error": "Invalid input"}), 400

    # 텍스트를 줄바꿈 문자로 분할
    lines = text.strip().split('\n')
    print("TEXT:", text)
    sentence_list = []
    for line in lines:
        line_sentences = kiwi.split_into_sents(line)
        line_sentences = [sentence[0] for sentence in line_sentences]
        print("LINE_SENTENCES:", line_sentences)
        sentence_list.extend(line_sentences)

    

    # elif language == "en":
    #     sentence_list = []
    #     for line in lines:
    #         line_sentences = nltk.sent_tokenize(line)
    #         sentence_list.extend(line_sentences)

        # tokenizer = tokenizer_en
        # model = model_en
    # else:
    #     return jsonify({"error": "Unsupported language"}), 400

    # 텍스트를 토크나이징하고 모델을 통해 번역 수행
    translated_text = []
    print(sentence_list)
    for sentence in sentence_list:
        tokens = tokenizer.encode(sentence, return_tensors="pt", max_length=64, truncation=True)
        output = model.generate(tokens, max_length=64)
        translated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        translated_text.append(translated_sentence)
        print(translated_sentence)
    
    print(translated_text)
    return jsonify({"translated_text": " ".join(translated_text)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)