from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from kiwipiepy import Kiwi
kiwi = Kiwi()
import nltk
import os

app = Flask(__name__)

# 모델 디렉토리 설정
model_dir_ko = "ko_en_checkpoint"
model_dir_en = "en_ko_checkpoint-37500"

# 한국어 및 영어 토크나이저 및 모델 로드
tokenizer_ko = AutoTokenizer.from_pretrained(model_dir_ko, use_fast=False)
model_ko = AutoModelForSeq2SeqLM.from_pretrained(model_dir_ko)
tokenizer_en = AutoTokenizer.from_pretrained(model_dir_en, use_fast=False)
model_en = AutoModelForSeq2SeqLM.from_pretrained(model_dir_en)

# 파일 경로 설정
original_texts_file_ko = 'original_texts_ko_en.tsv'
translations_file_ko = 'translations_ko_en.tsv'
original_texts_file_en = 'original_texts_en_ko.tsv'
translations_file_en = 'translations_en_ko.tsv'

def initialize_files():
    # 원문 텍스트 파일 초기화
    if not os.path.exists(original_texts_file_ko):
        with open(original_texts_file_ko, 'w', encoding='utf-8') as f:
            f.write("ko\ten\n")
    if not os.path.exists(original_texts_file_en):
        with open(original_texts_file_en, 'w', encoding='utf-8') as f:
            f.write("en\tko\n")

    # 번역 파일 초기화
    if not os.path.exists(translations_file_ko):
        with open(translations_file_ko, 'w', encoding='utf-8') as f:
            f.write("ko\ten\n")
    if not os.path.exists(translations_file_en):
        with open(translations_file_en, 'w', encoding='utf-8') as f:
            f.write("en\tko\n")

def save_original_text(source_text:str, translated_text:str, language:str):
    if language == 'ko':
        with open(original_texts_file_ko, 'a', encoding='utf-8') as f:
            f.write(f"{source_text}\t{translated_text}\n")
    elif language == 'en':
        with open(original_texts_file_en, 'a', encoding='utf-8') as f:
            f.write(f"{source_text}\t{translated_text}\n")

def save_translation(source_text:str, translated_text:str, language:str):
    if language == 'ko':
        with open(translations_file_ko, 'a', encoding='utf-8') as f:
            f.write(f"{source_text}\t{translated_text}\n")
    elif language == 'en':
        with open(translations_file_en, 'a', encoding='utf-8') as f:
            f.write(f"{source_text}\t{translated_text}\n")

def get_split_text(request):
    # TODO: apply logging
    # print("DATA:", data)
    # print("JSON", request.json)
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "Invalid input"}), 400
    
    sentence_list = text.strip().split('\n')

    return sentence_list

@app.route('/translate/en', methods=['POST'])
def translate_en():
    sentence_list = get_split_text(request)

    splited_sentence_list = []
    for sentence in sentence_list:
        splited_list = nltk.sent_tokenize(sentence)
        # print("splited_list:", splited_list)
        splited_sentence_list.extend(splited_list)
    
    # 텍스트를 토크나이징하고 모델을 통해 번역 수행
    translated_list = []
    # print(splited_sentence_list)
    for sentence in splited_sentence_list:
        tokens = tokenizer_en.encode(sentence, return_tensors="pt", max_length=64, truncation=True)
        output = model_en.generate(tokens, max_length=64)
        translated_sentence = tokenizer_en.decode(output[0], skip_special_tokens=True)
        print(sentence)
        save_translation(sentence, translated_sentence, 'en')
        translated_list.append(translated_sentence)
    
    print(translated_list)
    save_original_text(' '.join(sentence_list), ' '.join(translated_list), 'en')
    return jsonify({"translated_text": " ".join(translated_list)})

@app.route('/translate/ko', methods=['POST'])
def translate_ko():
    sentence_list = get_split_text(request)
    
    # print("TEXT:", text)
    splited_sentence_list = []
    for sentence in sentence_list:
        splited_list = kiwi.split_into_sents(sentence)
        splited_list = [sentence[0] for sentence in splited_list]
        # print("splited_list:", splited_list)
        splited_sentence_list.extend(splited_list)

    # 텍스트를 토크나이징하고 모델을 통해 번역 수행
    translated_list = []
    # print(splited_sentence_list)
    for sentence in splited_sentence_list:
        tokens = tokenizer_ko.encode(sentence, return_tensors="pt", max_length=64, truncation=True)
        output = model_ko.generate(tokens, max_length=64)
        translated_sentence = tokenizer_ko.decode(output[0], skip_special_tokens=True)
        save_translation(sentence, translated_sentence, 'ko')
        translated_list.append(translated_sentence)
        # print(translated_sentence)
    
    print(translated_list)
    save_original_text(' '.join(sentence_list), ' '.join(translated_list), 'ko')
    return jsonify({"translated_text": " ".join(translated_list)})

if __name__ == '__main__':
    initialize_files()
    app.run(host='0.0.0.0', port=5000)