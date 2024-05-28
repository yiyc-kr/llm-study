from konlpy.tag import Kkma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 형태소 분석기 로드
kkma = Kkma()

# 모델 디렉토리 설정
model_dir = "ko_en_checkpoint"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# 긴 한글 문장 예제
long_text = """
문명 붕괴 45년 후,
황폐해진 세상 속 누구에게도 알려지지 않은
풍요가 가득한 ‘녹색의 땅’에서 자란 ‘퓨리오사’(안야 테일러-조이)는
바이커 군단의 폭군 ‘디멘투스’(크리스 헴스워스)의 손에 모든 것을 잃고 만다.

가족도 행복도 모두 빼앗기고 세상에 홀로 내던져진 ‘퓨리오사’는
반드시 고향으로 돌아가겠다는 어머니와의 약속을 지키기 위해
인생 전부를 건 복수를 시작하는데...

‘매드맥스’ 시리즈의 전설적인 사령관 ‘퓨리오사’의 대서사시
5월 22일, 마침내 분노가 깨어난다!
"""

# 텍스트를 줄바꿈 문자로 분할
lines = long_text.strip().split('\n')

# 줄바꿈 문자로 분할된 각 줄을 문장 단위로 다시 분할
sentence_list = []
for line in lines:
    line_sentences = kkma.sentences(line)
    sentence_list.extend(line_sentences)

# 각 문장을 번역
# translated_sentences = []
for sentence in sentence_list:
    tokens = tokenizer.encode(sentence, return_tensors="pt", max_length=64, truncation=True)
    output = model.generate(tokens, max_length=64)
    translated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    # translated_sentences.append(translated_sentence)
    print(translated_sentence)

# 번역된 문장 결합
# final_translation = " ".join(translated_sentences)
# print(final_translation)


#### Result:
# Five years after the collapse of civilization, 45 years later, the collapse of civilization.
# It has not been known to anyone in the deserted world.
# "Prie Lee Osa" (Ahn Ya Taylor-Jo), who grew up in the "green land" full of wealth, is a "green land" filled with wealth.
# The man of the Bugg's gang, "D Mentus" (Chris Huffer) loses everything.
# "Pie Lee Osa," who lost all the family and happiness and was thrown out of the world alone, is "Ppee Lee Osa," who was lost from the world alone.
# To keep the promise with my mother that he will always return to his hometown, he will keep his promise to return to his hometown.
# I start to resent the revenge that has all of my life.
# The epic epic of "Prie Osa," the legendary commander of the "Maddmax" series, is a epic epic of "Prie Osa".
# On May 22nd, anger finally awakens.