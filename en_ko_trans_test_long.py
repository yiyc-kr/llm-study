from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# NLTK 다운로드 (최초 실행 시 필요)
# nltk.download('punkt')

# 모델 디렉토리 설정
model_dir = "en_ko_checkpoint-56250"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

# 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# 테스트를 위해 샘플 텍스트를 토크나이징하고 모델을 통해 추론
sample_text = """
A bank is a financial institution that accepts deposits from the public and creates a demand deposit while simultaneously making loans. Lending activities can be directly performed by the bank or indirectly through capital markets. Whereas banks play an important role in financial stability and the economy of a country, most jurisdictions exercise a high degree of regulation over banks. Most countries have institutionalized a system known as fractional-reserve banking, under which banks hold liquid assets equal to only a portion of their current liabilities. In addition to other regulations intended to ensure liquidity, banks are generally subject to minimum capital requirements based on an international set of capital standards, the Basel Accords. Banking in its modern sense evolved in the fourteenth century in the prosperous cities of Renaissance Italy but, in many ways, functioned as a continuation of ideas and concepts of credit and lending that had their roots in the ancient world. In the history of banking, a number of banking dynasties – notably, the Medicis, the Fuggers, the Welsers, the Berenbergs, and the Rothschilds – have played a central role over many centuries. The oldest existing retail bank is Banca Monte dei Paschi di Siena (founded in 1472), while the oldest existing merchant bank is Berenberg Bank (founded in 1590).
"""

# 문장 단위로 텍스트 분할
sentences = nltk.sent_tokenize(sample_text)

# 각 문장을 번역
# translated_sentences = []
for sentence in sentences:
    tokens = tokenizer.encode(sentence, return_tensors="pt", max_length=64, truncation=True)
    output = model.generate(tokens, max_length=64)
    translated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    # translated_sentences.append(translated_sentence)
    print(translated_sentence)

# 번역된 문장 결합
# final_translation = " ".join(translated_sentences)
# print(final_translation)
