from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델 디렉토리 설정
model_dir = "en_ko_checkpoint-56250"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

# 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# 테스트를 위해 샘플 텍스트를 토크나이징하고 모델을 통해 추론
sample_text = """
I got my peaches out in Georgia
"""
tokens = tokenizer.encode(sample_text, return_tensors="pt", max_length=64, truncation=True)
output = model.generate(tokens, max_length=64)

# 생성된 텍스트 디코딩
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
