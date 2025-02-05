# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python (gen_que_env)
#     language: python
#     name: gen_que_env
# ---

# KOBART모델 
# 1. 한국어 사전학습 모델을 사용하자
# 2. context에서 n개의 키워드를 뽑아내자
# 3. n개의 키워드를 정답으로하는 문제를 출제하자
#
# ```jsx
# 입력:
# context: 서울은 대한민국의 수도로, 정치와 경제의 중심지이다.
# keywords: ["서울"]
#
# 출력:
# 1. 문제: 대한민국의 수도는 어디인가요?
#    정답: 서울
#
# -> 위와같은 결과를
# 질문 생성에 특화된 `Sehong/kobart-QuestionGeneration` 사용 내는 모델링 하기
# ```

# # 1. 환경 설정

# 가상환경을 통한 환경설정  
#     cd C:\Users\j2982\gen_qustion_env
#     Scripts\activate  

# +
#pip install transformers datasets

# +
#pip install transformers torch
# -

import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration



# # 2. KOBART모델 사용

# +
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def load_model(model_name="Sehong/kobart-QuestionGeneration"):
    """Kobart 모델 및 토크나이저 로드"""
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name) 
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model



# -

"""text = "1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의(폭력행위등처벌에관한법률위반)으로 지명수배되었다. 1989년 3월 12일 서울지방검찰청 공안부는 임종석의 사전구속영장을 발부받았다. 같은 해 6월 30일 평양축전에 임수경을 대표로 파견하여 국가보안법위반 혐의가 추가되었다. 경찰은 12월 18일~20일 사이 서울 경희대학교에서 임종석이 성명 발표를 추진하고 있다는 첩보를 입수했고, 12월 18일 오전 7시 40분 경 가스총과 전자봉으로 무장한 특공조 및 대공과 직원 12명 등 22명의 사복 경찰을 승용차 8대에 나누어 경희대학교에 투입했다. 1989년 12월 18일 오전 8시 15분 경 서울청량리경찰서는 호위 학생 5명과 함께 경희대학교 학생회관 건물 계단을 내려오는 임종석을 발견, 검거해 구속을 집행했다. 임종석은 청량리경찰서에서 약 1시간 동안 조사를 받은 뒤 오전 9시 50분 경 서울 장안동의 서울지방경찰청 공안분실로 인계되었다. <unused0> 1989년 2월 15일"

raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

summary_ids = model.generate(torch.tensor([input_ids]))
print(tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True))

# <unused0> is sep_token, sep_token seperate content and answer"""


"""
# 테스트 데이터 (여러 문단 + 정답)
test_data = [
    {
        "context": "서울은 대한민국의 수도로 정치, 경제, 문화의 중심지이다. 또한 한강이 흐르며 주요 행정 기관이 위치해 있다.",
        "keyword": "대한민국의 수도"
    },
    {
        "context": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것을 나타낸다.",
        "keyword": "피타고라스 정리"
    },
    {
        "context": "에디슨은 전구를 발명한 인물로 유명하며, 그는 전기를 실용적으로 사용하는 데 기여한 발명가이다.",
        "keyword": "전구"
    },
    {
        "context": "지구는 태양계의 세 번째 행성으로, 물과 생명체가 존재하는 유일한 행성이다.",
        "keyword": "태양계의 세 번째 행성"
    },
    {
        "context": "대한민국의 국화는 무궁화로, 이는 영원히 피고 또 피어난다는 의미를 가진다.",
        "keyword": "무궁화"
    }
]

for example in test_data:
    test_context = example["context"]
    test_keyword = example["keyword"]

    # 입력 형식 적용 (BOS/EOS 토큰 추가)
    input_text = f"{test_context} <unused0> {test_keyword}"
    raw_input_ids = tokenizer.encode(input_text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id] 

    # 모델 추론
    outputs = model.generate(
        torch.tensor([input_ids]), 
        max_length=50,
        num_beams=5,  # Beam Search 적용
        repetition_penalty=2.5,  # 반복 방지
        no_repeat_ngram_size=3,  # N-gram 반복 방지
        temperature=0.7,  # 출력 다양성 증가
        top_k=50,  # 상위 50개 단어 중 선택
        top_p=0.9,  # 누적 확률이 0.9가 되는 단어 중 선택
        early_stopping=True
    )

    # 생성된 질문 디코딩
    generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 결과 출력
    print("Context:", test_context)
    print("Generated Question:", generated_question)
    print("Answer:", test_keyword)
    print("-" * 80)"""

# # 3. 데이터준비(Fiine Tuning)

# pip install --upgrade transformers accelerate


# HuggingFace 코르쿼드(korQuad) 데이터셋 <PDF자료로 대체?>

# def preprocess_korquad_for_question_generation(examples):
#     inputs = []
#     targets = []
#
#     for context, question, answers in zip(examples["context"], examples["question"], examples["answers"]):
#         for answer_text in answers["text"]:
#             # 입력: context와 keyword(정답)
#             inputs.append(f"context: {context} keyword: {answer_text}")
#             # 출력: question
#             targets.append(question)
#
#     return {"input_text": inputs, "target_text": targets}
#
# # 데이터 전처리
# processed_dataset = dataset_sampled.map(preprocess_korquad_for_question_generation, batched=True)

# def tokenize_data(examples):
#     model_inputs = tokenizer(
#         examples["input_text"], max_length=512, padding="max_length", truncation=True
#     )
#     labels = tokenizer(
#         examples["target_text"], max_length=128, padding="max_length", truncation=True
#     )
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# # 토크나이징
# tokenized_dataset = processed_dataset.map(tokenize_data, batched=True)
#

# # 4. 모델링

# training_args = TrainingArguments(
#     output_dir="./kobart_finetuned",
#     evaluation_strategy="steps",    # 평가를 steps마다 실행
#     save_strategy="steps",          # 모델 저장도 steps마다 실행
#     save_steps=500,                 # 500 스텝마다 저장
#     eval_steps=500,                 # 500 스텝마다 평가
#     learning_rate=3e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=100,
#     load_best_model_at_end=True,    # 최적 모델 로드
# )


# from transformers import Trainer
#
# # Trainer 생성
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=validation_dataset,
#     tokenizer=tokenizer,
# )
#

# test_context = "서울은 대한민국의 수도이다."
# test_keyword = "대한민국"
#
# # 특별 토큰을 사용한 입력
# input_text = f"<context> {test_context} <keyword> {test_keyword}"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#
# # 모델 추론
# # 추론 시 설정 개선
# outputs = model.generate(
#     input_ids,
#     max_length=50,             # 출력 최대 길이
#     num_beams=5,               # Beam Search로 후보 문장 다양성 확보
#     repetition_penalty=2.5,    # 반복 패널티 추가
#     no_repeat_ngram_size=2,    # N-gram 반복 방지
#     early_stopping=True,       # 적절한 시점에 종료
# )
# print("Generated Question:", tokenizer.decode(outputs[0], skip_special_tokens=True))
# print("Answer:", keyword)

# for example in examples:
#     test_context = example["context"]
#     for test_keyword in example["keywords"]:
#         input_text = f"<context> {test_context} <keyword> {test_keyword}"
#         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#
#         # 모델 추론
#         outputs = model.generate(
#             input_ids,
#             max_length=50,
#             num_beams=5,
#             repetition_penalty=2.5,
#             no_repeat_ngram_size=3,
#             early_stopping=True,
#         )
#
#         # 질문과 정답 생성
#         generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print("Generated Question:", generated_question)
#         print("Answer:", test_keyword)
#         print()
