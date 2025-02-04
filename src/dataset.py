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

# 파인 튜닝하기위한 데이터셋 준비 과정
# KorQuAD 데이터셋 불러와 GQ태스크에 적합한 형식으로 변환
# `questoin + Text` 형식으로 변환  
#
# {     
#     "context": "서울은 대한민국의 수도이며 정치, 경제, 문화의 중심지이다."  ,
#     "question": "대한민국의 수도는 어디인가요?  ",
#     "answers"  : {
#         "text": ["서  울"],
#         "answer_start"  : [0]
#   
#     ->  
# {  
#     "input_text": "질문 생성: 서울 문맥: 서울은 대한민국의 수도이며 정치, 경제, 문화의 중심지이다."  ,
#     "target_text": "대한민국의 수도는 어디인가요    ?"  
# }
#

# +
from datasets import load_dataset

def get_datasets():
    """Hugging Face Hub에서 KorQuAD 데이터셋을 로드하고 질문 생성(QG) 형식으로 변환"""
    dataset = load_dataset("KorQuAD/squad_kor_v1")

    def preprocess_korquad(examples):
        inputs = []
        targets = []

        for context, question, answers in zip(examples["context"], examples["question"], examples["answers"]):
            for answer_text in answers["text"]:
                inputs.append(f"정답: {answer_text} 문맥: {context}")
                targets.append(question)

        return {"input_text": inputs, "target_text": targets}

    train_dataset = dataset["train"].map(preprocess_korquad, batched=True, remove_columns=dataset["train"].column_names)
    valid_dataset = dataset["validation"].map(preprocess_korquad, batched=True, remove_columns=dataset["validation"].column_names)

   
    
    return train_dataset, valid_dataset
if __name__ == "__main__":
    train_dataset, valid_dataset = get_datasets()
    print("✅ 데이터 변환 완료!")

    print("\n✅ 변환된 Train Dataset 샘플:")
    print(f"input_text: {train_dataset[0]['input_text']}")
    print(f"target_text: {train_dataset[0]['target_text']}")
# -




