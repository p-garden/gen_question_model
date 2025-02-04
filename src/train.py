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

# +
from dataset import get_datasets
from model import load_model
from transformers import TrainingArguments, Trainer

# 모델 및 데이터 로드
tokenizer, model = load_model()
train_dataset, valid_dataset = get_datasets()  #  `dataset.py`에서 변환된 데이터 가져오기

# 데이터 전처리
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], 
        max_length=512, 
        padding="max_length", 
        truncation=True
    )
    
    labels = tokenizer(
        examples["target_text"], 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_dataset.map(preprocess_function, batched=True)
valid_tokenized = valid_dataset.map(preprocess_function, batched=True)


# +
# ✅ 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir="../saved_model/kobart_qg_finetuned",  # 모델 저장 경로
     evaluation_strategy="steps",    # 평가를 steps마다 실행
    save_strategy="steps",          # 모델 저장도 steps마다 실행
    save_steps=500,                 # 500 스텝마다 저장
    eval_steps=500,                 # 500 스텝마다 평가
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,    # 최적 모델 로드
)

# ✅ Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    tokenizer=tokenizer,
)

# ✅ 학습 실행
if __name__ == "__main__":
    print("🚀 모델 학습 시작...")
    trainer.train()
    
    # 학습된 모델 저장
    model.save_pretrained("../models/kobart_qg_finetuned")
    tokenizer.save_pretrained("../models/kobart_qg_finetuned")
    print("✅ 모델 학습 완료 및 저장됨!")
# -


