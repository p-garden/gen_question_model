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

TRAIN_SAMPLE_SIZE = 800  # 학습 데이터 샘플 개수
VALID_SAMPLE_SIZE = 200  # 검증 데이터 샘플 개수

train_dataset = train_dataset.shuffle(seed=42).select(range(min(len(train_dataset), TRAIN_SAMPLE_SIZE)))
valid_dataset = valid_dataset.shuffle(seed=42).select(range(min(len(valid_dataset), VALID_SAMPLE_SIZE)))

print(f"✅ 학습 데이터 개수: {len(train_dataset)}개")
print(f"✅ 검증 데이터 개수: {len(valid_dataset)}개")

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
    save_strategy="steps",          # 모델 저장도 steps마다 실|행
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
    model.save_pretrained("../saved_model/kobart_qg_finetuned")
    tokenizer.save_pretrained("../saved_model/kobart_qg_finetuned")
    print("✅ 모델 학습 완료 및 저장됨!")
# -

# 추가학습 코드 / 1000개 단위로 학습할 예정(random data)

# +
from dataset import get_datasets
from model import load_model
from transformers import TrainingArguments, Trainer

# ✅ 기존 학습된 모델 로드 (추가 학습)
MODEL_PATH = "../saved_model/kobart_qg_finetuned"  # 이전 학습된 모델 경로
tokenizer, model = load_model(MODEL_PATH)  # 기존 모델 불러오기
train_dataset, valid_dataset = get_datasets()  # 새로운 데이터셋 로드

# ✅ 추가 학습 시 데이터 일부만 사용 (샘플링)
TRAIN_SAMPLE_SIZE = 1000  # 학습 데이터 샘플 개수
VALID_SAMPLE_SIZE = 300  # 검증 데이터 샘플 개수

train_dataset = train_dataset.shuffle(seed=42).select(range(min(len(train_dataset), TRAIN_SAMPLE_SIZE)))
valid_dataset = valid_dataset.shuffle(seed=42).select(range(min(len(valid_dataset), VALID_SAMPLE_SIZE)))

print(f"✅ 추가 학습 데이터 개수: {len(train_dataset)}개")
print(f"✅ 검증 데이터 개수: {len(valid_dataset)}개")

train_tokenized = train_dataset.map(preprocess_function, batched=True)
valid_tokenized = valid_dataset.map(preprocess_function, batched=True)

# ✅ 추가 학습을 위한 TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="../saved_model/kobart_qg_finetuned_v2",  # 새로운 모델 저장 경로
    evaluation_strategy="steps",  # 매 steps마다 평가
    save_strategy="steps",          # 모델 저장도 steps마다 실|행
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # 추가 학습 3 Epoch
    save_steps=500,  # 500 스텝마다 모델 저장
    save_total_limit=2,  # 최신 모델 2개만 저장
    logging_dir="../logs",
    logging_steps=500,  # 500 스텝마다 로그 출력
    load_best_model_at_end=True,  # 가장 좋은 모델 불러오기
    metric_for_best_model="eval_loss",
    resume_from_checkpoint=True,  # ✅ 기존 체크포인트에서 이어서 학습
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
    print("🚀 추가 학습 시작...")
    trainer.train()
    
    # 학습된 모델 저장
    model.save_pretrained("../saved_model/kobart_qg_finetuned")
    tokenizer.save_pretrained("../saved_model/kobart_qg_finetuned")
    print("✅ 추가 학습 완료 및 저장됨!")

