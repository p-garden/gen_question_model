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
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# ✅ 학습된 모델 및 토크나이저 로드
MODEL_PATH = "../saved_model/kobart_qg_finetuned"  # 학습된 모델 경로
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()  # 평가 모드

def generate_question(context, answer):
    """
    문맥(context)과 정답(answer)을 입력하면 질문을 생성하는 함수
    """
    input_text = f"질문 생성: {answer} 문맥: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    # ✅ 모델 추론
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=50,
            num_beams=5,  # Beam Search 적용
            repetition_penalty=2.5,  # 같은 표현 반복 방지
            no_repeat_ngram_size=3,  # n-gram 반복 방지
            temperature=0.7,  # 샘플링 다양성 증가
            top_k=50,  # 확률 높은 50개 중 선택
            top_p=0.9,  # 확률 누적 90% 내에서 선택
            early_stopping=True
        )

    # ✅ 생성된 질문 디코딩
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ 테스트 실행
if __name__ == "__main__":
    print("🚀 질문 생성 테스트 시작!")

    test_data = [
        {
            "context": "서울은 대한민국의 수도로 정치, 경제, 문화의 중심지이다. 또한 한강이 흐르며 주요 행정 기관이 위치해 있다.",
            "answer": "대한민국의 수도"
        },
        {
            "context": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것을 나타낸다.",
            "answer": "피타고라스 정리"
        },
        {
            "context": "에디슨은 전구를 발명한 인물로 유명하며, 그는 전기를 실용적으로 사용하는 데 기여한 발명가이다.",
            "answer": "전구"
        }
    ]

    for example in test_data:
        test_context = example["context"]
        test_answer = example["answer"]
        generated_question = generate_question(test_context, test_answer)

        print("\n🎯 Context:", test_context)
        print("✅ Generated Question:", generated_question)
        print("📝 Answer:", test_answer)
        print("-" * 80)

    print("🚀 질문 생성 테스트 완료!")

