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


# -

# ✅ 테스트 실행
if __name__ == "__main__":
    print("🚀 질문 생성 테스트 시작!")

    test_data = [
        {
            "context": "조선왕조는 1392년 이성계가 건국한 조선에서 1897년 고종이 대한제국을 선포하기까지의 시기를 말한다. 조선은 유교를 국교로 삼았으며, 성리학을 중심으로 한 학문과 교육이 발달하였다. 또한, 조선 시대에는 한글이 창제되었으며, 세종대왕의 업적 중 하나로 꼽힌다. 조선은 사대교린 정책을 기본으로 하여 중국과 우호적인 관계를 유지하는 한편, 일본 및 여진족과의 외교에도 신경을 썼다.",
            "answer": "세종대왕"
        }, 
        {
            "context": "뉴턴의 운동 법칙은 고전 역학의 기본 원리로서 물체의 운동을 설명하는 데 사용된다. 제1법칙(관성의 법칙)은 외부에서 힘이 작용하지 않으면 정지한 물체는 계속 정지해 있고, 운동하는 물체는 같은 속도로 계속 운동한다는 것이다. 제2법칙(가속도의 법칙)은 물체에 작용하는 힘이 클수록 더 큰 가속도가 발생하며, 질량이 클수록 가속도가 작아진다는 원리를 설명한다. 마지막으로, 제3법칙(작용과 반작용의 법칙)은 한 물체가 다른 물체에 힘을 가하면, 반대 방향으로 같은 크기의 힘이 작용한다는 원칙이다.",
            "answer": "뉴턴의 운동 법칙"
        },
        {
            "context": "인체 면역 시스템은 우리 몸을 외부 병원체로부터 보호하는 역할을 한다. 면역 체계는 선천 면역과 후천 면역으로 구분되며, 선천 면역은 태어날 때부터 갖추고 있는 자연 방어 기제이다. 반면, 후천 면역은 감염이나 백신 접종을 통해 형성되며, 특정 항원에 대해 기억을 형성하여 다음 공격 시 더 강한 면역 반응을 유도한다. 백혈구는 이러한 면역 반응에서 중요한 역할을 하며, 특히 T세포와 B세포는 항체를 생성하거나 감염된 세포를 직접 제거하는 기능을 한다.",
            "answer": "면역 체계"
        },
        {
            "context": "수요와 공급의 법칙은 시장 경제에서 가격이 형성되는 원리를 설명하는 기본 개념이다. 수요가 증가하면 가격이 상승하고, 반대로 공급이 증가하면 가격이 하락하는 경향이 있다. 시장에서 균형 가격은 수요량과 공급량이 일치하는 지점에서 형성된다. 또한, 가격 탄력성의 개념을 통해 특정 상품의 가격 변화에 대한 수요의 반응 정도를 측정할 수 있다.",
            "answer": "수요와 공급의 법칙"
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



