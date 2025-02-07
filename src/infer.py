# -*- coding: utf-8 -*-
"""infer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zRQ7cBeaOf93DSsha3cl0PyyZYPxGLlb
"""

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/BITAMIN/gen_question/requirements.txt
!pip install -r /content/drive/MyDrive/BITAMIN/gen_question/requirements.txt

!pip freeze > ../requirements.txt

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
        },
        {
            "context": "1592년 일본이 조선을 침략하면서 시작된 임진왜란은 조선과 일본 간의 전쟁으로, 명나라 또한 조선을 도와 참전하게 되었다. 전쟁 초기 일본군은 조선의 수도 한성을 점령하고, 급속도로 북진하며 개성을 거쳐 평양까지 진격하였다. 이에 맞서 조선군과 의병들은 전국 각지에서 항전하였으며, 특히 이순신 장군이 이끄는 조선 수군은 한산도 대첩과 명량 해전에서 큰 승리를 거두며 일본의 해상 보급로를 차단하는 데 성공했다. 그러나 전쟁은 장기화되었고, 결국 1598년 도요토미 히데요시가 사망한 후 일본군은 철수하였다. 이 전쟁은 조선과 일본 모두에게 큰 피해를 남겼으며, 이후 조선은 국방 강화를 위해 군제 개혁을 단행하였다.",
            "answer": "임진왜란"
        },
        {
            "context": "양자 역학은 20세기 초반에 등장한 물리학의 한 분야로, 미시 세계에서 입자와 에너지가 어떻게 상호작용하는지를 설명한다. 고전 물리학에서는 입자가 특정한 위치와 속도를 가지지만, 양자 역학에서는 입자의 상태가 확률적으로 결정되며, 관측하기 전까지는 특정한 값이 정해지지 않는다. 이 개념은 하이젠베르크의 불확정성 원리와 슈뢰딩거의 고양이 사고 실험을 통해 잘 알려져 있다. 또한, 양자 얽힘이라는 현상은 두 입자가 아무리 멀리 떨어져 있어도 서로 연결되어 있다는 것을 의미하며, 이는 현대 정보 기술에서 양자 컴퓨팅과 암호화 기술에 응용되고 있다.",
            "answer": "양자 역학"
        },
        {
            "context": "1789년 프랑스에서 시작된 프랑스 혁명은 전 세계적으로 민주주의와 자유의 이념을 확산시키는 계기가 되었다. 당시 프랑스는 경제적 어려움과 계급 간 불평등으로 인해 사회적 불만이 팽배해 있었으며, 특히 제3신분이라 불리는 평민 계층이 정치적 권리를 요구하며 혁명을 주도하였다. 바스티유 감옥 습격을 시작으로 혁명이 본격화되었고, 1792년 프랑스는 왕정을 폐지하고 공화국을 선언하였다. 이후 로베스피에르가 이끄는 공포 정치 시기가 있었으며, 수많은 귀족과 반혁명 인사들이 단두대에서 처형되었다. 결국 1799년 나폴레옹 보나파르트가 쿠데타를 일으켜 정권을 장악하며 혁명은 새로운 국면을 맞이하게 되었다.",
            "answer": "프랑스 혁명"
        },
        {
            "context": "1969년 7월 20일, 미국 항공우주국(NASA)의 아폴로 11호가 인류 최초로 달 착륙에 성공했다. 이 임무에서 닐 암스트롱과 버즈 올드린은 달 표면에 발을 디뎠으며, 암스트롱은 '이것은 한 인간에게는 작은 한 걸음이지만, 인류에게는 거대한 도약이다'라는 유명한 말을 남겼다. 아폴로 11호는 지구를 떠나 달로 향하는 동안 여러 차례 궤도 수정 작업을 수행하였으며, 착륙 후 약 21시간 동안 달 표면을 탐사하고 다양한 실험을 진행했다. 이들은 달에서 암석과 토양 샘플을 수집하여 지구로 가져왔으며, 이를 통해 과학자들은 달의 기원과 지질학적 특성을 연구할 수 있었다. 이 임무는 우주 탐사의 중요한 이정표가 되었으며, 이후 인류의 우주 탐사에 대한 가능성을 넓히는 계기가 되었다.",
            "answer": "아폴로 11호"
        },
        {
            "context": "유전자 편집 기술은 특정 유전자를 수정하거나 제거하여 생명체의 특성을 변화시키는 기술로, 최근 CRISPR-Cas9 시스템의 개발로 인해 혁신적인 발전을 이루었다. CRISPR-Cas9은 박테리아가 바이러스를 방어하기 위해 사용하는 메커니즘에서 착안한 기술로, 과학자들은 이를 이용하여 DNA를 정밀하게 편집할 수 있게 되었다. 이 기술은 유전 질환 치료, 농업 분야에서 작물 개량, 그리고 질병 연구에 널리 사용되고 있으며, 현재 여러 임상 시험에서도 적용되고 있다. 그러나 윤리적 문제도 제기되고 있으며, 특히 인간 배아 유전자 편집과 관련하여 규제와 논의가 계속되고 있다.",
            "answer": "CRISPR-Cas9"
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

