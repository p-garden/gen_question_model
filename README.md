# 🏆  비타민 겨울방학 프로젝트
## 📌 주제: 학습자료를 바탕으로 문제 생성 모델
### ⚙️ 프로세스 (Processing Flow)
#### 📝 1. 이미지 처리 (PDF 학습자료 OCR 변환)
	📂 입력: PDF 학습자료
	🖼 처리: OCR을 통해 텍스트 변환 (EasyOCR)
	📊 결과: 일반 텍스트, 표, 차트 분석

#### 📝 2. 텍스트 처리 (요약 및 주요 정보 추출)
	🔍 요약 및 키워드 추출: KeyBERT, KoBERT 활용
	📌 핵심 문장 분석: 중요 내용 도출

#### 📝 3. 질문 생성 (Question Generation)
	 💡 빈칸 문제 생성: KoBART 활용
	🧐 주관식 문제 생성: KoBART + 문장 유사도 검사
	✅ O/X 문제 생성: 핵심 키워드 기반 생성

### 🏗 시스템 아키텍처 (System Architecture)
	mermaid
	graph TD
	    📂Input(학습자료.pdf)--레이아웃 및 텍스트 추출--> 🖼YOLO[DocLayout-YOLO]
	    
	    subgraph Layout_Analysis
	        YOLO--📄일반 텍스트--> 📝Text[EasyOCR]
	        YOLO--📊표--> 📋Table[성능 개선 시도 중]
	        YOLO--📉차트--> 📈Chart[kor_deplot]
	        Text --> 📂result1(result1.json)
			    Table --> 📂result1
			    Chart --> 📂result1
	    end
	    
	    result1 --🔍키워드--> 🎯KeyBERT
	    result1 --🧐핵심 문장--> 🏆KoBERT
	
	    subgraph NLP_Processing
	        KeyBERT --> 📂result2(result2.json)
	        KoBERT --> 📂result2
	    end
	
	    subgraph Question_Generation
	        result2 --✍빈칸<br>{context, answer}--> 🤖GQ1[KoBART]
	        result2 --📝주관식--> 🤖GQ2[KOBART + 문장유사도검사]
	        result2 --✅O/X--> 🤖GQ3[KeyWord 추출]
	        GQ1 --> 🎯result3
	        GQ2 --> 🎯result3
	        GQ3 --> 🎯result3
	    end
### 🧩 역할: 주관식 문제 생성 모델 개발
#### 🏗 모델 기반 질문 생성
	🎯 문장 유사도를 활용한 질문 품질 개선
	✅ 정확도 높은 질문 생성 시스템 구축
	📥 입력 데이터 형식 (텍스트 데이터 처리 결과)
	json
	{
	    "input_text": "대한민국의 수도는 서울이다. 서울은 한국의 경제와 문화의 중심지이다. [SEP] 서울"
	}
#### 📤 출력 데이터 형식 (주관식 문제 예시)
	json
	
	{
	    "target_text": "대한민국의 수도는 어디인가?"
	}
#### 📚 질문 생성 과정 (Fine-tuning & Training)
##### 🔹 1차: KoBART 모델을 활용한 질문 생성 Fine-tuning
	✅ 모델: KoBART (사전 학습된 한국어 BART 모델)
	✅ 입력: {Context + Answer} 조합
	✅ 출력: 질문(Question) 생성

#### 🔹 2차: 문장 유사도를 활용한 Fine-tuning (Supervised Learning)
	✅ 기존 Fine-tuning된 모델을 추가 학습
	✅ 문장 유사도를 Loss로 활용하여 자연스러운 질문 생성
	✅ 기존 정답과 완전히 일치하지 않더라도, 의미가 유사하면 정답으로 학습 가능

#### 💡 예시 (문장 유사도 활용)
	정답 질문: "대한민국의 수도는 어디인가?"
	모델이 생성한 질문: "한국의 수도는 무엇인가?"
