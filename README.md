# gen_question_model
비타민 겨울 프로젝트
주제: 학습자료를 바탕으로 문제생성 모델
프로세스
  1. 이미지 처리: PDF형식으로 들어온 학습자료 데이터를 OCR해서 전문 생성
  2. 텍스트 처리: 단순 OCR된 내용들을 요약, 중요한 부분, 키워드 출력 
  3. 생성: 문제및 답변 생성
역할: 생성PART

KOBART모델 사용해서 [context ; Answer]을 입력받으면 Question을 생성하는 모델 제작
1. 한국어 사전학습 모델을 사용하자 - KOBART
2. context에서 n개의 키워드를 뽑아내자 
3. n개의 키워드를 정답으로하는 문제를 출제하자

KOBART + kor_quad 
