### 5회차 
## RAG 실습과 과제 제출을 위한 Repository입니다 

### 예시 

```bash
git clone git@github.com:HateSlop/~~~  # 클론
cd 2-RAG # 프로젝트 루트로 이동
git checkout -b Hyeonseong # 브랜치 생성 (본인의 브랜치, 폴더 등 생성)
mkdir Hyeonseong # 개인 폴더 만들기
cd Hyeonseong # 개인 폴더로 이동
# 작업을 진행해주세요
git add . # 작업 후 add
git commit -m "[feat] ~~" # 커밋
git push origin Hyeonseong # 오리진에 푸시
```

### 폴더구조
```bash
├── .env
├── .gitignore
├── source_data
    └── docs1.txt
    └── docs2.txt
├── build_vector_db.py
├── rag_chatbot.py
├── README.md
├── requirements.txt
```

### 커밋 컨벤션

feat: 새로운 기능 추가  
fix: 버그 수정  
docs: 문서 수정  
style: 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우  
refactor: 코드 리팩토링  
test: 테스트 코드, 리팩토링 테스트 코드 추가  
chore: 빌드 업무 수정, 패키지 매니저 수정, production code와 무관한 부분들 (.gitignore, build.gradle 같은)  
comment: 주석 추가 및 변경  
remove: 파일, 폴더 삭제  
rename: 파일, 폴더명 수정