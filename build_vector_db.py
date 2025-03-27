# https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/
import os 
from openai import OpenAI 
import chromadb 
from chromadb.config import Settings 
from dotenv import load_dotenv

# 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화
# do it

# 매 실행 시 DB 폴더를 삭제 후 새로 생성
def init_db(db_path="./chroma_db"):
    # do it

# 텍스트 로딩 함수
def load_text_files(folder_path):
    # do it

# OpenAI Embeddings 생성 함수 
def get_embedding(text, model="text-embedding-3-large"):
    # do it
    

# 문서 청크 단위로 나누기
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    # do it 


# 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입
if __name__ == "__main__":
    # db 초기화, 경로 지정
    # load_text_files 함수로 처리할 문서 데이터 불러오기기
    # do it

    # 전처리 과정 
    # do it
    
    print("모든 문서 벡터DB에 저장 완료")