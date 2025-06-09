# https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/
import os 
from openai import OpenAI 
import chromadb 
from chromadb.config import Settings 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests




def crawl_namuwiki(urls):

    full_text = ''

    for url in urls:    
        response = requests.get(url)  # 요청 보내기
        html = response.text  # 응답 받은 HTML 문서
        soup = BeautifulSoup(html, 'html.parser')  # BeautifulSoup으로 파싱

        content_elements = soup.select("div.wiki-paragraph")
        # content_elements = soup.find_all(['p', 'div'])

        for element in content_elements:
            text_c = element.get_text(strip=True).replace('\r', '').replace('\n', '')
            # full_text += element.get_text(separator='\n', strip=True) + '\n'
            full_text += text_c + '\n'


    print("크롤링 완료")
    print(len(full_text), "문자열 길이")
    print(full_text[:3000])  # 크롤링된 텍스트의 처음 500자 출력
    return full_text





# 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화
# do it
load_dotenv()  # .env 파일에서 환경 변수 로드
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 OpenAI API 키 가져오기
client = OpenAI(api_key=api_key)  # Open

# 매 실행 시 DB 폴더를 삭제 후 새로 생성
def init_db(db_path="./chroma_db"):
    dbclient = chromadb.PersistentClient(path = db_path)
    collection = dbclient.create_collection(name="rag_collection", get_or_create=True)
    return dbclient, collection


# OpenAI Embeddings 생성 함수 
def get_embedding(text, model="text-embedding-3-large"):
    # do it
    response = client.embeddings.create(input = [text], model = model)
    embedding = response.data[0].embedding
    return embedding  # 임베딩 벡터 반환

# 문서 청크 단위로 나누기
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)
    
    return chunks


# 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입
if __name__ == "__main__":
    
    urls = [
        "https://namu.wiki/w/%EB%AF%B8%EC%85%98%20%EC%9E%84%ED%8C%8C%EC%84%9C%EB%B8%94:%20%ED%8C%8C%EC%9D%B4%EB%84%90%20%EB%A0%88%EC%BD%94%EB%8B%9D?from=%EB%AF%B8%EC%85%98%20%EC%9E%84%ED%8C%8C%EC%84%9C%EB%B8%94%20%EB%8D%B0%EB%93%9C%20%EB%A0%88%EC%BD%94%EB%8B%9D%20%ED%8C%8C%ED%8A%B8%202#s-7",
        "https://namu.wiki/w/%EB%AF%B8%EC%85%98%20%EC%9E%84%ED%8C%8C%EC%84%9C%EB%B8%94:%20%ED%8C%8C%EC%9D%B4%EB%84%90%20%EB%A0%88%EC%BD%94%EB%8B%9D/%EC%A4%84%EA%B1%B0%EB%A6%AC"
    ]



    # 함수를 호출하여 정보를 크롤링하고 결과를 출력합니다.
    crawled_data = crawl_namuwiki(urls)
    chunks = chunk_text(crawled_data, chunk_size=400, chunk_overlap=50) # chunking
    print("청크 단위로 나누기 완료")
    print(f"총 {len(chunks)}개의 청크로 나누어짐")
    
    
    # db 초기화
    dbclient, collection = init_db("./chroma_db")
    doc_id = 0

    for idx, chunk in enumerate(chunks): # 각 청크와 해당 청크의 인덱스 가져옴
        doc_id += 1 # 인덱스 하나씩 증가 시키면서
        embedding = get_embedding(chunk) # 각 청크 임베딩 벡터 생성
        # vectorDB에 다음 정보 추가
        collection.add(
            documents=[chunk], # 실제 청크 text
            embeddings=[embedding], # 생성된 임베딩 벡터
            metadatas=[{"chunk_index": idx}], # 파일 이름과 청크 인덱스를 포함하는 메타데이터
            ids=[str(doc_id)] # 각 청크의 Unique한 id 저장
            # 이 고유 id를 통해 db에서 업데이트, 삭제등의 작업 가능 
        )
        if doc_id % 100 == 0:
            print(f"{doc_id}번째 청크 벡터DB에 저장 완료")
        
        
    print("모든 문서 벡터DB에 저장 완료")