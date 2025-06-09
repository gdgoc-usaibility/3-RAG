import os
from openai import OpenAI
from build_vector_db_hw3 import get_embedding
from chromadb import Client
import chromadb 
from chromadb.config import Settings 
from dotenv import load_dotenv


load_dotenv()
dbclient = chromadb.PersistentClient(path="./chroma_db")
collection = dbclient.get_or_create_collection("rag_collection")

# query를 임베딩해 chroma에서 가장 유사도가 높은 top-k개의 문서 가져오는 함수 
def retrieve(query, top_k=3):
    query_embedding = get_embedding(query)  # query 임베딩 생성
    results = collection.query(
        query_embeddings=[query_embedding],  # 쿼리 임베딩
        n_results=top_k  # top_k 개의 결과 반환
    )
    return results
    # do it


# 1) query에 대해 벡터 DB에서 top_k개 문서 retrieval
# 2) 그 문서들을 context로 묶어 GPT에 prompt
#3) 최종 답변 반환 하는 함수 

def generate_answer_with_context(query, top_k=5):
    results = retrieve(query, top_k) # retrieve 함수로 결과 얻기
    # top_k에 대한 documents와 metadatas 리스트로 추출
    found_docs = results["documents"][0] 
    found_metadatas = results["metadatas"][0]

    # context 구성 (검색된 문서들을 하나의 문맥으로 결합)
    context_texts = []
    # zip을 이용해 두 리스트의 같은 인덱스에 있는 값들을 한 쌍으로 묶음
    for doc_text, meta in zip(found_docs, found_metadatas): 
        context_texts.append(f"\n{doc_text}")
    # context_texts 리스트에 있는 모든 문자열이 \n\n으로 이어 붙여짐
    context_str = "\n\n".join(context_texts)

    # 프롬프트 작성
    system_prompt = """
    당신은 영화 '미션 임파서블: 데드 레코닝 파트2'에 대한 문서 검색을 바탕으로 사용자 질문에 답변하는
    지능형 어시스턴트입니다. 다음 원칙을 엄격히 지키세요:

    1. 모든 질문은 해당 영화에 관한 질문으로 간주하고 답변합니다. "미션 임파서블: 데드 레코닝 파트2 에서 ~ (질문 내용)" 형태로 질문을 이해합니다.
    -  예시1: "매출이 얼마야?" -> "미션 임파서블: 데드 레코닝 파트2 에서 매출이 얼마야?"
    - 예시2: "주인공이 죽어?" -> "미션 임파서블: 데드 레코닝 파트2 에서 주인공이 죽어?"
    4. 지나치게 장황하지 않게, 간결하고 알기 쉽게 설명하세요.
    5. 사용자가 질문을 한국어로 한다면, 한국어로 답변하고, 
    다른 언어로 질문한다면 해당 언어로 답변하도록 노력하세요.
    6. 문서 출처나 연도가 중요하다면, 가능한 정확하게 전달하세요.
    7. 영화 내에 특정 사건, 또는 특정인물의 행동, 생사 여부와 와 관련된 내용을 묻는 다면, 반드시 스포일러를 듣는 것에 대한 동의를 한 번 구할 것. 질문자가 동의한다면 질문에 답변할 것.
    - 영화 내에 특정 사건, 또는 특정인물의 행동, 생사 여부와 와 관련된 내용을 묻는 다면, 문서에 없는 내용은 일어나지 않은 것으로 간주하고 답변해도 됩니다.
    - 예를 들어, 문서에 특정 인물의 죽음이 언급 되지 않았다면 "해당 인물은 죽지 않았습니다.", 사건이 전혀 언급되지 않았다면 "해당 사건은 일어나지 않았습니다."라고 답변하세요.
    - 이외에 문서에 언급되지 않은 내용이라면, 함부로 추측하거나 만들어내지 마세요. 
    8. 스포일러에 관한 답변 허락을 구하고 난 이후에는 지속해서 스포일러를 포함한 답변이 가능합니다.

    친절하고 이해하기 쉬운 어투를 구사합니다. 
    """

    user_prompt =f"""아래는 검색된 문서들의 내용입니다:
    {context_str}
    질문: {query}"""

    # ChatGPT 호출 
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role":"system", "content": system_prompt},
        {"role":"user", "content": user_prompt}]
    )

    answer = response.choices[0].message.content
    return answer 


if __name__ == "__main__":
    while True:
        user_query = input("질문을 입력하세요(종료: quit): ")
        if user_query.lower() == "quit":
            break 
        answer = generate_answer_with_context(user_query, top_k=3)
        print("===답변===")
        print(answer)
        print("==========\n")