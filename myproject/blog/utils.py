import os
import re
import faiss
import numpy as np
from langchain_ollama import OllamaLLM  # 최신 LangChain Ollama 모듈
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph
from autogen import UserProxyAgent, AssistantAgent  # AutoGen을 활용한 에이전트 기반 실행

# FAISS 벡터 DB 초기화 (문장 벡터화 후 저장 및 검색을 위한 DB)
vector_dim = 512  # 벡터 차원 (임베딩 크기)
index = faiss.IndexFlatL2(vector_dim)  # L2 거리 기반의 벡터 검색 인덱스 생성
vector_store = {}  # 질문과 답변을 저장하는 딕셔너리 형태의 벡터 저장소

# LangChain 메모리 객체 생성 (대화 기록 저장)
memory = ConversationBufferMemory()

# Ollama 모델 설정 (사용할 로컬 모델 지정)
OLLAMA_MODEL_NAME = "hf.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M"

# 데이터 캐싱 변수 (data.txt 파일 내용을 저장하여 불필요한 읽기를 줄임)
DATA_CACHE = None


class ChatState:
    """LangGraph에서 사용할 상태 클래스"""
    def __init__(self, question="", response=""):
        self.question = question  # 사용자가 입력한 질문
        self.response = response  # LLM이 생성한 응답


def remove_ansi_escape(text):
    """ANSI escape 코드 제거 (터미널 출력 시 스타일 관련 코드 삭제)"""
    print("[LangChain] ANSI escape 코드 제거 중...")
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mGKHJ]|\x1b\[[0-9;]*[@-~]')
    cleaned_text = ansi_escape.sub('', text).strip()
    print("[LangChain] ANSI escape 코드 제거 완료.")
    return cleaned_text


def load_data():
    """data.txt 파일을 읽어와 캐싱 (이전 요청에서 불필요한 재읽기 방지)"""
    global DATA_CACHE
    if DATA_CACHE is None:  # 캐시가 없을 경우 파일을 로드
        print("[LangChain] 데이터 파일 로드 시작...")
        try:
            file_path = r"C:\Workspace\langchain\myproject\data.txt"

            if not os.path.exists(file_path):  # 파일이 존재하는지 확인
                print(f"[ERROR] data.txt 파일을 찾을 수 없습니다: {file_path}")
                DATA_CACHE = ""
                return DATA_CACHE

            with open(file_path, 'r', encoding='utf-8') as f:
                DATA_CACHE = f.read().strip()

            print("[LangChain] data.txt 로드 완료")
        except Exception as e:
            print(f"[ERROR] data.txt 읽기 오류: {e}")
            DATA_CACHE = ""

    return DATA_CACHE


def ollama_generate_response(prompt: str):
    """LangChain 기반 Ollama 모델 실행 (질문을 모델에 전달하고 응답을 받음)"""
    print("[LangChain] Ollama LLM 실행 시작...")
    print(f"[LangChain] 실행 프롬프트:\n{prompt[:300]}...")  # 실행할 프롬프트 일부 출력

    try:
        llm = OllamaLLM(model=OLLAMA_MODEL_NAME, temperature=0.1, top_p=0.2)  # Ollama 모델 생성
        response = llm.invoke(prompt)  # 프롬프트 실행 후 응답 받기
        print("[LangChain] Ollama 응답 수신 완료.")
        return remove_ansi_escape(response)  # ANSI escape 코드 제거 후 반환

    except Exception as e:
        print(f"[ERROR] Ollama 실행 오류: {e}")
        return f"오류 발생: {e}"


def generate_response(state: ChatState):
    """LangGraph에서 질문을 처리하고 응답을 생성하는 함수"""
    print("\n[LangGraph] 실행 시작")
    print(f"[LangGraph] 입력 질문: {state.question}")

    # 벡터 DB에서 유사한 질문 검색 (AutoGen과 연계 가능)
    retrieved_answer = search_vector_db(state.question)
    if retrieved_answer:
        print("[LangGraph] 벡터 DB 검색 결과 있음, 기존 응답 반환.")
        return ChatState(question=state.question, response=retrieved_answer)

    # 데이터 파일 로드 (data.txt 내용 포함)
    data_from_file = load_data()
    if not data_from_file:
        print("[ERROR] data.txt 읽기 실패")
        return ChatState(question=state.question, response="Error: data.txt의 내용을 읽지 못했습니다.")

    # 프롬프트 구성 (LLM이 답변할 맥락을 제공)
    prompt = f"""
    시스템 메시지: 당신은 한국어로 답변하는 유용한 도우미입니다.

    [data.txt 내용]:
    {data_from_file}

    사용자 질문: {state.question}
    """

    # Ollama를 실행하여 답변 생성
    response = ollama_generate_response(prompt)

    if not response:
        return ChatState(question=state.question, response="응답이 없습니다.")

    # 벡터 DB에 질문-답변 저장 (AutoGen 연계 가능)
    store_vector_db(state.question, response)

    # LangChain 메모리에 대화 기록 저장
    memory.save_context({"input": state.question}, {"output": response})

    print("[LangGraph] 실행 종료")
    return ChatState(question=state.question, response=response)


# AutoGen 설정 (Docker 비활성화)
print("[AutoGen] 에이전트 초기화...")
code_execution_config = {"use_docker": False}

user_agent = UserProxyAgent(name="User", code_execution_config=code_execution_config)
ollama_agent = AssistantAgent(name="OllamaExecutor", llm_config=False)
print("[AutoGen] 에이전트 초기화 완료.")


def store_vector_db(question: str, answer: str):
    """질문을 벡터화하여 FAISS DB에 저장"""
    global index, vector_store

    existing_answer = search_vector_db(question)
    if existing_answer:
        return

    question_vector = np.random.random((1, vector_dim)).astype('float32')  # 질문을 벡터화

    index.add(question_vector)  # FAISS 벡터 DB에 추가
    vector_store[len(vector_store)] = (question, answer)  # 딕셔너리에 저장


def search_vector_db(query: str, k=1):
    """벡터 DB에서 가장 유사한 질문 검색"""
    global index, vector_store

    if index.ntotal == 0:
        return None

    query_vector = np.random.random((1, vector_dim)).astype('float32')

    D, I = index.search(query_vector, k)
    closest_index = I[0][0]

    if closest_index in vector_store and vector_store[closest_index][0] == query:
        return vector_store[closest_index][1]

    return None


def run_model(question: str):
    """LangGraph 실행"""
    print(f"\n[LangGraph] 모델 실행 요청: {question}")

    try:
        state = app.invoke(ChatState(question=question))
        print(f"[LangGraph] 모델 응답 완료: {state.response[:100]}...")
        return state.response
    except Exception as e:
        print(f"[ERROR] LangGraph 실행 중 오류 발생: {e}")
        return "LangGraph 실행 오류가 발생했습니다."


# LangGraph 설정 (대화 흐름 구성)
print("[LangGraph] LangGraph 초기화 시작...")
graph = StateGraph(ChatState)

graph.add_node("generate_response", generate_response)

graph.set_entry_point("generate_response")
graph.set_finish_point("generate_response")

app = graph.compile()
print("[LangGraph] LangGraph 초기화 완료.")


# 실행 테스트 (테스트용 질문)
if __name__ == "__main__":
    print("[System] 테스트 실행 중...")

    test_question1 = "1 + 1 = ?"
    print("\n=== 첫 번째 질문 실행 ===")
    answer1 = run_model(test_question1)
    print("\n=== 모델 답변 ===")
    print(answer1)

    test_question2 = "3 + 3 = ?"
    print("\n=== 두 번째 질문 실행 ===")
    answer2 = run_model(test_question2)
    print("\n=== 모델 답변 ===")
    print(answer2)

    print("[System] 테스트 종료.")
