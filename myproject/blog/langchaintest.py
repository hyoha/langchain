from langchain.llms import Ollama

# ✅ CLI에서 실행되는 정확한 모델명 사용
llm = Ollama(model="hf.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M")

response = llm.invoke("안녕하세요?")

print(response)
