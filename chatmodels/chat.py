from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("groq:qwen/qwen3-32b")
# print(model)

response = model.invoke("What is cricket?")

print(response.content)