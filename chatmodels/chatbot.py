from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = init_chat_model("groq:llama-3.1-8b-instant")

# print(model)

messages = [
    SystemMessage(content="You are a funny AI agent")
]

print("-----------Welcome type 0 to exit the app------------")
while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    
    if prompt == "0":
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot: ", response.content)
    
print(messages)