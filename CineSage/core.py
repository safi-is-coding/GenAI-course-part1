from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional

from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ---------------- Pydantic Schema ---------------- #

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# ---------------- Parser ---------------- #

parser = PydanticOutputParser(
    pydantic_object=Movie
)

# ---------------- Model ---------------- #

model = init_chat_model(
    "groq:llama-3.1-8b-instant"
)

# ---------------- Prompt ---------------- #

extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert movie information extractor.

Extract movie information from the provided paragraph.

{format_instructions}
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])

# ---------------- Input ---------------- #

paragraph = input("Give your paragraph: ")

# ---------------- Prompt Formatting ---------------- #

prompt = extraction_prompt.invoke({
    "paragraph": paragraph,
    "format_instructions": parser.get_format_instructions()
})

# ---------------- LLM Call ---------------- #

response = model.invoke(prompt)

print(response.content)


# Interstellar is a visually stunning science fiction epic directed by Christopher Nolan. Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Michael Caine. The story revolves around a group of astronauts who travel through a wormhole near Saturn in search of a new home for humanity as Earth faces environmental collapse. The movie was widely appreciated for its emotional depth, scientific accuracy, and Hans Zimmer's powerful soundtrack. It holds a rating of 8.6 on IMDb and is often considered one of the greatest sci-fi films ever made. can you extract the summary and information of the movie