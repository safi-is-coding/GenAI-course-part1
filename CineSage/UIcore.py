from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from typing import List, Optional

from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬"
)

# ---------------- PYDANTIC SCHEMA ---------------- #

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# ---------------- PARSER ---------------- #

parser = PydanticOutputParser(
    pydantic_object=Movie
)

# ---------------- MODEL ---------------- #

model = init_chat_model(
    "groq:llama-3.1-8b-instant"
)

# ---------------- PROMPT ---------------- #

extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert movie information extractor.

Extract movie information from the provided paragraph.

Return ONLY valid JSON.

{format_instructions}
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])

# ---------------- UI ---------------- #

st.title("🎬 Movie Information Extractor")

paragraph = st.text_area(
    "Enter Movie Description",
    height=250,
    placeholder="Paste a movie description here..."
)

# ---------------- PROCESSING ---------------- #

if st.button("Extract Information"):

    if not paragraph.strip():
        st.warning("Please enter a movie description.")
        st.stop()

    try:

        prompt = extraction_prompt.invoke({
            "paragraph": paragraph,
            "format_instructions": parser.get_format_instructions()
        })

        response = model.invoke(prompt)

        movie = parser.parse(response.content)

        st.subheader("Extracted Information")
        st.json(movie.model_dump())

    except Exception as e:

        st.error("Failed to parse the model response.")

        st.subheader("Raw Response")
        st.write(response.content)

        st.exception(e)