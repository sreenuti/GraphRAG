from dotenv import load_dotenv
import os
from typing import Tuple, List, Dict, Any

import streamlit as st
from langchain_neo4j import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

load_dotenv()

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")


kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)  # database=NEO4J_DATABASE,


# Hybrid Retrieval for RAG - create vector index
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)
entity_chain = prompt | chat.with_structured_output(Entities)

# Retriever
kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        # print(f" Getting Entity: {entity}")
        response = kg.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        # print(response)
        result += "\n".join([el["output"] for el in response])
    return result


# print(structured_retriever("Who is commudus?"))


# Final retrieval step
def retriever(question: str) -> str:
    # print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    # print(f"\nFinal Data::: ==>{final_data}")
    return final_data


# Define the RAG chain
# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | chat
    | StrOutputParser()
)

# # TEST it all out!
# res_simple = chain.invoke(
#     {
#         "question": "How did the Roman empire fall?",
#     }
# )

# print(f"\n Results === {res_simple}\n\n")

# res_hist = chain.invoke(
#     {
#         "question": "When did he become the first emperor?",
#         "chat_history": [
#             ("Who was the first emperor?", "Augustus was the first emperor.")
#         ],
#     }
# )
#
# print(f"\n === {res_hist}\n\n")


def build_chat_history_from_messages(
    messages: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """
    Convert Streamlit-style chat messages into (human, ai) pairs
    expected by the chain.
    """
    history: List[Tuple[str, str]] = []
    last_human: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            last_human = content
        elif role == "assistant" and last_human is not None:
            history.append((last_human, content))
            last_human = None
    return history


def answer_question(question: str, chat_messages: List[Dict[str, Any]] | None = None) -> str:
    """
    High-level helper to answer a user question using the RAG chain.
    """
    payload: Dict[str, Any] = {"question": question}
    if chat_messages:
        payload["chat_history"] = build_chat_history_from_messages(chat_messages)
    return chain.invoke(payload)


def get_suggested_followups(question: str, answer: str) -> List[str]:
    """
    Generate a small set of suggested follow-up questions based on
    the latest user question and answer.
    """
    try:
        suggestions_prompt = ChatPromptTemplate.from_template(
            "You are helping a user explore the History of India.\n"
            "Given the last question and answer, suggest 3 concise follow-up questions "
            "the user might reasonably ask next.\n\n"
            "Last question: {question}\n"
            "Answer: {answer}\n\n"
            "Return ONLY the questions, one per line, no numbering or bullets."
        )
        suggestions_chain = suggestions_prompt | chat | StrOutputParser()
        raw = suggestions_chain.invoke({"question": question, "answer": answer})
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # Deduplicate and limit to 3
        seen = set()
        unique: List[str] = []
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                unique.append(ln)
            if len(unique) >= 3:
                break
        return unique
    except Exception:
        # In case suggestion generation fails, just return an empty list.
        return []


def main() -> None:
    """
    Streamlit chat UI for querying the GraphRAG pipeline.
    Run with:
        streamlit run "roman_emp_graph_rag chat_ui.py"
    """
    st.set_page_config(page_title="History of India GraphRAG Chat", page_icon="🏛️")
    st.title("History of India GraphRAG Chatbot")
    st.write(
        "Ask questions about the History of India. "
        "Answers are generated using both Neo4j graph data and RAG over documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "suggested_followups" not in st.session_state:
        st.session_state.suggested_followups = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Render follow-up suggestion buttons under the last assistant message, if any
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.session_state.suggested_followups:
            st.markdown("**You can also ask:**")
            cols = st.columns(len(st.session_state.suggested_followups))
            for idx, suggestion in enumerate(st.session_state.suggested_followups):
                if cols[idx].button(suggestion):
                    # When a suggestion is clicked, immediately use it as the next user input
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    with st.chat_message("user"):
                        st.markdown(suggestion)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                answer = answer_question(
                                    suggestion, st.session_state.messages[:-1]
                                )
                            except Exception as e:
                                answer = (
                                    "Sorry, something went wrong while querying the graph/RAG pipeline: "
                                    f"{e}"
                                )
                            st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    # Update follow-ups for the new answer and stop rendering old buttons
                    st.session_state.suggested_followups = get_suggested_followups(
                        suggestion, answer
                    )
                    st.rerun()

    user_input = st.chat_input("Type your question about the History of India...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(user_input, st.session_state.messages[:-1])
                except Exception as e:
                    answer = (
                        "Sorry, something went wrong while querying the graph/RAG pipeline: "
                        f"{e}"
                    )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.suggested_followups = get_suggested_followups(user_input, answer)
        # Rerun so that newly generated follow-ups render immediately
        st.rerun()


if __name__ == "__main__":
    main()
