import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars


load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)


class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


_entity_prompt = ChatPromptTemplate.from_messages(
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
entity_chain = _entity_prompt | chat.with_structured_output(Entities)

kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


def _generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def _structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
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
            {"query": _generate_full_text_query(entity)},
        )
        result += "\n".join([el["output"] for el in response])
    return result


def _retriever(question: str) -> str:
    structured_data = _structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)

_answer_template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
_answer_prompt = ChatPromptTemplate.from_template(_answer_template)

_chain = (
    RunnableParallel(
        {
            "context": _search_query | _retriever,
            "question": RunnablePassthrough(),
        }
    )
    | _answer_prompt
    | chat
    | StrOutputParser()
)


def _build_chat_history(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
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


def _get_suggested_followups(question: str, answer: str) -> List[str]:
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
        return []


def answer(question: str, messages: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"question": question}
    if messages:
        payload["chat_history"] = _build_chat_history(messages)
    response_text = _chain.invoke(payload)
    followups = _get_suggested_followups(question, response_text)
    return {"answer": response_text, "followups": followups}


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            data = json.loads(body.decode("utf-8") or "{}")
            question = data.get("question", "")
            messages = data.get("messages", [])

            if not question:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Missing 'question' field"}).encode("utf-8")
                )
                return

            result = answer(question, messages)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

