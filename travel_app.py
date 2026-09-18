import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_FOLDER = Path(__file__).parent
KNOWLEDGE_FOLDER = PROJECT_FOLDER / "knowledge_base"
load_dotenv(PROJECT_FOLDER / ".env")


def _read_mcp_result(result):
    """Convert common MCP response shapes into a Python dictionary."""
    if isinstance(result, list) and result:
        first_item = result[0]
        result = getattr(first_item, "text", first_item)
        if isinstance(result, dict) and "text" in result:
            result = result["text"]
    if isinstance(result, str):
        return json.loads(result)
    return result


def get_weather_tool():
    """Return a function that calls the separate MCP weather server."""
    async def call_mcp_weather(forecast_days=3):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        server_path = PROJECT_FOLDER / "weather_mcp_server.py"
        client = MultiServerMCPClient({
            "weather": {
                "command": "py",
                "args": ["-3.14", str(server_path)],
                "transport": "stdio",
            }
        })
        # Discover the tools from the MCP server instead of hard-coding API calls.
        tools = await client.get_tools()
        find_city = next(tool for tool in tools if tool.name == "find_city")
        get_weather = next(tool for tool in tools if tool.name == "get_weather")
        city = _read_mcp_result(await find_city.ainvoke({"city": "Singapore"}))
        weather = await get_weather.ainvoke({
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "forecast_days": forecast_days,
        })
        return _read_mcp_result(weather)

    def weather_tool(forecast_days=3):
        try:
            return {"source": "MCP Weather Server", **asyncio.run(call_mcp_weather(forecast_days))}
        except Exception as error:
            return {"source": "MCP Weather Server", "error": f"Unavailable: {error}"}

    return weather_tool


def get_currency_tool():
    """Return a function that calls the separate MCP currency server."""
    async def call_mcp_currency(amount, from_currency, to_currency):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        server_path = PROJECT_FOLDER / "currency_mcp_server.py"
        client = MultiServerMCPClient({
            "currency": {
                "command": "py",
                "args": ["-3.14", str(server_path)],
                "transport": "stdio",
            }
        })
        # The MCP server owns the live exchange-rate lookup.
        tools = await client.get_tools()
        convert = next(tool for tool in tools if tool.name == "convert_currency")
        return _read_mcp_result(await convert.ainvoke({
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
        }))

    def currency_tool(amount, from_currency="INR", to_currency="SGD"):
        try:
            return {"source": "MCP Currency Server", **asyncio.run(
                call_mcp_currency(float(amount), from_currency.upper(), to_currency.upper())
            )}
        except Exception as error:
            return {
                "source": "MCP Currency Server",
                "from": from_currency.upper(),
                "to": to_currency.upper(),
                "error": f"Unavailable: {error}",
            }

    return currency_tool


def build_knowledge_base(folder=KNOWLEDGE_FOLDER, chunk_size=850, overlap=120):
    """Load Markdown sources and split them into citation-friendly chunks."""
    documents = []
    for file_path in sorted(Path(folder).glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        # Paragraph boundaries keep related travel guidance together in each chunk.
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks = []
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > chunk_size:
                chunks.append(current)
                current = current[-overlap:]
            current = f"{current}\n\n{paragraph}".strip()
        if current:
            chunks.append(current)
        documents.append({
            "title": file_path.stem.replace("_", " ").title(),
            "source": file_path.name,
            "url": next(
                (line.split("URL:", 1)[1].strip() for line in text.splitlines() if line.startswith("URL:")),
                "",
            ),
            "text": text,
            "chunks": chunks,
        })

    if not documents:
        raise FileNotFoundError(f"No knowledge files found in {folder}")
    return documents


class TravelAssistant:
    """Singapore travel assistant with local vector retrieval and MCP tools."""

    def __init__(self, folder=KNOWLEDGE_FOLDER, use_openai=False):
        self.use_openai = use_openai
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a careful Singapore travel assistant.
Use only the supplied knowledge-base for stable destination facts.
Use MCP results only for current weather and exchange-rate facts.
If the excerpts do not contain enough evidence, say so clearly and do not answer from general knowledge.
Never invent missing facts, prices, opening hours, weather, or currency values.
Clearly distinguish sourced facts, current MCP data, and your recommendations.
If a tool result contains an error, say that current data is unavailable instead of guessing.
For every knowledge-base claim, include its matching [Source N] reference and source URL when available.
Preserve relevant preferences from the conversation context.
Structure the answer with these headings when applicable:
1. Knowledge-base facts with [Source N] references
2. Current MCP information (name the weather or currency MCP tool)
3. Recommendation (clearly label generated suggestions)
Use concise headings and bullets, and include a short limitation note when evidence is incomplete."""),
            ("human", """Question: {question}

Conversation context:
{history}

Knowledge-base excerpts:
{knowledge}

MCP results:
{tools}"""),
        ])
        self.documents = build_knowledge_base(folder)
        self.chunks = [
            {
                "title": document["title"],
                "source": document["source"],
                "url": document["url"],
                "text": chunk,
            }
            for document in self.documents
            for chunk in document["chunks"]
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words=[*TfidfVectorizer(stop_words="english").get_stop_words(), "singapore"],
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform([chunk["text"] for chunk in self.chunks])
        self.vector_store = None
        self.retrieval_mode = "tfidf-fallback"
        self._build_embedding_store()
        self.weather_tool = get_weather_tool()
        self.currency_tool = get_currency_tool()

    def _build_embedding_store(self):
        """Build an OpenAI-embedded FAISS index when the configured key is available."""
        if not self.use_openai or not os.getenv("OPENAI_API_KEY"):
            return
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings

            self.vector_store = FAISS.from_texts(
                [chunk["text"] for chunk in self.chunks],
                embedding=OpenAIEmbeddings(model=self.embedding_model),
                metadatas=self.chunks,
            )
            self.retrieval_mode = "openai-faiss"
        except Exception:
            # Keep the app usable; search_knowledge will use the local fallback index.
            self.vector_store = None

    def _openai_answer(self, question, history, sources, tool_results):
        if not sources or not self.use_openai or not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from langchain_openai import ChatOpenAI

            knowledge = "\n\n".join(
                f"[Source {index}] {source['title']} ({source['source']})\n{source['text']}"
                for index, source in enumerate(sources, start=1)
            ) or "No relevant knowledge-base excerpt was retrieved."
            prompt = self.answer_prompt.invoke({
                "question": question,
                "history": json.dumps(history[-3:], indent=2),
                "knowledge": knowledge,
                "tools": json.dumps(tool_results, indent=2),
            })
            response = ChatOpenAI(
                model=self.openai_model,
                temperature=0,
            ).invoke(prompt)
            return response.content
        except Exception as error:
            return f"OpenAI answer generation was unavailable, so I used the grounded local response. ({error})"

    def search_knowledge(self, question, limit=3):
        """Retrieve relevant chunks from FAISS embeddings or the offline TF-IDF index."""
        if self.vector_store is not None:
            # Production mode uses semantic embeddings stored in FAISS.
            matches = self.vector_store.similarity_search_with_relevance_scores(question, k=limit)
            return [
                document.metadata
                for document, score in matches
                if score >= 0.15
            ]
        # Tests and offline runs use this deterministic local retrieval path.
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [self.chunks[index] for index, score in ranked[:limit] if score >= 0.12]

    @staticmethod
    def _weather_requires_indoor_plan(weather):
        payload = json.dumps(weather).lower()
        if any(word in payload for word in ["rain", "storm", "showers"]):
            return True
        current = weather.get("current", {})
        if float(current.get("precipitation", 0) or 0) > 0:
            return True
        daily = weather.get("daily", {})
        probabilities = daily.get("precipitation_probability_max", [])
        if any(float(probability or 0) >= 50 for probability in probabilities):
            return True
        return any(51 <= int(code) <= 99 for code in daily.get("weather_code", []))

    @staticmethod
    def _forecast_day_is_wet(weather, day_index):
        """Check one forecast day so each itinerary day can use its own alternative."""
        daily = weather.get("daily", {})
        probabilities = daily.get("precipitation_probability_max", [])
        codes = daily.get("weather_code", [])
        if day_index < len(probabilities) and float(probabilities[day_index] or 0) >= 50:
            return True
        return day_index < len(codes) and 51 <= int(codes[day_index]) <= 99

    @staticmethod
    def _parse_currency(question):
        currency_names = {
            "inr": "INR",
            "indian rupee": "INR",
            "indian rupees": "INR",
            "usd": "USD",
            "us dollar": "USD",
            "us dollars": "USD",
            "dollar": "USD",
            "dollars": "USD",
            "eur": "EUR",
            "euro": "EUR",
            "euros": "EUR",
            "gbp": "GBP",
            "british pound": "GBP",
            "british pounds": "GBP",
            "pound": "GBP",
            "pounds": "GBP",
            "sgd": "SGD",
            "singapore dollar": "SGD",
            "singapore dollars": "SGD",
        }
        currency_pattern = "|".join(
            re.escape(name) for name in sorted(currency_names, key=len, reverse=True)
        )
        match = re.search(
            rf"(?:convert|exchange)\s+(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<from>{currency_pattern})\s*(?:to|in)\s*(?P<to>{currency_pattern})",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                rf"(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<from>{currency_pattern})\s*(?:to|in)\s*(?P<to>{currency_pattern})",
                question,
                flags=re.IGNORECASE,
            )
        if not match:
            match = re.search(
                rf"(?P<from>{currency_pattern})\s*(?P<amount>[\d,]+(?:\.\d+)?)\s*(?:to|in)\s*(?P<to>{currency_pattern})",
                question,
                flags=re.IGNORECASE,
            )
        if not match:
            return None
        return (
            float(match.group("amount").replace(",", "")),
            currency_names[match.group("from").lower()],
            currency_names[match.group("to").lower()],
        )

    def answer_question(self, question, chat_history=None):
        """Answer a question using retrieved chunks and selected MCP tools."""
        history = chat_history or []
        contextual_question = f"{history[-1]['question']} {question}" if history else question
        question_lower = contextual_question.lower()
        sources = self.search_knowledge(contextual_question)
        tool_results = {}
        answer_parts = []

        # Route only current-information intents to MCP; destination facts stay in RAG.
        if any(word in question_lower for word in ["weather", "forecast", "rain"]):
            forecast_days = 7 if "next week" in question_lower else 3
            tool_results["weather"] = self.weather_tool(forecast_days)

        currency = self._parse_currency(contextual_question)
        currency_requested = any(word in question_lower for word in ["currency", "convert", "exchange", "budget"])
        if currency:
            amount, from_currency, to_currency = currency
            tool_results["currency"] = self.currency_tool(amount, from_currency, to_currency)
        elif currency_requested:
            answer_parts.append(
                "To use the Currency MCP tool, please provide an amount and both currencies, "
                "for example: `Convert 500 USD to Singapore dollars`."
            )

        # Combine stable itinerary guidance with the live forecast when planning a trip.
        if "itinerary" in question_lower or "three-day" in question_lower or "3-day" in question_lower:
            weather = tool_results.get("weather", {})
            day_one = (
                "National Gallery Singapore and indoor museum time"
                if self._forecast_day_is_wet(weather, 0)
                else "Marina Bay, Merlion Park, and Gardens by the Bay"
            )
            day_two = (
                "Chinatown heritage indoor stops and a hawker-centre lunch"
                if self._forecast_day_is_wet(weather, 1)
                else "Chinatown, Kampong Glam, and Little India for heritage and food"
            )
            day_three = (
                "Indoor family attractions or museums"
                if self._forecast_day_is_wet(weather, 2)
                else "Sentosa and its beaches"
            )
            answer_parts.append(
                "Recommendation (generated from retrieved itinerary and transport guidance):\n"
                f"Day 1: {day_one}.\n"
                f"Day 2: {day_two}.\n"
                f"Day 3: {day_three}.\n"
                "Use the MRT between districts and keep water and an umbrella available."
            )

        if tool_results:
            answer_parts.append("Current information from MCP tools (live data):\n" + json.dumps(tool_results, indent=2))

        failed_tools = [name for name, result in tool_results.items() if result.get("error")]
        if failed_tools:
            answer_parts.append(
                "I could not retrieve current information from "
                + ", ".join(failed_tools)
                + ". I have not substituted an invented value."
            )

        if sources:
            answer_parts.append("Knowledge-base facts:\n" + "\n\n".join(
                f"{source['title']}:\n{source['text']}" for source in sources
            ))
        else:
            answer_parts.append("I could not find a matching travel document. Try asking about attractions, food, transport, weather, or an itinerary.")

        local_answer = "\n\n".join(answer_parts)
        model_answer = self._openai_answer(question, history, sources, tool_results)
        answer = model_answer if model_answer and not model_answer.startswith("OpenAI answer generation") else local_answer
        if model_answer and model_answer.startswith("OpenAI answer generation"):
            answer += "\n\n" + model_answer

        return {
            "answer": answer,
            "tool_results": tool_results,
            "sources": [
                {"title": source["title"], "file": source["source"], "url": source["url"]}
                for source in sources
            ],
            "used_tool_names": list(tool_results),
            "destination": "Singapore",
        }


if __name__ == "__main__":
    assistant = TravelAssistant(use_openai=True)
    print(assistant.answer_question("Create a three-day itinerary and check the weather.")["answer"])
