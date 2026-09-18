# AI Travel Planning Assistant

An AI travel assistant for Singapore that combines destination knowledge from a local RAG knowledge base with current information retrieved through MCP tools.

The application uses LangChain, OpenAI models, OpenAI embeddings, FAISS, MCP, and Streamlit.

## Deliverables

- Source code in this Git repository
- Working Streamlit application
- Four knowledge-base documents with public source metadata
- Architecture, RAG, MCP, prompt, context, and setup documentation
- Sample questions and expected application behavior
- Offline demonstration for RAG, MCP, combined planning, and conversation context

## Technology stack

| Requirement | Implementation |
| --- | --- |
| Application orchestration and prompts | LangChain `ChatPromptTemplate` and `ChatOpenAI` |
| LLM | OpenAI model configured by `OPENAI_MODEL`, default `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | LangChain FAISS |
| Offline retrieval fallback | TF-IDF and cosine similarity |
| MCP client | `MultiServerMCPClient` from `langchain-mcp-adapters` |
| Current information tools | Weather MCP and Currency MCP |
| User interface | Streamlit |

## Architecture

```text
User question
	|
	v
Streamlit interface and conversation state
	|
	v
Intent routing --------------------------+
	|                                   |
	| destination question              | current information
	v                                   v
RAG retrieval                       MCP tool selection
	|                              Weather / Currency
	v                                   |
Knowledge-base chunks                    |
	|                                   |
	+---------------+-------------------+
				 v
		Grounded OpenAI response
		facts + MCP data + recommendation
```

### RAG workflow

1. Markdown files are loaded from `knowledge_base/`.
2. Each document is split into overlapping paragraph chunks.
3. With an OpenAI API key, chunks are embedded using `text-embedding-3-small` and stored in a FAISS vector store.
4. The most relevant chunks are retrieved for each question.
5. Retrieved chunks are passed to the OpenAI answer prompt with source numbers and metadata.
6. The final response cites the source title and URL in the interface.
7. If retrieval confidence is insufficient, the assistant states that the knowledge base does not contain enough information.

For offline tests or missing API configuration, the application uses an explicit TF-IDF fallback so the workflow remains runnable without pretending that it is using OpenAI embeddings.

### MCP workflow

The application connects to two separate stdio MCP servers through `MultiServerMCPClient`:

- `weather_mcp_server.py` uses Open-Meteo geocoding and forecast APIs. It returns current conditions and a three- or seven-day forecast.
- `currency_mcp_server.py` uses the Frankfurter exchange-rate service. It converts arbitrary ISO currency pairs.

The router selects tools by intent:

- Weather, forecast, rain, or indoor/outdoor questions call Weather MCP.
- Convert, exchange, currency, or budget questions call Currency MCP.
- Destination-only questions use RAG and do not call MCP.

Tool results are returned in the final response and are visible in the Streamlit `MCP results` section. Tool failures are reported explicitly; the application does not fabricate current values.

## Knowledge-base sources

The project contains four original compact summaries with source title and URL metadata:

1. [Visit Singapore Essential Travel Information](https://www.visitsingapore.com/travel-tips/essential-travel-information/)
2. [Visit Singapore Sample Itineraries](https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/)
3. [Visit Singapore Top Things To Do](https://www.visitsingapore.com/things-to-do/top-things-to-do/)
4. [Wikivoyage Singapore Travel Guide](https://en.wikivoyage.org/wiki/Singapore)

Coverage includes attractions, neighbourhoods, transport, culture, practical advice, food, family activities, indoor and outdoor activities, and sample itineraries. The checked-in files are summaries rather than wholesale copies. Review each source's reuse terms before redistribution; current prices, opening hours, events, weather, and exchange rates are retrieved live where supported.

## Prompt and context strategy

The OpenAI system prompt instructs the model to:

- Use only retrieved knowledge-base excerpts for stable destination facts.
- Use MCP results only for current weather and exchange-rate facts.
- Avoid unsupported claims; when retrieval is empty or insufficient, state that the knowledge base does not contain enough information.
- Distinguish `Knowledge-base facts`, `Current MCP information`, and `Recommendation`.
- Cite every retrieved knowledge claim using its `[Source N]` reference and source URL when available.
- Mention the MCP tool used for live information.
- Preserve relevant preferences from previous conversation turns.
- Use concise headings and bullets, with a limitation note when evidence is incomplete.

Streamlit stores the question and response in session state. The previous questions are supplied as context to follow-up requests, allowing preferences such as cultural activities or indoor alternatives to persist.

## Setup

Use the project virtual environment from the assignment parent directory:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a local `.env` file. It is excluded from Git by `.gitignore`:

```dotenv
OPENAI_API_KEY=your-replacement-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Never commit or share API keys. Rotate any key that has been exposed.

## Run the application

```powershell
..\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens a Streamlit page with a question field, response spinner, answer, citations, selected tools, MCP payloads, and conversation context.



## Tests

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers source metadata, semantic retrieval, citations, currency parsing, weather routing, currency routing, combined planning, next-week forecasts, multi-turn context, destination-only intent isolation, unsupported knowledge, and MCP failure handling.

## Project files

- `app.py` - Streamlit application and conversation state
- `travel_app.py` - RAG indexing, retrieval, intent routing, MCP client calls, and OpenAI answer generation
- `weather_mcp_server.py` - Weather MCP server
- `currency_mcp_server.py` - Currency MCP server
- `knowledge_base/` - cited Singapore travel summaries
- `tests/` - automated tests
- `requirements.txt` - Python dependencies
- `.gitignore` - excludes `.env`, virtual environments, caches, and bytecode
