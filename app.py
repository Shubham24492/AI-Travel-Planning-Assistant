import streamlit as st

from travel_app import TravelAssistant

# set page title, header
st.set_page_config(page_title="AI TRAVEL PLANNING ASSISTANT")
st.title('AI Travel Planning Assistant')
st.header("Singapore Travel Planning Assistant")
st.caption("A travel assistant that combines a document-based knowledge base with current information retrieved through MCP tools like weather and currency conversion.")

# create travel assistant instance.
if "assistant" not in st.session_state:
    st.session_state.assistant = TravelAssistant(use_openai=True)

# capture user input
question = st.text_input("Please enter your question here", placeholder="For example: Plan 3 days next week and adjust for rain")

# handle Get answer button click
if st.button("Get answer", type="primary"):
    # validate user input
    if question.strip() == "":
        st.warning("Please enter your question.")
    else:
        # show loading spinner till we get response from mcp calls
        with st.spinner("Loading..."):
            result = st.session_state.assistant.answer_question(
                question, chat_history=st.session_state.get("history", [])
            )

        # Add the questions and result to history for context awareness
        # so follow-up questions can reuse the user's preferences.
        st.session_state.setdefault("history", []).append({"question": question, **result})
        st.subheader("Answer")
        st.write(result["answer"])



        # Display citations separately from the generated answer for easy review.
        st.subheader("Sources")
        for source in result["sources"]:
            title = source.get("title", "Travel source")
            file_name = source.get("file", source.get("source", "local knowledge base"))
            url = source.get("url")
            if url:
                st.markdown(f"- [{title}]({url}) ({file_name})")
            else:
                st.write(f"- {title} ({file_name})")

        if result["used_tool_names"]:
            st.subheader("Tools used")
            st.write(", ".join(result["used_tool_names"]))
            with st.expander("MCP results"):
                st.json(result["tool_results"])

if st.session_state.get("history"):
    with st.expander("Conversation context"):
        for item in st.session_state.history:
            st.write(f"**You:** {item['question']}")

