import os
import json
import streamlit as st # type: ignore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch # type: ignore
from langchain.agents import create_agent
from langchain.tools import tool
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
)

missing = []
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")
if not TAVILY_API_KEY:
    missing.append("TAVILY_API_KEY")
if not WEATHER_API_KEY:
    missing.append("WEATHER_API_KEY")

if missing:
    st.error(f"Missing environment variable(s): {', '.join(missing)}. Add them to your .env file.")
    st.stop()

search_tool = TavilySearch(max_results=2)


@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a given city."""
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_API_KEY}&query={city}"
    )
    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch the weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


@st.cache_resource
def get_agent():
    llm = ChatOpenAI(
        model="poolside/laguna-s-2.1:2.5",
        temperature=0,
        max_tokens=200,
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    tools = [search_tool, get_weather_data]
    return create_agent(llm, tools)


agent_executor = get_agent()

st.title("🤖 Agentic AI Assistant")
st.caption("Search + Weather AI Agent using LangChain")

query = st.text_input(
    "Enter your query:",
    placeholder="Find the capital of France and the current weather",
)

run_clicked = st.button("Run Agent", type="primary")

if run_clicked:
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_executor.invoke(
                    {"messages": [("user", query)]}
                )
                final_answer = response["messages"][-1].content

                st.success("Response Generated")
                st.subheader("Final Response")
                st.write(final_answer)

                # Optional: show full trace (tool calls, intermediate steps)
                with st.expander("View full agent trace"):
                    for msg in response["messages"]:
                        role = getattr(msg, "type", "unknown")
                        st.markdown(f"**{role}**")
                        st.text(msg.content)

            except Exception as e:
                st.error(f"Something went wrong: {e}")