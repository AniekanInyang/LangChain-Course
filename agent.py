from datetime import date
from time import perf_counter
from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from utils import get_langchain_llm

# Next wrap each tool in a try except, so that an erorr does not crash the entire system


llm = get_langchain_llm(temperature=0, timeout=25, max_retries=1)


def start_step(label: str) -> float:
    print(f"{label} [started]", flush=True)
    return perf_counter()


def end_step(label: str, started_at: float) -> None:
    elapsed = perf_counter() - started_at
    print(f"{label} [done in {elapsed:.2f}s]", flush=True)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)


@tool
def safe_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information."""
    try:
        return wiki.invoke(query)
    except Exception as e:
        return (
            "Wikipedia is temporarily unavailable or returned an invalid response. "
            f"Error: {type(e).__name__}: {e}"
        )

agent = create_agent(
    model=llm,
    tools=[safe_wikipedia],
    system_prompt="""
    You are a helpful assistant.
    Use safe_wikipedia whenever factual information is required.
    """
)

step_started = start_step("[1/8] Invoke: Wikipedia bio query")
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Who is Tom M. Mitchell and what book did he write?"
        }
    ]
})
end_step("[1/8] Invoke: Wikipedia bio query", step_started)
print(response["messages"][-1].content)

python = PythonREPLTool()
print("Testing Python REPL tool:")
print("python.invoke(\"2 + 2\")")
print(python.invoke("2 + 2"))
print("python.invoke(\"print(2 + 2)\")")
print(python.invoke("print(2 + 2)"))

agent = create_agent(
    model=llm,
    tools=[python],
    system_prompt="""
    You can execute Python whenever computation is needed.
    """
)

customers = [
    ["Harrison", "Chase"],
    ["Lang", "Chain"],
    ["Dolly", "Too"],
    ["Elle", "Elem"],
    ["Geoff", "Fusion"],
    ["Trance", "Former"],
    ["Jen", "Ayai"]
]

step_started = start_step("[2/8] Stream: sort customers with Python tool")
print("[2/8] Waiting for model/tool events...", flush=True)
step2_final = None
for step in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": f"Sort these customers by last name then first name:\n{customers}",
            }
        ]
    },
    stream_mode="updates",
    config={"recursion_limit": 5},
):
    for step_name, step_data in step.items():
        msg = step_data["messages"][-1]
        print(f"[2/8] update: {step_name}", flush=True)
        step2_final = msg

if step2_final is not None:
    if hasattr(step2_final, "content"):
        print(step2_final.content)
    else:
        step2_final.pretty_print()

end_step("[2/8] Stream: sort customers with Python tool", step_started)

step_started = start_step("[3/8] Stream: calculate 25% of 300")
print("Streaming response:")

for step in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Calculate 25% of 300. Use Python and print the result."
            }
        ]
    },
    stream_mode="values",
    config={"recursion_limit": 5},
):
    step["messages"][-1].pretty_print()
end_step("[3/8] Stream: calculate 25% of 300", step_started)

@tool
def today() -> str:
    """Return today's date."""
    return str(date.today())

agent = create_agent(
    model=llm,
    tools=[today],
    system_prompt="""
    Use the today tool whenever the user asks about today's date.
    """
)

step_started = start_step("[4/8] Invoke: today's date")
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What's today's date?"
        }
    ]
})
end_step("[4/8] Invoke: today's date", step_started)
print(response["messages"][-1].content)


agent = create_agent(
    model=llm,
    tools=[
        safe_wikipedia,
        python,
        today
    ],
    system_prompt="""
    You are a helpful AI assistant.

    You have access to:

    - safe_wikipedia
    - Python
    - Today's date

    Use tools whenever appropriate.

    Important:
    Whenever you use Python_REPL, your code MUST print the final result.
    Do not end Python code with a bare variable or expression.
    After Python_REPL prints the answer, stop using tools and give the final answer.
    """
)

step_started = start_step("[5/8] Invoke: days since Alan Turing birth")
response = agent.invoke({
    "messages":[
        {
            "role":"user",
            "content":
            "How many days has it been since Alan Turing was born?"
        }
    ]
})
end_step("[5/8] Invoke: days since Alan Turing birth", step_started)
print(response["messages"][-1].content)

step_started = start_step("[6/8] Stream: multi-tool Alan Turing query")
print("Streaming response with multiple tools: ---------")

for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "How many days has it been since Alan Turing was born?"
            }
        ]
    },
    stream_mode="updates",
    config={"recursion_limit": 12},
):
    for step_name, step_data in chunk.items():
        print("\n==============================")
        print("STEP:", step_name)

        msg = step_data["messages"][-1]
        msg.pretty_print()
    end_step("[6/8] Stream: multi-tool Alan Turing query", step_started)



class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(description="The city and state, e.g. San Francisco, CA")

llm_with_tools = llm.bind_tools([GetWeather])

step_started = start_step("[7/8] Invoke: tool calling schema check")
ai_msg = llm_with_tools.invoke("What is the weather like in Seattle?")
end_step("[7/8] Invoke: tool calling schema check", step_started)
print(ai_msg.tool_calls)

def get_weather(location: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {location}!"


agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Stream agent responses
step_started = start_step("[8/8] Stream: weather in SF")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].text}")
end_step("[8/8] Stream: weather in SF", step_started)



tool_map = {
    "GetWeather": get_weather
}

tool_call = ai_msg.tool_calls[0]

result = tool_map[tool_call["name"]](
    **tool_call["args"]
)
