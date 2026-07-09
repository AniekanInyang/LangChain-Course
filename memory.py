from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from utils import get_langchain_llm

# Number of conversation turns to remember
MEMORY_WINDOW = 3  # Keep last 3 turns (6 messages: 3 human + 3 AI)

if __name__ == '__main__':
    llm = get_langchain_llm()
    
    # Create a simple conversation prompt with history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm
    
    # Create message history storage
    chat_history = ChatMessageHistory()

    # Function to get windowed history
    def get_session_history(session_id: str):
        # Trim messages to keep only last N turns (N*2 messages)
        max_messages = MEMORY_WINDOW * 2
        if len(chat_history.messages) > max_messages:
            # Keep only the last N turns
            chat_history.messages = chat_history.messages[-max_messages:]
        return chat_history
    
    # Wrap chain with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # Test conversation
    print("=" * 50)
    response1 = chain_with_history.invoke(
        {"input": "Hi, my name is Andrew"},
        config={"configurable": {"session_id": "user1"}}
    )
    print("User: Hi, my name is Andrew")
    print(f"Assistant: {response1.content}\n")
    
    print("=" * 50)
    response2 = chain_with_history.invoke(
        {"input": "What is 1+1?"},
        config={"configurable": {"session_id": "user1"}}
    )
    print("User: What is 1+1?")
    print(f"Assistant: {response2.content}\n")
    
    print("=" * 50)
    response3 = chain_with_history.invoke(
        {"input": "What is my name?"},
        config={"configurable": {"session_id": "user1"}}
    )
    print("User: What is my name?")
    print(f"Assistant: {response3.content}\n")
    
    # Show conversation history
    print("=" * 50)
    print("Full Conversation History:")
    for msg in chat_history.messages:
        print(f"{msg.type}: {msg.content}")

    # Test conversation
    conversations = [
        "Hi, my name is Andrew",
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is my name?",  # After 5 turns, it won't remember "Andrew"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print("=" * 50)
        print(f"Turn {i}")
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user1"}}
        )
        print(f"User: {user_input}")
        print(f"Assistant: {response.content}")
        print(f"Messages in memory: {len(chat_history.messages)} (max: {MEMORY_WINDOW * 2})\n")
    
    # Show final conversation history
    print("=" * 50)
    print(f"Final Memory (last {MEMORY_WINDOW} turns):")
    for msg in chat_history.messages:
        print(f"  {msg.type}: {msg.content}")