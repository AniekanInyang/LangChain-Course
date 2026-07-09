"""
Azure OpenAI Service Module
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, APIConnectionError, RateLimitError, chat
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try loading from current directory
    load_dotenv()

logger = logging.getLogger(__name__)


def _check_azure_openai_config():
    """Check if Azure OpenAI credentials are configured."""
    required_vars = ['AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_MODEL_NAME']
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        return False, f"Missing Azure OpenAI config: {', '.join(missing)}"
    return True, None


def get_openai_client():
    """Initialize and return Azure OpenAI client. Raises error if config missing."""
    is_configured, error_msg = _check_azure_openai_config()
    if not is_configured:
        raise ValueError(error_msg)
    return AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
    )


def get_completion(prompt, model=None):
    """Generate a response from Azure OpenAI with the given prompt."""
    if model is None:
        model = os.environ.get("AZURE_OPENAI_MODEL_NAME")
    
    client = get_openai_client()
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def get_langchain_llm(temperature=0, timeout=30, max_retries=1):
    llm = AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    return llm

def get_langchain_embeddings():
    """Get Azure OpenAI embeddings for vector operations."""
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  # Usually "text-embedding-ada-002"
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
    )
    return embeddings


if __name__ == "__main__":
    try:
        is_configured, error_msg = _check_azure_openai_config()
        if not is_configured:
            logger.warning(error_msg)
            print("Azure OpenAI not configured. Please set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_MODEL_NAME.")
        else:
            client = get_openai_client()
            print("Azure OpenAI client initialized successfully!\n")
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI client: {e}")
        print(f"Failed to initialize Azure OpenAI client: {e}")


    # Test the get_completion function
    test_prompt = "What is the capital of France?"
    print(f"Test Prompt: {test_prompt}")
    response = get_completion(test_prompt)
    print(f"Response: {response}")