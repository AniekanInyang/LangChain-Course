from langchain_community.document_loaders import CSVLoader  # ✅ New location
from langchain_community.vectorstores import DocArrayInMemorySearch  # ✅ New location
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import StdOutCallbackHandler
from utils import get_langchain_llm, get_langchain_embeddings

# ============================================================================
# DEBUG MODE - See chain internals
# ============================================================================
import langchain
#langchain.debug = True  # ← Turn on debug mode

# Optional: Set verbosity level
#langchain.verbose = True

#RAG Chain
# ============================================================================
# MODERN RAG (Retrieval Augmented Generation) - Replaces RetrievalQA
# ============================================================================

def create_rag_chain():
    """Modern RAG implementation using LCEL."""
    
    # 1. Load documents
    loader = CSVLoader(file_path="data.csv")
    docs = loader.load()
    
    # 2. Create vector store
    vectorstore = DocArrayInMemorySearch.from_documents(
        docs,
        embedding=get_langchain_embeddings()  # You'll need this
    )
    
    # 3. Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Retrieve top 5 relevant docs. Leave empty if to return all
    )

    def format_docs(docs):
        """Convert list of Document objects to a single string."""
        return "\n\n".join([doc.page_content for doc in docs])
    
    # 4. Create RAG prompt
    template = """Answer the question based only on the following context:
    
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Create LLM
    llm = get_langchain_llm()
    
    # 6. Build RAG chain using LCEL (replaces RetrievalQA)
    rag_chain = (
        {
            "context": retriever | format_docs,  # Retrieves relevant docs
            "question": RunnablePassthrough()  # Passes question through
        }
        | prompt  # Formats with context and question
        | llm  # Generates answer
        | StrOutputParser()  # Extracts string
    )
    
    return rag_chain

# Use it
if __name__ == "__main__":
    chain = create_rag_chain()
    answer = chain.invoke("What is the price of product X?",
                          config = {
                              "run_name": "ExperimentA", #useful for monitoring in LangSmith and debugging
                              "tags": ["experiment", "version_b", "pricing"],
                              "callbacks": [StdOutCallbackHandler()], #For debugging and printing logs. Per invocation log. Debugging logs for the specific invocation, as opposed to debug = True or False set globally
                              "metadata": {
                                "user_id": "user_123",
                                "session_id": "session_456",
                                "timestamp": "2024-01-15T10:30:00"
                                }
                          }
    )
    print(answer)

# This uses stuff chaining (i.e all documents passed to the prompt)
# You can have map reduce (ask all documents separately, combine answers and get a final result)
# Refine (ask document 1, get answer, refine answer using document 1 and 2, get refined answer, refine again with document 3 and so on)
# Map Rerank (ask all documents separately, get separate answers, then have a final step to choose the best answer or combine them in some way)

# Chunking Guidelines
# Good chunk sizes:
#chunk_size=500      # For detailed retrieval
#chunk_size=1000     # For general use
#chunk_size=2000     # For broader context

# Overlap (prevents info loss at boundaries):
#chunk_overlap=50    # 10% overlap
#chunk_overlap=100   # 20% overlap
