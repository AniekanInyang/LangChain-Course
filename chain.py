from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from utils import get_langchain_llm

llm = get_langchain_llm(temperature=0.9)

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

chain = prompt | llm 

product = "Queen Size Sheet Set"
response = chain.invoke({"product": product})
print(response)
print("Response:")
print(response.content)

# Simple Sequential Chain

# OLD WAY (deprecated):
# from langchain.chains import SimpleSequentialChain
# chain1 = LLMChain(llm=llm, prompt=prompt1)
# chain2 = LLMChain(llm=llm, prompt=prompt2)
# overall_chain = SimpleSequentialChain(chains=[chain1, chain2])

# NEW WAY (LCEL):
# Chain 1: Generate company name
prompt1 = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

# Chain 2: Write a catchphrase for the company
prompt2 = ChatPromptTemplate.from_template(
    "Write a catchy slogan for the following company: {company_name}"
)

# Connect chains using pipe operator
chain1 = prompt1 | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

# Create sequential chain
sequential_chain = (
    chain1  # Output: company_name (string)
    | (lambda company_name: {"company_name": company_name})  # Convert to dict
    | chain2  # Takes dict with "company_name" key
)

# Run it
product = "Queen Size Sheet Set"
result = sequential_chain.invoke({"product": product})
print("Final Result:")
print(result)

# Sequential Chain

# Chain 1: Generate company name
prompt1 = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

# Chain 2: Generate company description
prompt2 = ChatPromptTemplate.from_template(
    "Write a 20-word description for the following company: {company_name}"
)

# Chain 3: Generate slogan using BOTH product and company name
prompt3 = ChatPromptTemplate.from_template(
    "Write a catchy slogan for {company_name} that makes {product}"
)

# Build the sequential chain
chain = (
    # Start with input
    RunnablePassthrough.assign(
        company_name=(prompt1 | llm | StrOutputParser())
    )
    # Now we have: {"product": "...", "company_name": "..."}
    | RunnablePassthrough.assign(
        description=(prompt2 | llm | StrOutputParser())
    )
    # Now we have: {"product": "...", "company_name": "...", "description": "..."}
    | RunnablePassthrough.assign(
        slogan=(prompt3 | llm | StrOutputParser())
    )
    # Final output: {"product": "...", "company_name": "...", "description": "...", "slogan": "..."}
)

# Run it
result = chain.invoke({"product": "Queen Size Sheet Set"})

print("Product:", result["product"])
print("Company Name:", result["company_name"])
print("Description:", result["description"])
print("Slogan:", result["slogan"])

# MultiPromptChain and LLMRouter

# Define different prompts for different topics
physics_template = """You are a physics professor. \
Answer the following question in a technical way:

{input}"""

math_template = """You are a math professor. \
Answer the following question with equations and proofs:

{input}"""

history_template = """You are a history professor. \
Answer the following question with historical context:

{input}"""

general_template = """Answer the following question:

{input}"""

# Create prompt templates
physics_prompt = ChatPromptTemplate.from_template(physics_template)
math_prompt = ChatPromptTemplate.from_template(math_template)
history_prompt = ChatPromptTemplate.from_template(history_template)
general_prompt = ChatPromptTemplate.from_template(general_template)

# Create chains for each topic
physics_chain = physics_prompt | llm | StrOutputParser()
math_chain = math_prompt | llm | StrOutputParser()
history_chain = history_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()

# Create a classifier to determine which chain to use
classifier_template = """Given the user question below, classify it as either being about \
`physics`, `math`, `history`, or `general`.

Do not respond with more than one word.

Question: {input}
Classification:"""

classifier_prompt = ChatPromptTemplate.from_template(classifier_template)
classifier_chain = classifier_prompt | llm | StrOutputParser()

# Create the router using RunnableBranch
def route_question(info):
    """Route to appropriate chain based on classification."""
    classification = classifier_chain.invoke({"input": info["input"]}).lower().strip()
    
    if "physics" in classification:
        return physics_chain
    elif "math" in classification:
        return math_chain
    elif "history" in classification:
        return history_chain
    else:
        return general_chain

# Alternative: Using RunnableBranch (more declarative)
router_chain = RunnableBranch(
    (
        lambda x: "physics" in classifier_chain.invoke({"input": x["input"]}).lower(),
        physics_chain
    ),
    (
        lambda x: "math" in classifier_chain.invoke({"input": x["input"]}).lower(),
        math_chain
    ),
    (
        lambda x: "history" in classifier_chain.invoke({"input": x["input"]}).lower(),
        history_chain
    ),
    general_chain  # Default chain
)

# Test the router
questions = [
    "What is the speed of light?",
    "What is the integral of x^2?",
    "When did World War 2 end?",
    "What's the weather like today?"
]

print("=" * 60)
print("ROUTER CHAIN DEMO")
print("=" * 60)

for question in questions:
    print(f"\nQuestion: {question}")
    
    # First, classify
    classification = classifier_chain.invoke({"input": question})
    print(f"Classification: {classification}")
    
    # Then route and answer
    answer = router_chain.invoke({"input": question})
    print(f"Answer: {answer}\n")
    print("-" * 60)

# Function based LLM Router and multi prompt chain

# Define specialized prompts
PROMPT_TEMPLATES = {
    "physics": """You are a physics professor. Answer technically:
{input}""",
    
    "math": """You are a math professor. Show equations and proofs:
{input}""",
    
    "history": """You are a history professor. Provide historical context:
{input}""",
    
    "general": """Answer the following question:
{input}"""
}

def create_router_chain():
    """Create a router that classifies and routes questions."""
    
    # Classifier prompt
    classifier_prompt = ChatPromptTemplate.from_template(
        """Classify this question as 'physics', 'math', 'history', or 'general'.
        
        Question: {input}

        Respond with only ONE word - the category."""
    )
    
    def route_and_answer(question: str) -> dict:
        """Classify question and route to appropriate expert."""
        
        # Step 1: Classify
        classification = (classifier_prompt | llm | StrOutputParser()).invoke(
            {"input": question}
        ).lower().strip()
        
        # Step 2: Get the right prompt template
        template = PROMPT_TEMPLATES.get(classification, PROMPT_TEMPLATES["general"])
        
        # Step 3: Create and run the specialized chain
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"input": question})
        
        return {
            "question": question,
            "category": classification,
            "answer": answer
        }
    
    return route_and_answer

# Create and test the router
router = create_router_chain()

# Test questions
test_questions = [
    "What is Newton's second law?",
    "Solve x^2 + 5x + 6 = 0",
    "Who was the first US president?",
    "What's a good recipe for pasta?"
]

print("=" * 60)
print("ROUTER CHAIN - Simpler Version")
print("=" * 60)

for question in test_questions:
    result = router(question)
    print(f"\n📝 Question: {result['question']}")
    print(f"🏷️  Category: {result['category']}")
    print(f"💬 Answer: {result['answer']}")
    print("-" * 60)

#Using RunnableLambda

# Prompts for each specialty
physics_prompt = ChatPromptTemplate.from_template(
    "You are a physics expert. {input}"
)
math_prompt = ChatPromptTemplate.from_template(
    "You are a math expert. {input}"
)
history_prompt = ChatPromptTemplate.from_template(
    "You are a history expert. {input}"
)
general_prompt = ChatPromptTemplate.from_template(
    "{input}"
)

# Chains for each specialty
physics_chain = physics_prompt | llm | StrOutputParser()
math_chain = math_prompt | llm | StrOutputParser()
history_chain = history_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()

# Classifier
classifier = (
    ChatPromptTemplate.from_template(
        "Classify as 'physics', 'math', 'history', or 'general': {input}"
    )
    | llm
    | StrOutputParser()
)

# Router function
def route(info):
    topic = classifier.invoke(info).lower()
    
    if "physics" in topic:
        return physics_chain.invoke(info)
    elif "math" in topic:
        return math_chain.invoke(info)
    elif "history" in topic:
        return history_chain.invoke(info)
    else:
        return general_chain.invoke(info)

# Create the full router chain
router_chain = RunnableLambda(route)

# Test it
questions = [
    {"input": "What is gravity?"},
    {"input": "What is 2+2?"},
    {"input": "When was the Roman Empire founded?"},
    {"input": "How do I bake bread?"}
]

for q in questions:
    print(f"\nQ: {q['input']}")
    print(f"A: {router_chain.invoke(q)}")
    print("-" * 60)

#Using Pydantic output parser

print("\n" + "=" * 70)
print("PYDANTIC PARSER DEMO")
print("=" * 70)

# Define structure
class RouteDecision(BaseModel):
    category: str = Field(description="One of: physics, math, history, general")
    confidence: str = Field(description="One of: high, medium, low")
    reasoning: str = Field(description="Brief explanation of classification")

# Create parser
route_parser = PydanticOutputParser(pydantic_object=RouteDecision)

# Get format instructions
format_instructions = route_parser.get_format_instructions()

# Create prompt
classifier_with_parser = ChatPromptTemplate.from_template(
    """Classify this question and explain your reasoning.

Question: {input}

{format_instructions}
"""
)

# Build chain WITH parser
classification_chain_parsed = (
    classifier_with_parser
    | llm
    | route_parser  # ← Parser is called here automatically
)

# Test it
test_question = "What is the speed of light in a vacuum?"

print(f"\nQuestion: {test_question}\n")

result = classification_chain_parsed.invoke({
    "input": test_question,
    "format_instructions": format_instructions
})

print(f"Type of result: {type(result)}")
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")

# Compare with simple string approach
print("\n" + "-" * 70)
print("COMPARE: Simple String Approach")
print("-" * 70)

simple_classifier = ChatPromptTemplate.from_template(
    "Classify as ONE WORD: physics, math, history, or general.\n\nQuestion: {input}"
)

simple_chain = simple_classifier | llm | StrOutputParser()

simple_result = simple_chain.invoke({"input": test_question})

print(f"\nType of result: {type(simple_result)}")
print(f"Result: {simple_result}")
print("\nNotice: Simple approach gives you just a string, no structure!")