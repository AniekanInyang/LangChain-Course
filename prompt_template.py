# LangChain Prompt Template

import os
from langchain_core.prompts import ChatPromptTemplate
from utils import get_langchain_llm
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


def langchain_prompt_template(template_string):
    prompt_template = ChatPromptTemplate.from_template(template_string)
    return prompt_template

class ReviewOutput(BaseModel):
    gift: bool = Field(description="Was the item purchased\
                                as a gift for someone else? \
                                Answer True if yes,\
                                False if not or unknown.")
    delivery_days: int = Field(description="How many days\
                                        did it take for the product\
                                        to arrive? If this \
                                        information is not found,\
                                        output -1.")
    price_value: list[str] = Field(description="Extract any\
                                        sentences about the value or \
                                        price, and output them as a \
                                        comma separated Python list.")


if __name__ == "__main__":
    llm = get_langchain_llm()

    template_string = """Translate the text that is delimited by triple backticks \
    into a style that is {style}. text: ```{text}```"""

    prompt_template = langchain_prompt_template(template_string)
    print(prompt_template.messages[0].prompt)

    style = """American English in a calm and respectful tone"""

    text = """Arrr, I be fuming that me blender lid flew off and splattered me \
    kitchen walls with smoothie! And to make matters worse, the warranty don't \
    cover the cost of cleaning up me kitchen. I need yer help right now, matey!"""

    customer_messages = prompt_template.format_messages(
                    style=style,
                    text=text)
    print(type(customer_messages))
    print(type(customer_messages[0]))
    print(customer_messages[0])

    # Call the LLM to translate to the style of the customer message
    customer_response = llm.invoke(customer_messages)
    print("Response:")
    print(customer_response.content)

    service_reply = """Hey there customer, the warranty does not cover \
    cleaning expenses for your kitchen because it's your fault that \
    you misused your blender by forgetting to put the lid on before \
    starting the blender. Tough luck! See ya!"""

    service_style_pirate = """A polite tone that speaks in English Pirate"""

    # Call the LLM to generate and translate a response to the customer
    service_messages = prompt_template.format_messages(
                    style=service_style_pirate,
                    text=service_reply)
    print(service_messages[0].content)

    service_response = llm.invoke(service_messages)
    print("Response:")
    print(service_response.content)

    # Extract and parse an expected output using a prompt template
    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """

    review_template = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product \
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value

    text: {text}
    """

    review_prompt_template = langchain_prompt_template(review_template)

    review_messages = review_prompt_template.format_messages(
        text=customer_review
    )

    output = llm.invoke(review_messages)
    print("Output:")
    print(output.content)
    print(type(output.content))
    #It's a string, so we need to parse it as a dict

    output_parser = JsonOutputParser(pydantic_object=ReviewOutput)

    # This generates the prompt to parse the output as expected, from the schemas given
    format_instructions = output_parser.get_format_instructions()

    print(format_instructions)

    review_template_2 = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product\
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    text: {text}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=review_template_2)

    messages = prompt.format_messages(text=customer_review, 
                                    format_instructions=format_instructions)
    print(messages[0].content)

    response = llm.invoke(messages)
    print(response.content)

    output_dict = output_parser.parse(response.content)
    print(output_dict)
    print(type(output_dict))
    print(output_dict['gift'])
    print(output_dict.get('delivery_days'))



