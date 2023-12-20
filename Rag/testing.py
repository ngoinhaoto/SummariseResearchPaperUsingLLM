from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser

load_dotenv()

hub_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

prompt = PromptTemplate(
    input_variables = ['question'],
    template = "Translate English to SQL: {question}"
)

hub_chain = LLMChain(prompt = prompt, llm = hub_llm, verbose = True)

# runnable = prompt | hub_llm | StrOutputParser()

print(hub_chain.run("What is the average age of the respondents using a mobile device"))
# print(runnable.invoke({"question": "What is the average age of the respondents using a mobile device"}))