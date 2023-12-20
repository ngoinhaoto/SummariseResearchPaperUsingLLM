from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub


import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.on_chat_start
async def on_chat_start():

    elements = [
        cl.Image(name="image1", display="inline", path="./robot.jpg")
    ]

    await cl.Message(content = "Hello there. Welcome to AskAnythingBot", elements = elements).send()
    files = None
    
    while files is None:
        files = await cl.AskFileMessage(
            content = "Please upload the PDF File to begin!",
            accept = ['application/pdf'],
            max_size_mb = 20,
            timeout = 180
        ).send()

    file = files[0]

    msg = cl.Message(content = f"Processing {file.name}")
    await msg.send()

    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""

    for page_number in range(len(pdf.pages)):
        pdf_text += pdf.pages[page_number].extract_text()
    
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name= 'all-MiniLM-L6-v2')
    # embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    llm = HuggingFaceHub(repo_id = 'mistralai/Mistral-7B-v0.1',
                         model_kwargs = {'temperature': 0.01, "max_new_tokens": 200})

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        # ChatOpenAI(temperature=0),
        llm, 
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    print(message.content)

    res = await chain.acall(message.content, callbacks = [cb])
    answer = res["answer"]
    sources = res["sources"].strip()

    print(answer)
    print(sources)

    source_elements = []

    # Get the metadata and texts from user sessino
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m['source'] for m in metadatas]
    texts = cl.user_session.get("texts")


    if sources:
        found_sources = []

        for source in sources.split(","):
            source_name = source.strip().replace('.', "")
            source_name = source_name.replace(' ', '')

            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue


            text = texts[index]
            found_sources.append(source_name)

            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    final_answer = cl.Message(content = "")
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:   
        # await cl.Message(content=answer, elements=source_elements).send()
        for chunk in answer.split(" "):
            await final_answer.stream_token(chunk + " ")
        
        final_answer.elements = source_elements
        await final_answer.send()

        
