from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
import chainlit as cl
from io import BytesIO
import os
import PyPDF2
from PyPDF2 import PdfFileWriter, PdfFileReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma



async def summarize(llm, id_sources, embeddings):
    
    msg = cl.Message(content = "")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)

    files = None
    while files is None:
        files = await cl.AskFileMessage(
                    content = "Please upload the PDF File to begin!",
                    accept = ['application/pdf'],
                    max_size_mb = 20,
                    timeout = 180
        ).send()

    file = files[0]
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)

    pdf_text = ""

    for page_number in range(len(pdf.pages)):
        pdf_text += pdf.pages[page_number].extract_text()
    
    texts = text_splitter.split_text(pdf_text)

    num_element = id_sources['id']
    num_sources = id_sources['num_sources']

    metadatas = [{"source": f"{num_element + i}-pl-{num_sources}"} for i in range(len(texts))]

    save_db = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas = metadatas, persist_directory = './database'
    )

    # docs = [Document(text, metadata) for text, metadata in zip(texts, metadatas)]

    docs = []

    for text, metadata in zip(texts, metadatas):
        print(text)
        print(metadata)
        docs.append(Document(page_content = text, 
                             metadata = metadata))

    chain = load_summarize_chain(llm, chain_type = 'map_reduce')

    await cl.Message("Summarizing...").send()    
    async for chunk in chain.astream(docs):
        print(chunk)
        await msg.stream_token(chunk['output_text'])

    await msg.send()

    return texts, metadatas
