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
import chromadb
from chromadb.utils import embedding_functions


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

    metadatas = [{"source": f"{num_element + i}-pl-{num_sources}",
                  "documents": file.name} for i in range(len(texts))]

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

    return texts, metadatas, file.name


async def summarize_one_file(llm, docs):
    msg = cl.Message(content = "")
    database = chromadb.PersistentClient(path = './database')

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key= os.getenv('OPENAI_API_KEY'),
    )

    collection = database.get_collection('langchain',
                                         embedding_function = openai_ef)

    collect = collection.get(
        where = {"documents": {'$eq': docs}},
    )

    list_docs = collect['documents']
    list_metas = collect['metadatas']

    docs = [Document(page_content = a,
                     metadata = b) for a, b in zip(list_docs, list_metas)]
    
    # print(docs)

    chain = load_summarize_chain(llm, chain_type = 'map_reduce')
    await cl.Message("Summarizing...").send()    


    async for chunk in chain.astream(docs):
        print(chunk)
        await msg.stream_token(chunk['output_text'])

    await msg.send()


async def upload_file(id_sources, embeddings):
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

    msg = cl.Message(content = f"Processing {file.name}")
    await msg.send()

    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)

    pdf_text = ""

    for page_number in range(len(pdf.pages)):
        pdf_text += pdf.pages[page_number].extract_text()
    
    texts = text_splitter.split_text(pdf_text)

    num_element = id_sources['id']
    num_sources = id_sources['num_sources']

    metadatas = [{"source": f"{num_element + i}-pl-{num_sources}",
                  "documents": file.name} for i in range(len(texts))]

    save_db = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas = metadatas, persist_directory = './database'
    )

    # docs = [Document(text, metadata) for text, metadata in zip(texts, metadatas)]
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    return texts, metadatas, file.name
