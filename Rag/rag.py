import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import Runnable


loader = WebBaseLoader(
    web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",) ,
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(
            class_ = {'post-content', 'post-title', 'post-header'}
        )
    )
)

docs = loader.load()

text_splitter