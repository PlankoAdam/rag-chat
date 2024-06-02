from main_chain import chain
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.vectorstores import faiss
from langchain_experimental.text_splitter import SemanticChunker
from PyPDF2 import PdfReader
import os
from time import time

import langchain.text_splitter 

EMBEDDING_PATH = os.path.abspath("./models/embedding/nomic-embed-text-v1.5.f32.gguf")
PDF_PATH = os.path.abspath("./tests/contract.pdf")
TEST_QUERY = 'How much is the deposit and what is it for?'

embedding_model = LlamaCppEmbeddings(
  model_path=EMBEDDING_PATH, 
  n_gpu_layers=-1, 
  verbose=False
)

semantic_splitter = SemanticChunker(embedding_model)

splitters = [
  langchain.text_splitter.RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=300, chunk_overlap=50),
  langchain.text_splitter.RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=500, chunk_overlap=100),
  langchain.text_splitter.RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200),
]

def read_pdf(fpath):
    f = open(fpath, 'rb')
    pdf = PdfReader(f);
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    f.close()
    return text

def similarity_top_n_result(n, retriever_results):
  ret = ''
   
  for i in range(n):
    ret += retriever_results[i].page_content
    
  return ret 

def test_splitter(splitter, text, query, top_n):
  vs_start = time()  
  chunks = splitter.split_text(text)
  vs = faiss.FAISS.from_texts(chunks, embedding=embedding_model)
  vs_end = time()

  retriever = vs.as_retriever()

  res_start = time()
  ret_result = similarity_top_n_result(top_n, retriever.invoke(query))
  answer = chain.invoke({'context':ret_result, 'query':query})
  res_end = time()

  print(f'\nExtracted context: \n"{ret_result}"')
  print(f'\nAnswer: \n{answer}')

  print(f'\nTime to create vector store: {(vs_end-vs_start) * 1000} ms')
  print(f'Time to find context and answer question: {(res_end-res_start) * 1000} ms')


def run_test():
  text = read_pdf(PDF_PATH)

  print(f'Testing query: {TEST_QUERY}\n')

  print('Testing semantic chunker...')
  test_splitter(semantic_splitter, text, TEST_QUERY, 1)
  print('\n################################################################################################')

  for n in range(3):
    print(f'\n\nTesting recursive splitters (top {n+1} result)...')
    for idx, spl in enumerate(splitters):
      print(f'Splitter number {idx}:')
      test_splitter(spl, text, TEST_QUERY, n+1)
      print('\n################################################################################################')
