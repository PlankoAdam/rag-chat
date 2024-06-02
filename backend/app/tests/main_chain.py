from langchain_community.llms import llamacpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import os

MODEL_PATH = os.path.abspath("./models/llm/mistral-7b-instruct-v0.3.Q4_K_M.gguf")

model = llamacpp.LlamaCpp(
  model_path=MODEL_PATH,
  temperature=0.1,
  n_gpu_layers=-1,
  stop=['\n'],
  verbose=True,
  echo=True,
  n_ctx=4096
)

template = """[INST]You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer.
If the question cannot be answered from the given context, say you don't know the answer.[/INST]
#####START OF CONTEXT#####
'...{context}...'
#####END OF CONTEXT#####
Question: {query}
Answer:"""

template2 = """[INST]
You are an AI assistant trained to answer questions based on provided context from a document. Only answer the question if the context contains relevant information. If the context does not have the information needed to answer the question, simply respond with "The context does not provide sufficient information to answer the question." Do not attempt to answer the question if the relevant information is not present in the context.

#####START OF CONTEXT#####
{context}
#####END OF CONTEXT#####

Question: {query}
Answer:
[/INST]
"""

prompt_template = PromptTemplate(template=template2, input_variables=["context", "query"])

chain = prompt_template | model | StrOutputParser()