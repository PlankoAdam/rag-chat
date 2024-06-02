from langchain_community.llms import llamacpp
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings

# The main chain

template = """[INST]You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer.
Context:
'...{context}...'[/INST]

Human: {human}
Chatbot:"""
prompt_template = PromptTemplate(template=template, input_variables=["context", "human"])
model = llamacpp.LlamaCpp(
    model_path="./models/llm/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0,
    max_tokens=1500,
    stop=['\n'],
    verbose=True,
    echo=True,
    # n_ctx=1024,
    n_gpu_layers=-1,
)
rag_chain = prompt_template | model | StrOutputParser()

# Query contextualization chain

contextualize_q_system_prompt = """[INST]Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is:
{history}
{question}
[/INST]"""
contextualize_q_prompt = PromptTemplate(template=contextualize_q_system_prompt, input_variables=['history', 'question'])
contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()

embedding_model = LlamaCppEmbeddings(model_path="./models/embedding/nomic-embed-text-v1.5.f32.gguf", n_gpu_layers=-1, verbose=False,)

# Document splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=150, separators=['.', '?', '!']
)

def split(text):
    chunks = splitter.split_text(text=text)
    return chunks

def processTXT(fpath):
    f = open(fpath, 'r', encoding="utf-8");
    text = f.read()
    f.close()
    return split(text)

def processPDF(fpath):
    f = open(fpath, 'rb')
    pdf = PdfReader(f);
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    f.close()
    return split(text)

def createVectorStore(textChunks):
    vector_store = FAISS.from_texts(textChunks, embedding=embedding_model)
    return vector_store

def getSerializedVectorStore(textChunks):
    return createVectorStore(textChunks).serialize_to_bytes()

def getStandaloneQuery(query, msgs):
    if (len(msgs) > 0):
        history_str = ''
        for i,s in enumerate(msgs):
            if i % 2 == 0:
                history_str += "Human: "+s+'\n'
            else:
                history_str += "Chatbot: "+s+'\n'
        standalone_q = contextualize_q_chain.invoke({
            "history": history_str,
             "question": query
        })
    else:
        standalone_q = query

    return standalone_q

def getContext(vector_store, standalone_q):
    retriever = vector_store.as_retriever()
    retriever_result = retriever.invoke(standalone_q)
    return retriever_result[0].page_content

def answer(serialized_vector_store, query, msgs):
    sa_q = getStandaloneQuery(query, msgs)

    vector_store = FAISS.deserialize_from_bytes(serialized=serialized_vector_store, embeddings=embedding_model)
    ctx = getContext(vector_store, sa_q)

    ans = rag_chain.invoke({"context": ctx, "human": sa_q})
    return {"answer": ans, "context": ctx, "standalone_q": sa_q}