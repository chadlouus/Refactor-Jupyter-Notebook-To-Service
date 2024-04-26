from langchain.document_loaders import TextLoader
#from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
#from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

print("start loading embeddings")
embeddings = HuggingFaceEmbeddings()
#docsearch = Chroma.from_documents(texts, embeddings)
if os.path.isdir("./chroma_db"):
    print("chroma db already exists - 1sec")
    docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    print("creating chromadb and persisting - 9sec")
    filename="state_of_the_union.txt"
    loader = TextLoader(filename)
    documents = loader.load()
    print("doc len", len(documents))
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("texts len", len(texts))
    
    docsearch = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")


credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("IC_API_KEY") 
}

project_id = os.getenv("WATSON_PROJECT_ID")

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

model_id = ModelTypes.GRANITE_13B_CHAT_V2

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

from langchain.llms import WatsonxLLM
#from langchain_community.llms import WatsonxLLM
#from langchain_ibm import WatsonxLLM

watsonx_granite = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())

query = "What did the president say about Ketanji Brown Jackson"
answer = qa.run(query)
print("answer", answer)

if not os.getenv("SERVER"):
    exit()

from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def handle_question():
    question = request.args.get('q', query)
    if not question:
        return "/?q=question"

    answer = qa.run(question)
    return answer

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
