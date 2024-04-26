# Converting a Jupyter Notebook to a Service
## Existing Jupyter Notebook using WatsonX, Chroma and LangChain
IBM WatsonX provides a Jupyter Notebook illustrating how to do Retrieval Augmented Generation using Chroma, LLM and Langchain.

The Jupyter Notebook requires specific Python libraries at particular versions, in order to avoid module conflicts. When running the Notebook, it is recommended to start with a fresh Python environment. If you have other versions of Python, or other versions of the required modules already in your running environment, then it is likely that the Notebook could fail.

Here are the steps of ensuring the correct versions of the modules are setup in your running environment.

## Environment Setup for running Notebook
Start with a fresh Python environment.

    python3 -m venv venv
    source venv/bin/activate

Here we create a new virtual environment and then activate to use the new environment.

## Installing required versions of Python Modules
Create a `requirements.txt` file, and then add the required modules and their version numbers.

    langchain==0.0.345
    chromadb==0.3.26
    ibm-watson-machine-learning>=1.0.335
    pydantic==1.10.0
    tqdm
    sentence-transformers==2.7.0

Make sure that the `langchain` version and `ibm-watson-machine-learning` version is consistent with those in the Notebook.

Install the required modules in the virtual environment by
    pip install -r requirements.txt

## Run the Code from the Notebook
Then we install the required versions of modules for running the Notebook.

    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings

It is a good practice to run `python3` interactively and paste in the code, to make sure that all the imports are loaded correctly.

### Setup Environment Variables using .env

Create a .env file with the keys. (The values below are simulated)

    API_KEY=Nf_8CZvwhoysVEBL2NbQDHKbUNAg-SVgkjWIln_XnrQe
    WATSON_PROJECT_ID=7708b8af-9fc9-4e39-b158-fb5c74bd69a3

## Create a Docker Image
### Create a Dockerfile

The python-3.11-alpine image is smaller, but there is an issue that pytorch cannot be installed in the base image. So `python-3.11` image should be used.

In the Dockerfile, we install the python modules listed in requirements.txt.

### Performance Enhancement for Docker Image
In order to start the Docker image fast, we want to perform initialization during Docker image build. The Embeddings module takes quite some time to initialize, so it's better to do it during base image build.

So we run the following during module installation steup.

    python3 -c 'from langchain.embeddings import HuggingFaceEmbeddings;HuggingFaceEmbeddings()'

### Running the Notebook Code

For running the Notebook code, we can just run

    CMD ["sh", "-c", "python ragsetup.py"]

### Performance Enhancement for Generating the VectorStore

The process of creating the vectore store takes quite some time. During this process, the knowledge text is converted into vectors. This process takes about 9 seconds.

If the vector store is already created on disk, then loading it only takes 1 second.

So we write code to test if the vector store has already been created, and only create it if it has not been created.

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

### Running the Notebook Code from Docker

docker build -t ragnotebook .
docker run --env-file .env -p 5002:5000 ragnotebook

## Refactoring the Notebook to Handle Additional Questions

A Jupyter Notebook typically answers one question, in this case the hard-coded question is "What did the president say about Ketanji Brown Jackson".

We can create a service that can answer other questions based on the knowledge text. For this service, we are going to create a web service, using Python's Flask framework.

    from flask import Flask, request

    app = Flask(__name__)

    @app.route("/", methods=['GET'])
    def hello_world():
        question = request.args.get('q', query)
        if not question:
            return "/?q=question"

        answer = qa.run(question)
        return answer

    if __name__ == "__main__":
        app.run()

Now we can ask additional questions based on the knowledge text.

curl "http://localhost:5002/?q=what+is+the+topic+about"



