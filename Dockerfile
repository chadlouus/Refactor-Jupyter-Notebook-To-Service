FROM python:3.10

# copy every content from the local file to the image
COPY . /app

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python3 -c 'from langchain.embeddings import HuggingFaceEmbeddings;HuggingFaceEmbeddings()' && \
    #echo "initializing VectorStore and preprocessing" && \
    #python3 ragsetup.py && \
    echo "done installing"
# Set the environment variable

CMD ["sh", "-c", "export SERVER=true && python3 ragsetup.py"]
