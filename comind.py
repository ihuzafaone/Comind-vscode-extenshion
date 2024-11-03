
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import sys
def safe_load_documents(loader):
    """
    Attempt to load documents using the provided loader, skipping any that cause a UnicodeDecodeError.
    """
    documents = []
    for blob in loader.lazy_load():
        try:
            # Attempt to convert the blob to a string here, to catch encoding errors
            text = blob.as_string()
            documents.append(text)  # Append the text instead of the blob
        except UnicodeDecodeError as e:
            print(f"Skipping a document due to encoding error: {e}")
    return documents
def initialize_the_db(repo_path):
    try:
        # Load documents
        loader = GenericLoader.from_filesystem(
            repo_path,
            glob="**/*",
            suffixes=[".py", ".js", ".css", ".html", ".tsx", ".jsx", "ipynb"],
            exclude=["**/non-utf8-encoding.py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        )
        documents = loader.load()

        # Split documents
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=200, chunk_overlap=20
        )
        texts = python_splitter.split_documents(documents)

        # Create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load embeddings into Chroma
        db = Chroma.from_documents(texts, embedding_function)
        return db
    except Exception as e:
        print(f"Error initializing the database: {e}")
        sys.exit(1)

def initialize_the_qa_chain(db, user_question):

    # Prompt template remains the same
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    # Retrieving documents
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(user_question)
    
    # Initialize LLM (Large Language Model)
    llm = Ollama(model="llama2")
    chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
    answer = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return answer["output_text"]