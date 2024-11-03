# Comind VS Code Extension

A VS Code extension that provides intelligent question-answering (QA) capabilities using LangChain's tools and Llama-based large language models (LLMs). This extension is designed to support developers by allowing them to ask questions related to codebases and receive relevant, context-aware answers in real time.

## Features

- **Contextual QA**: Ask questions about code, configuration, or documentation within your project. The extension retrieves and processes relevant context from the repository to provide concise answers.
- **Document Loading**: Automatically loads and parses files in your repository, supporting multiple languages (e.g., Python, JavaScript, HTML, CSS).
- **Efficient Text Splitting**: Uses a language-aware text splitter to ensure relevant chunks of documents are processed without cutting off important information.
- **Vector Search with Chroma**: Embeds document chunks using `SentenceTransformer` models and stores them in a `Chroma` vector database for efficient retrieval.
- **LLM-based Answer Generation**: Utilizes `Ollama`'s Llama2 model for accurate and concise answer generation.

## Installation

To install the extension:
1. Clone this repository or download the code.
2. Open the project in Visual Studio Code.
3. Build and run the extension within VS Code's extension development environment.

## Usage

1. **Initialize the Database**: This extension loads documents from the repository and prepares them for question-answering. It will:
   - Parse code and text files within the repository (supporting `.py`, `.js`, `.css`, `.html`, `.tsx`, `.jsx`, and `.ipynb` file types).
   - Handle Unicode errors gracefully, skipping files with incompatible encodings.
2. **Ask Questions**: Use the extension to pose questions about your code. The QA system will fetch relevant context from your documents, process it through the LLM, and provide a concise answer.
3. **Database Setup and Querying**: 
   - Call `initialize_the_db(repo_path)` to load and process documents from your repository.
   - Use `initialize_the_qa_chain(db, user_question)` to ask questions, leveraging the embedded context for responses.

### Example

```python
# Initializing the database
db = initialize_the_db("path/to/your/repo")

# Asking a question
question = "How does the main function in main.py work?"
answer = initialize_the_qa_chain(db, question)
print(answer)
