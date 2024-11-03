from comind import *
import gradio as gr


def answer_query(repo_path, user_question):
    # Initialize the database with documents from the given repository path
    db = initialize_the_db(repo_path)
    # Use the initialized database and the user's query to get an answer
    answer = initialize_the_qa_chain(db, user_question)
    return answer

# Define the Gradio interface
iface = gr.Interface(
    fn=answer_query,
    inputs=[
        gr.Textbox(label="Repository Path", placeholder="Enter the path to your code repository here..."),
        gr.Textbox(label="User Question", placeholder="Enter your question here...")
    ],
    outputs=[
        gr.Textbox(label="Answer")
    ],
    title="Code Repository QA System",
    description="Enter the path to your code repository and ask a question to get an answer based on the repository's contents."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(share=True)
