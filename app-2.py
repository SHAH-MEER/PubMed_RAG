import os
import gradio as gr
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Configuration - Use Hugging Face Spaces secrets
openai_api_key = os.getenv("OPENAI_API_KEY")
DB_DIR = "vector_db"

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 16}  # Reduced batch size for HF Spaces
)

# Global variables for lazy loading
db = None
qa_chain = None

def initialize_system():
    """Initialize the RAG system - called once when first question is asked"""
    global db, qa_chain
    
    if qa_chain is not None:
        return True
    
    try:
        # Load FAISS vector database
        db = FAISS.load_local(DB_DIR, embeddings=embed_model, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            openai_api_key=openai_api_key
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            memory=memory,
            return_source_documents=True
        )
        
        return True
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False

def chat_with_rag(message, history):
    """
    Chat function optimized for Hugging Face Spaces
    """
    # Check API key
    if not openai_api_key:
        return "⚠️ OpenAI API key not configured. Please set the OPENAI_API_KEY secret in your Hugging Face Space settings."
    
    # Initialize system on first use
    if not initialize_system():
        return "❌ Failed to initialize the RAG system. Please check if the vector database is properly uploaded."
    
    if not message.strip():
        return "Please enter a question about medical research."
    
    try:
        # Convert Gradio history format to LangChain format
        chat_history = []
        if history:
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    chat_history.append((history[i], history[i + 1]))
        
        # Get response from QA chain
        response = qa_chain.invoke({
            "question": message,
            "chat_history": chat_history
        })
        
        # Extract answer
        answer = response["answer"]
        
        return answer
        
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg.lower():
            return "⚠️ Invalid OpenAI API key. Please check your API key in the Space settings."
        elif "rate limit" in error_msg.lower():
            return "⚠️ Rate limit exceeded. Please wait a moment before asking another question."
        else:
            return f"❌ An error occurred: {error_msg}"

# Create Gradio interface optimized for HF Spaces
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="PubMed RAG Chatbot",
        css="""
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
        }
        /* Ensure proper contrast in both light and dark modes */
        .gr-button {
            transition: all 0.2s ease;
        }
        .gr-button:hover {
            transform: translateY(-1px);
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: var(--body-text-color);">
            <h1 style="color: var(--body-text-color);">🔬 PubMed RAG Chatbot</h1>
            <p style="color: var(--body-text-color-subdued);">Ask questions about medical research and get answers from PubMed literature</p>
        </div>
        """)
        
        # Status indicator (dark mode friendly)
        
        # Chat interface
        chatbot = gr.Chatbot(
            height=500,
            placeholder="Ask me anything about medical research...",
            avatar_images=["👤", "🤖"]
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="e.g., What are the latest findings on COVID-19 treatments?",
                container=False,
                scale=7,
                lines=1
            )
            submit = gr.Button("Send", scale=1, variant="primary")
        
        with gr.Row():
            clear = gr.Button("Clear Chat", scale=1)
            
        # Examples
        gr.Examples(
            examples=[
                "What are the side effects of metformin?",
                "Tell me about recent cancer immunotherapy research",
                "What is the mechanism of action of aspirin?",
                "What are the latest findings on Alzheimer's disease?"
            ],
            inputs=msg
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: var(--body-text-color-subdued); font-size: 12px;">
            <p>This chatbot searches through medical literature to provide research-based answers.</p>
            <p><strong>Disclaimer:</strong> This is for informational purposes only and not medical advice.</p>
        </div>
        """)
        
        # Event handlers
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            
            # Update status to show processing
            bot_response = chat_with_rag(message, chat_history)
            chat_history.append([message, bot_response])
            return "", chat_history
        
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot)
        
        return interface

# Create and launch the interface
demo = create_interface()

if __name__ == "__main__":
    # For local testing
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
else:
    # For Hugging Face Spaces
    demo.launch()