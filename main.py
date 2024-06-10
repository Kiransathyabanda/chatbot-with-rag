
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import GPT4AllEmbeddings
from dotenv import load_dotenv
import re
import os
 
load_dotenv()
 
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = ""  
 
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
 
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
 
user_template = '''
<div class="chat-message user">
    <div class="avatar">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
 
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
 
user_template = '''
<div class="chat-message user">
    <div class="avatar">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
 
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text_1 = page.extract_text()
            if text_1:  
                text_1 = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text_1)
                text_1 = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text_1.strip())
                text_1 = re.sub(r"\n\s*\n", "\n\n", text_1)
                text += text_1
    return text
 
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=600,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
 
def get_vectorstore(text_chunks):
    embeddings = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
 
def get_conversation_chain(vectorstore):
    llm = ChatGroq(groq_api_key=groq_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
 
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
 
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
 
def main():
    st.write(css, unsafe_allow_html=True)
 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mode" not in st.session_state:
        st.session_state.mode = 'chat'
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
 
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)
 
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
 
    if st.session_state.mode == 'chat':
        st.title("Chat with Me!")
        st.write("Hello! I'm your friendly chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")
 
        for message in st.session_state.chat_history:
            st.write(f"**You:** {message['human']}")
            st.write(f"**Chatbot:** {message['AI']}")
 
        def submit_question():
            user_question = st.session_state.user_question
            if user_question:
                groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
                prompt = ChatPromptTemplate.from_messages([
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ])
                conversation = LLMChain(llm=groq_chat, prompt=prompt, verbose=True, memory=memory)
                response = conversation.predict(human_input=user_question)
 
                message = {'human': user_question, 'AI': response}
                st.session_state.chat_history.append(message)
 
                st.session_state.user_question = ""
 
        st.text_input("Ask a question:", value=st.session_state.user_question, key='user_question', on_change=submit_question)
        st.button("Send", on_click=submit_question)
 
        if st.button("Switch to Document Analysis"):
            st.session_state.mode = 'document_analysis'
            st.experimental_rerun()
 
    elif st.session_state.mode == 'document_analysis':
        st.header("Document Analysis")
 
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
 
        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
 
        if st.button("Switch to Chat Mode"):
            st.session_state.mode = 'chat'
            st.experimental_rerun()
 
if __name__ == '__main__':
    main()
 
