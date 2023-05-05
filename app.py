# Bring in deps
import os 
# from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

# os.environ['OPENAI_API_KEY'] = apikey

TEMPERATURE = 0.1

# App framework
st.title('🦜🔗 Tech White Paper GPT Creator')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a technical white paper title about {topic}'
)

exec_summary_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me executive summary paragraph based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

introduction_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'exec_summary'], 
    template='write me introduction paragrapgh for the white paper based on this title: {title} while leveraging this wikipedia research: {wikipedia_research} and this executive paragraph: {exec_summary} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
exec_summary_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
introduction_memory = ConversationBufferMemory(input_key='exec_summary', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=TEMPERATURE) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
exec_summary_chain = LLMChain(llm=llm, prompt=exec_summary_template, verbose=True, output_key='script', memory=exec_summary_memory)
introduction_chain = LLMChain(llm=llm, prompt=introduction_template, verbose=True, output_key='script', memory=introduction_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    exec_summary = exec_summary_chain.run(title=title, wikipedia_research=wiki_research)
    introduction = introduction_chain.run(title=title, wikipedia_research=wiki_research, exec_summary=exec_summary)

    st.write(title) 
    st.write(exec_summary)
    st.write(introduction)

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Executive Summary History'): 
        st.info(exec_summary_memory.buffer)

    with st.expander('Introduction Paragraph History'): 
        st.info(introduction_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
