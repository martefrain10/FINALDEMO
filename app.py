import os
from operator import itemgetter

import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

primary_qa_llm = ChatOpenAI(model_name="gpt-4o-2024-05-13", temperature=0)

loader = PyMuPDFLoader(
    "/Users/chelsea.hottovy/Final/Final_Demo/updated_salesforce_cases_3.pdf",
)

documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=  6800, chunk_overlap=300)

documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_vector_store = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="T3ES - Salesforce",
)
retriever = qdrant_vector_store.as_retriever()
template = """Answer the question based only on the following context. Make sure you are thorough in your answers, don't leave out information. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hey there! What's the name of the customer and the reservation number?").send()

    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Based on the following context, Summarize ALL closed and opened cases related to the reservation number.
                Provide your answer in this format:

            case number:
            date opened:
            case status:
            summary: 

                Use all cases associated with the same name to provide 3 specific ways the agent can add a personal touch to the interaction.  
                {context}""",
            ),
            ("human", "{question}"),

            
        ]
    )

    retrieval_augmented_qa_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        # | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
        | (prompt | primary_qa_llm)
    )
    cl.user_session.set("runnable", retrieval_augmented_qa_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk.content)

    await msg.send()






