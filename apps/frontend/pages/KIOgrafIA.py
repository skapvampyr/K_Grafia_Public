import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable
import uuid
import requests
import json
import sys
import time
import random

# Env variables needed by langchain
#os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
langserve_url = os.environ["LANGSERVE_URL"]

# app config
st.set_page_config(page_title="KIOgrafIA", page_icon="🤖", layout="wide")

with st.sidebar:
    st.markdown("""# Instrucciones""")
    st.markdown("""

Este ChatBot esta enfocado a dar información general sobre los clientes de KIO CyberSecurity.

Tiene acceso a la siguiente información:

- Service Desk (SD+), aquí puedes encontrar información relacionada a los tickets.
- CMDB, aquí puedes encontrar información relacionada a los equipos administrados por el SOC.
- SalesForce, aquí puedes encontrar información relacionada a las oportunidades de los clientes. 


Ejemplos:

- Cuántos TICKETS, tiene el CLIENTE Abilia?
- Dame el detalle del TICKET número 123098
- Qué OPORTUNIDADES tiene el CLIENTE Procesar?
- Cúal es el TCV de cada OPORTUNIDAD?
- Qué TECNOLOGIAS se administran para el CLIENTE Toka?
                

    """)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)


# ENTER HERE YOUR LANGSERVE FASTAPI ENDPOINT
# for example: "https://webapp-backend-botid-zf4fwhz3gdn64-staging.azurewebsites.net"

#url = " http://127.0.0.1:8000" + "/agent/stream_events"
url = langserve_url + "/agent/stream_events"

def get_or_create_ids():
    """Generate or retrieve session and user IDs."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())
    return st.session_state['session_id'], st.session_state['user_id']

    
def consume_api(url, user_query, session_id, user_id):
    """Uses requests POST to talk to the FastAPI backend, supports streaming."""
    headers = {'Content-Type': 'application/json'}
    config = {"configurable": {"session_id": session_id, "user_id": user_id}}
    payload = {'input': {"question": user_query}, 'config': config}
    
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        try:
            response.raise_for_status()  # Raises an HTTPError if the response is not 200.
            for line in response.iter_lines():
                if line:  # Check if the line is not empty.
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        # Extract JSON data following 'data: '.
                        json_data = decoded_line[len('data: '):]
                        try:
                            data = json.loads(json_data)
                            if "event" in data:
                                kind = data["event"]
                                if kind == "on_chat_model_stream":
                                    content = data["data"]["chunk"]["content"]
                                    if content:  # Ensure content is not None or empty.
                                        yield content  # Two newlines for a paragraph break in Markdown.
                                elif kind == "on_tool_start":
                                        tool_inputs = data['data'].get('input')
                                        if isinstance(tool_inputs, dict):
                                            # Joining the dictionary into a string format key: 'value'
                                            inputs_str = ", ".join(f"'{v}'" for k, v in tool_inputs.items())
                                        else:
                                            # Fallback if it's not a dictionary or in an unexpected format
                                            inputs_str = str(tool_inputs)
                                        yield f"Searching Tool: {data['name']} with input: {inputs_str} ⏳\n\n"
                                elif kind == "on_tool_end":
                                        yield "Search completed.\n\n"
                            elif "content" in data:
                                # If there is immediate content to print, with added Markdown for line breaks.
                                yield f"{data['content']}\n\n"
                            elif "steps" in data:
                                yield f"{data['steps']}\n\n"
                            elif "output" in data:
                                yield f"{data['output']}\n\n"
                        except json.JSONDecodeError as e:
                            yield f"JSON decoding error: {e}\n\n"
                    elif decoded_line.startswith('event: '):
                        pass
                    elif ": ping" in decoded_line:
                        pass
                    else:
                        yield f"{decoded_line}\n\n"  # Adding line breaks for plain text lines.
        except requests.exceptions.HTTPError as err:
            yield f"HTTP Error: {err}\n\n"
        except Exception as e:
            yield f"An error occurred: {e}\n\n"


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hola! estoy para apoyarte como facilitador en el consumo de información centralizada de nuestros clientes en Cyber. Tengo conocimiento operativo (tickets en SD+), comercial (oportunidades de Salesforce) así como de los activos (CMDB) para la entrega de nuestro servicio. \n\n A la izquierda podrás ver algunas palabras clave que pueden guiarte en nuestra interacción.\n\n  ¿Cómo puedo ayudarte? ")]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input

session_id, user_id = get_or_create_ids()

user_query = st.chat_input("Esquibe tu pregunta aquí...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner(text=""):
            response = st.write_stream(consume_api(url, user_query, session_id, user_id))

    st.session_state.chat_history.append(AIMessage(content=response))