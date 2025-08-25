import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('한국보훈복지의료공단_영상의학과 진료과별 검사건수_인천보훈병원_20241231_with_embedding.csv', encoding='cp949')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('진료과별 챗봇')
st.markdown("based-rule-chatbot With NLP")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input("User Input", "")
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    embedding = model.encode(user_input)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    response = df.loc[df['distance'].idxmax()]['횟수']
    
    st.session_state.past.append(str(user_input))
    st.session_state.generated.append(str(response))

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state["generated"][i], key=str(i) + '_bot')