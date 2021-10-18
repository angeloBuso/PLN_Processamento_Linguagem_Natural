# importar os pacotes necessarios para projeto
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
import time
from wordcloud import WordCloud
import streamlit as st
import pt_core_news_md
#pip install --force-reinstall streamlit==0.69
from PIL import Image
nltk.download(['stopwords', 'punkt'])


def main():
    stopwords = nltk.corpus.stopwords.words('portuguese')

    # SideBar -> ambiente de controle
    st.sidebar.subheader("Desenvolvido por:")
    st.sidebar.markdown('[![Angelo Buso]'
                            '(https://img.shields.io/badge/Angelo-Buso-green)]'
                            '(https://linktr.ee/angelobuso)')

    st.sidebar.title("Parâmetros de Controle")
    st.sidebar.subheader('Visualizando as stopwords')
    st.sidebar.markdown("""
    São palavras que não agregam informações para gerar uma nuvem de palavras.
    Possuem uma frequência alta no texto e não trasmitem a ideia central,
    por isso são desconsideradas para criação da nuvem!
    """)
    st.sidebar.dataframe(stopwords_view(stopwords))

    novas_stops = nltk.word_tokenize(st.sidebar.text_input('Insira palavra por palavra, separadas por vírgula'))
    atualizar_stops = st.sidebar.button("Adicionar stopwords")
    st.sidebar.markdown("""
    Dica: para atualizar sua lista de stopwords, conheça bem o domínio do texto com que esta trabalhando.
    """)

    st.title("Construindo uma Nuvem de Palavras")
    st.markdown("""
    O objetivo deste aplicativo é ter uma **visão inicial** das principais palavras do seu texto, utilizando algumas técnicas de
    PLN (Processamento da Linguagem Natural).
    Para mais informações e formas de customizar acesse o [script completo](https://github.com/angeloBuso/PLN_Processamento_Linguagem_Natural/blob/main/WordCloud_PLN.ipynb).
    """)
    st.subheader('Instruções:')
    st.markdown("""
    (a) Insira seu texto no campo abaixo.

    (b) Clique no botão "Gerar Nuvem Palavras".

    (c) **Opcional** após gerar sua nuvem, se sentir necessidade de atualizar a lista de stopwords, na lateral "Parâmetros de Controle"
    existe a opção de atualizar as *stopwords*.
    """)
    df = st.text_area('Texto para a Nuvem de Palavras')

    if st.button("Gerar Nuvem Palavras"):
        with st.spinner('Espere um pouquinho, estou contando as palavras do seu texto!'):
            time.sleep(2)
            st.pyplot(gera_nuvem(stopwords, df))
            st.subheader("Top 30 palavras mais frequentes no texto")
            st.dataframe(more_freq(stopwords, df))
    elif atualizar_stops:
        with st.spinner('Espere um pouquinho, estou recontando as palavras do seu texto!'):
            time.sleep(2)
            for i in novas_stops:
                stopwords.append(i)
            st.sidebar.dataframe(stopwords_view(stopwords))        
            st.pyplot(gera_nuvem(stopwords, df))
            st.subheader("Top 30 palavras mais frequentes no texto")
            st.dataframe(more_freq(stopwords, df))


@st.cache
def tokenizar(df):
    nlp = spacy.load('pt_core_news_md')
    doc = nlp(df)
    tokens_letras = [token.orth_ for token in doc if token.is_alpha]
    return tokens_letras

def more_freq(stopwords, df):    
    frequencia = nltk.FreqDist([token for token in tokenizar(df) if token.lower() not in stopwords])
    top_30 = pd.DataFrame(frequencia.most_common(30), columns=['Palavras', 'Quantidade'])
    return top_30

def palavras_nuvem(stopwords, df):
    only_texto = " ".join(token for token in tokenizar(df) if token.lower() not in stopwords)
    return only_texto

def gera_nuvem(stopwords, df):
    wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(palavras_nuvem(stopwords, df))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    plt.tight_layout()
    return fig

def stopwords_view(stopwords):
    stopword_view = pd.DataFrame(stopwords, columns=['Stopwords'])
    return stopword_view

if __name__ == '__main__':
    main()

st.subheader("Este é um projeto usando apenas linguagem Python e técnicas de *PLN*")
st.markdown("""
Assim como a *Visão Computacional* ensina as máquinas como "enxergamos" o mundo com suas cores e formas,
***Processamento Linguagem Natural - PLN*** surge com suas técnicas e modelos para instruí-las a lerem,
escreverem e compreenderem a **língua humana**. Os usos de técnicas de PLN 
no nosso dia-a-dia vão desde os *apps* de tradução automáticas, nuvens de palavras, até *chatbot's*.

Processamento da Linguagem Natural - PLN em sua essência, é uma das sub-áreas da inteligência artificial que
combina a Linguística (ciência das línguas) e a Ciência da Computação.


(**ps.** a língua em que me refiro é o **código de comunicação**, exemplos português, libras, python, etc e
 não o músculo do rosto do corpo humano rsrsrsr)""")

st.markdown("Para mais informações entre em contato")

qr_code = Image.open('angelobuso.png')

st.image(qr_code, width = 100)
st.markdown("""
[Angelo Buso](https://linktr.ee/angelobuso)
""")

    
