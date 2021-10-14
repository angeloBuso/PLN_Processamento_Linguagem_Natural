# importar os pacotes necessarios para projeto
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
from wordcloud import WordCloud
import streamlit as st
#pip install --force-reinstall streamlit==0.69
nltk.download(['stopwords', 'punkt'])

# 1. Variáveis Globais
stopwords = nltk.corpus.stopwords.words('portuguese')

# 2. Função Gerar Nuvem
def gera_nuvem():
    """
    Pipeline que (a) instancia um modelo de linguagem, (b) cria um objeto doc
    (c) itera sobre os tokens que contém letras, (d) retira as stopwords e cria
    uma lista de palavras.
    :return: uma Nuvem de Palavras
    """
    with st.spinner('Espere um pouquinho, estou contando as palavras do seu texto!'):
        nlp = spacy.load('pt_core_news_lg')
        doc = nlp(texto_completo)
        tokens_letras = [token.orth_ for token in doc if token.is_alpha]
        only_texto = " ".join(token for token in tokens_letras if token.lower() not in stopwords)
        wordcloud = WordCloud(stopwords=stopwords,
                              background_color="black").generate(only_texto)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(wordcloud, interpolation='bilinear')
        plt.tight_layout()
        st.pyplot(fig)
        frequencia = nltk.FreqDist([token for token in tokens_letras if token.lower() not in stopwords])
        top_30 = pd.DataFrame(frequencia.most_common(30), columns=['Palavras', 'Quantidade'])
        st.subheader("Top 30 palavras mais frequentes no texto")
        st.write(top_30)

# 3. Construção do dash

# 3.1 SideBar -> ambiente de controle
st.sidebar.subheader("""
Desenvolvido por:
[Angelo Buso](https://github.com/angeloBuso)
""")

st.sidebar.title("Parâmetros de Controle")
st.sidebar.subheader('Visualizando as stopwords')
st.sidebar.markdown("""
São palavras que não agregam informações para gerar uma nuvem de palavras.
Possuem uma frequência alta no texto e não trasmitem a ideia central,
por isso são desconsideradas para criação da nuvem!
""")

stopwords_teste = pd.DataFrame(stopwords, columns=['Stopwords'])
st.sidebar.dataframe(stopwords_teste)

st.sidebar.subheader("Se deseja atualizar a lista padrão de stopwords")
novas_stops = nltk.word_tokenize(st.sidebar.text_input('Insira palavra por palavra, separadas por vírgula'))
novas_stops_bottom = st.sidebar.button("Adicionar stopwords")
st.sidebar.markdown("""
Dica: para atualizar sua lista de stopwords, conheça bem o domínio do texto com que esta trabalhando.
""")

# 3.2 Main -> corpo do app

st.title("Construindo uma Nuvem de Palavras")
st.markdown("""
O objetivo deste aplicativo é ter uma visão inicial das principais palavras do seu texto, usando algumas técnica de
PLN (Processamento da Linguagem Natural).
Mais informações e formas de customizar acesse o [script completo](https://github.com/angeloBuso/PLN_Processamento_Linguagem_Natural/blob/main/WordCloud_PLN.ipynb).
""")
st.subheader('Instruções:')
st.markdown("""
(a) Insira seu texto no campo abaixo.

(b) Clique no botão "Gerar Nuvem Palavras".

(c) **Opcional** após gerar sua nuvem, se sentir necessidade de atualizar a lista de stopwords, na lateral "Parâmetros de Controle"
existe a opção de atualizar as *stopwords*.
""")

texto_completo = st.text_area('Texto para a Nuvem de Palavras')

if st.button("Gerar Nuvem Palavras"):
    gera_nuvem()
elif novas_stops_bottom:
    for i in novas_stops:
        stopwords.append(i)
        stopwords_teste = pd.DataFrame(stopwords, columns=['Stopwords'])
    st.sidebar.dataframe(stopwords_teste)
    gera_nuvem()

#stopwords

st.subheader("Este é um projeto usando apenas linguagem Python e técnicas de *PLN*")
st.markdown("""
Assim como a *Visão Computacional* ensinam as máquinas como "enxergamos" o mundo com suas cores e formas,
***Processamento Linguagem Natural - PLN*** surge com suas técnicas e modelos para instruir as máquinas a lerem,
escreverem e compreenderem a **língua humana**. Os usos de técnicas de PLN 
no nosso dia-a-dia vão desde os *apps* de tradução automáticas, nuvens de palavras, até *chatbot's*.

Processamento da Linguagem Natural - PLN em sua essência, é uma das sub-áreas da inteligência artificial que
combina a Linguística (ciência das línguas) e a Ciência da Computação.


(**ps.** a língua em que me refiro é o **código de comunicação**, exemplos português, libras, python, etc e
 não o músculo do rosto do corpo humano rsrsrsr)""")

st.markdown("Para mais informações entre em contato")
st.markdown("[Portifólio](https://github.com/angeloBuso/data_science_portifolio)")
st.markdown("[GitHub](https://github.com/angeloBuso)")
st.markdown("[LinkedIn](https://www.linkedin.com/in/angelo-buso/)")
