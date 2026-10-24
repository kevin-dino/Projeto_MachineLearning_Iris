import streamlit as st
import joblib
# Configuração inicial da página
st.set_page_config(
    page_title="Classificador de Flores Iris",
    page_icon="🌸",
    layout="centered"
)

# Título principal
st.title("🌸 Classificador de Flores Iris")

st.write(
    """
    Este aplicativo recebe as características de uma flor Iris e,
    futuramente, usará um modelo de Machine Learning para tentar adivinhar
    a espécie da planta.
    """
)

st.divider()

# Seção explicativa
st.header("Informe as características da flor")

st.info(
    """
    Preencha os valores abaixo com as medidas da flor em centímetros.
    
    O modelo Iris normalmente utiliza quatro características:
    
    - Comprimento da sépala
    - Largura da sépala
    - Comprimento da pétala
    - Largura da pétala
    """
)

# Criando duas colunas para deixar o layout mais bonito
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Comprimento da sépala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Comprimento da pétala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Largura da sépala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Largura da pétala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

st.divider()

# Mostrando os dados informados pelo usuário
st.subheader("Dados informados")

dados_iris = {
    "Comprimento da sépala": sepal_length,
    "Largura da sépala": sepal_width,
    "Comprimento da pétala": petal_length,
    "Largura da pétala": petal_width
}

st.write(dados_iris)

# Botão para fazer a previsão
botao_adivinhar = st.button("🔍 Adivinhar espécie da flor", type="primary")

if botao_adivinhar:
    modelo = joblib.load("modelos\\knn_flores_iris.joblib")
    
    entrada_modelo = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]
    previsão = modelo.predict(entrada_modelo)[0]
    if previsão == 0:
        st.title("setosa")
        st.image("https://th.bing.com/th/id/OIP.5hwM8IiYNrBYqYP4aHWK1AHaFh?w=265&h=197&c=7&r=0&o=7&pid=1.7&rm=3")
    elif previsão == 1:
        st.title("versicolor")
        st.image("https://th.bing.com/th/id/OIP.RwgtFmwDoPiXY-FU735vewHaHa?w=201&h=200&c=7&r=0&o=7&pid=1.7&rm=3")
    else:
        st.title("virgilica")
        st.image("https://th.bing.com/th/id/OIP.1Nf75qITrqW2dIADHZdRYAHaE8?w=274&h=183&c=7&r=0&o=7&pid=1.7&rm=3")

    st.code(entrada_modelo, language="python")

    st.warning(
        """
        A implementação do modelo ainda não foi adicionada.
        O próximo passo será carregar o arquivo `.joblib` e usar `model.predict()`.
        """
    )

st.divider()

#