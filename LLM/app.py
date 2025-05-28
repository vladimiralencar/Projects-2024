# App Web no Google Colab Para Criar Imagens a Partir de Texto com Stable Diffusion

# Exemplos de prompt:

# Um pôr do sol sobre um lago sereno, com cores vibrantes refletindo na água e silhuetas de árvores contra o céu crepuscular.
# Um astronauta no topo de uma montanha rochosa.
# Portrait of a samurai warrior in a busy city downtown.

# Um pôr do sol sobre um lago sereno arrodeado de árvores, com cores vibrantes

import asyncio
try:  
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Imports
import diffusers
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO

# Título da app
st.title("APP - Gerar imagens a partir de texto")

# Carrega o modelo 
@st.cache_resource
def carrega_modelo():

@st.cache_resource
def carrega_modelo():
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


# Cria o objeto
pipe = carrega_modelo()

# Entrada do usuário para o prompt e número de passos de inferência
prompt = st.text_input("Digite seu prompt para gerar a imagem:", value="Digite o prompt")
steps = st.slider("Escolha o número de passos de inferência", min_value=20, max_value=150, value=50)
seed = st.number_input("Escolha a seed (para reprodutibilidade)", value=1)

# Botão para gerar a imagem
if st.button("Gerar Imagem com IA"):

    with st.spinner('Gerando imagem...'):

        try:

            # Cria o gerador
            gerador = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

            # Executa o pipeline e gera iimagem
            image = pipe(prompt, num_inference_steps=steps, generator=gerador).images[0]

            # Imprime a imagem
            st.image(image, caption="Imagem Gerada")

            # Converte a imagem para o formato adequado para download
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # Botão para baixar a imagem
            st.download_button(
                label="Baixar Imagem",
                data=img_buffer,
                file_name="imagem_gerada_dsa.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Erro ao gerar a imagem: {e}")
