import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- 1. CONFIGURAÇÃO E CONEXÃO COM GOOGLE SHEETS ---
st.set_page_config(page_title="Corretor & Banco", page_icon="💾")

# Tenta conectar com o Google Sheets
try:
    # Procura as credenciais nos Segredos do Streamlit
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # Se você colou o JSON puro nos secrets com o nome "gcp_service_account":
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    client = gspread.authorize(creds)
    
    # Abre a planilha pelo nome (Tem que ser EXATAMENTE o nome da sua planilha no Google)
    sheet = client.open("Notas Provas").sheet1 
    CONEXAO_OK = True
except Exception as e:
    CONEXAO_OK = False
    ERRO_CONEXAO = e

# --- 2. BANCO DE GABARITOS ---
BANCO_DE_PROVAS = {
    "Matemática 9º Ano": {
        0: (1, 150.0), 1: (4, 320.0), 2: (0, 100.0), 3: (3, 500.0), 4: (1, 200.0)
    },
    "Ciências 8º Ano": {
        0: (0, 200.0), 1: (2, 200.0), 2: (2, 200.0), 3: (1, 200.0), 4: (4, 200.0)
    }
}
ALTERNATIVAS = 5

# --- FUNÇÃO DE PROCESSAMENTO (IGUAL À ANTERIOR) ---
def processar_imagem(image, gabarito_config):
    # (Copie a função processar_imagem inteira do código anterior para cá)
    # Para economizar espaço aqui, estou resumindo, mas você deve manter a lógica de visão computacional.
    # ... [Todo o código do OpenCV aqui] ...
    
    # Simulação do retorno para o exemplo (SUBSTITUA PELA FUNÇÃO REAL):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Placeholder
    return 850.0, image, None # Placeholder

# --- INTERFACE ---
st.title("🏫 Sistema de Correção Integrado")

if not CONEXAO_OK:
    st.error("⚠ Erro de conexão com o Banco de Dados (Google Sheets).")
    st.warning("Verifique se configurou os 'Secrets' e compartilhou a planilha com o email do robô.")
    st.code(str(ERRO_CONEXAO))

# Menu Lateral para Dados do Aluno
with st.sidebar:
    st.header("📝 Dados do Aluno")
    nome_aluno = st.text_input("Nome Completo")
    turma = st.selectbox("Turma", ["9º A", "9º B", "8º A", "8º B"])
    prova_selecionada = st.selectbox("Prova", list(BANCO_DE_PROVAS.keys()))

st.info(f"Corrigindo: **{prova_selecionada}**")
gabarito_atual = BANCO_DE_PROVAS[prova_selecionada]

img_file_buffer = st.camera_input("Foto do Gabarito")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Processamento
    nota_tri, img_result, erro = processar_imagem(cv2_img, gabarito_atual)
    # OBS: Lembre-se de colar a função processar_imagem completa lá em cima!

    if erro:
        st.error(erro)
    else:
        nota_final = int(round(nota_tri))
        max_nota = int(sum([v[1] for k, v in gabarito_atual.items()]))
        
        col1, col2 = st.columns(2)
        col1.metric("Nota Calculada", nota_final)
        col2.metric("Valor Total", max_nota)
        st.image(img_result, caption="Espelho", use_column_width=True)

        # --- BOTÃO DE SALVAR ---
        if CONEXAO_OK:
            st.divider()
            if st.button("💾 SALVAR NO BANCO DE DADOS", type="primary"):
                if nome_aluno == "":
                    st.warning("Por favor, digite o nome do aluno antes de salvar.")
                else:
                    try:
                        # Adiciona linha na planilha: [Data, Nome, Turma, Prova, Nota]
                        data_hoje = datetime.now().strftime("%d/%m/%Y %H:%M")
                        linha = [data_hoje, nome_aluno, turma, prova_selecionada, nota_final]
                        sheet.append_row(linha)
                        
                        st.success(f"Nota de {nome_aluno} salva com sucesso!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")
