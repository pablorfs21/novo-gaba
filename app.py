import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import pandas as pd # Biblioteca de tabelas
from datetime import datetime

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Corretor Planilha", page_icon="📝")

# Inicializa a tabela na memória (Session State) se não existir
if 'historico_notas' not in st.session_state:
    st.session_state['historico_notas'] = []

# --- 2. BANCO DE GABARITOS ---
BANCO_DE_PROVAS = {
    "Matemática 9º Ano": {
        0: (1, 150.0), 1: (4, 320.0), 2: (0, 100.0), 3: (3, 500.0), 4: (1, 200.0), 5: (1, 100.0), 6: (1, 100.0), 7: (1, 100.0), 8: (1, 100.0), 9: (1, 100.0), 10: (1, 100.0), 11: (1, 100.0), 12: (1, 100.0)
    },
    "Ciências 8º Ano": {
        0: (1, 150.0), 1: (4, 320.0), 2: (0, 100.0), 3: (3, 500.0), 4: (1, 200.0), 5: (1, 100.0), 6: (1, 100.0), 7: (1, 100.0), 8: (1, 100.0), 9: (1, 100.0), 10: (1, 100.0), 11: (1, 100.0), 12: (1, 100.0)
    }
}
ALTERNATIVAS = 5

# --- 3. FUNÇÃO DE VISÃO COMPUTACIONAL ---
def processar_imagem(image, gabarito_config):
    # Conversão para Cinza e Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Achar contornos do papel
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    
    if docCnt is None:
        return None, None, "Bordas não encontradas. Use fundo escuro."

    # Perspectiva
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Achar bolinhas
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    if not questionCnts:
        return None, None, "Nenhuma resposta detectada."

    try:
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    except:
        return None, None, "Erro na leitura das linhas."

    # Correção
    correct_score = 0.0
    paper_draw = paper.copy()
    
    for (q, i) in enumerate(np.arange(0, len(questionCnts), ALTERNATIVAS)):
        if q not in gabarito_config:
            break
            
        cnts_row = contours.sort_contours(questionCnts[i:i + ALTERNATIVAS])[0]
        bubbled = None

        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        k = gabarito_config[q][0]
        valor = gabarito_config[q][1]
        color = (0, 0, 255)

        if k == bubbled[1]:
            color = (0, 255, 0)
            correct_score += valor

        cv2.drawContours(paper_draw, [cnts_row[k]], -1, color, 3)

    return correct_score, paper_draw, None

# --- 4. INTERFACE DO APP ---
st.title("🏫 Corretor com Planilha")

# --- BARRA LATERAL (INPUTS) ---
with st.sidebar:
    st.header("1. Configuração")
    nome_prova = st.selectbox("Selecione a Prova", list(BANCO_DE_PROVAS.keys()))
    gabarito_atual = BANCO_DE_PROVAS[nome_prova]
    
    st.divider()
    st.header("2. Dados do Aluno")
    nome_aluno = st.text_input("Nome do Aluno")
    turma = st.selectbox("Turma", ["9º A", "9º B", "8º A", "8º B", "7º A"])

    # Botão para limpar a lista (caso comece uma turma nova)
    st.divider()
    if st.button("🗑️ Limpar Lista de Notas"):
        st.session_state['historico_notas'] = []
        st.rerun()

# --- ÁREA PRINCIPAL ---
col_cam, col_result = st.columns([1, 1])

with col_cam:
    st.subheader("📸 Captura")
    img_file_buffer = st.camera_input("Tirar Foto")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Processar
    nota_tri, img_result, erro = processar_imagem(cv2_img, gabarito_atual)

    if erro:
        st.error(erro)
    else:
        # Arredondar
        nota_final = int(round(nota_tri))
        max_nota = int(sum([v[1] for k, v in gabarito_atual.items()]))

        with col_result:
            st.subheader("✅ Resultado")
            st.metric("Nota Calculada", f"{nota_final} / {max_nota}")
            st.image(img_result, caption="Espelho da Correção")

            # --- BOTÃO DE CONFIRMAR ---
            # Só aparece se tiver foto e nota calculada
            if st.button("💾 Adicionar à Tabela", type="primary"):
                if nome_aluno == "":
                    st.warning("Escreva o nome do aluno antes de salvar!")
                else:
                    # Adiciona na memória
                    novo_registro = {
                        "Data": datetime.now().strftime("%d/%m %H:%M"),
                        "Nome": nome_aluno,
                        "Turma": turma,
                        "Prova": nome_prova,
                        "Nota": nota_final
                    }
                    st.session_state['historico_notas'].append(novo_registro)
                    st.success(f"Nota de {nome_aluno} salva!")

# --- 5. EXIBIÇÃO DA TABELA E DOWNLOAD ---
st.divider()
st.header("📊 Lista da Turma")

if len(st.session_state['historico_notas']) > 0:
    # Cria a tabela visual (DataFrame)
    df = pd.DataFrame(st.session_state['historico_notas'])
    st.dataframe(df, use_container_width=True)

    # Botão de Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Baixar Planilha (Excel/CSV)",
        data=csv,
        file_name=f"notas_{turma}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
else:
    st.info("Nenhuma nota salva ainda. Corrija uma prova e clique em 'Adicionar à Tabela'.")

