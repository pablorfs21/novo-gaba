import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Corretor Multi-Provas", page_icon="📚")

# --- 2. BANCO DE GABARITOS (AQUI VOCÊ EDITA) ---
# Estrutura: "Nome da Prova": { Questão: (Índice_Resposta, Pontos) }
# 0=A, 1=B, 2=C, 3=D, 4=E
BANCO_DE_PROVAS = {
    "Matemática 9º Ano (Recuperação)": {
        0: (1, 150.0), # Q1: B
        1: (4, 320.0), # Q2: E
        2: (0, 100.0), # Q3: A
        3: (3, 500.0), # Q4: D
        4: (1, 200.0)  # Q5: B
    },
    "Matemática 8º Ano (Bimestral)": {
        0: (0, 200.0), # Q1: A
        1: (2, 200.0), # Q2: C
        2: (2, 200.0), # Q3: C
        3: (1, 200.0), # Q4: B
        4: (4, 200.0)  # Q5: E
    },
    "Ciências 7º Ano": {
        0: (3, 100.0), # Q1: D
        1: (3, 100.0), # Q2: D
        2: (1, 100.0), # Q3: B
        3: (0, 350.0), # Q4: A
        4: (2, 350.0)  # Q5: C
    }
}
ALTERNATIVAS = 5

# --- FUNÇÃO DE PROCESSAMENTO (Adaptada para receber o gabarito escolhido) ---
def processar_imagem(image, gabarito_config):
    # 1. Pré-processamento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Achar papel
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
        return None, None, "Não achei o papel. Use fundo escuro."

    # 3. Perspectiva
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 4. Achar bolinhas
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    if not questionCnts:
        return None, None, "Não achei as bolinhas."

    try:
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    except:
        return None, None, "Erro ao ordenar questões."

    # --- CORREÇÃO DINÂMICA ---
    correct_score = 0.0
    paper_draw = paper.copy()
    
    # Usa o 'gabarito_config' que foi passado como parâmetro
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

        # Pega os dados do gabarito selecionado
        k = gabarito_config[q][0]     
        valor = gabarito_config[q][1] 
        color = (0, 0, 255)

        if k == bubbled[1]:
            color = (0, 255, 0)
            correct_score += valor

        cv2.drawContours(paper_draw, [cnts_row[k]], -1, color, 3)

    return correct_score, paper_draw, None

# --- INTERFACE VISUAL ---
st.title("🏫 Corretor Escolar")

# 1. MENU DE SELEÇÃO (A novidade está aqui)
nome_prova = st.selectbox(
    "Selecione a Prova para corrigir:",
    list(BANCO_DE_PROVAS.keys())
)

# Pega a configuração baseada na escolha
gabarito_selecionado = BANCO_DE_PROVAS[nome_prova]

st.info(f"Corrigindo: **{nome_prova}**")

# 2. CÂMERA
img_file_buffer = st.camera_input("Tirar Foto")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with st.spinner('Processando...'):
        # Passamos o gabarito escolhido para a função
        nota_tri, img_result, erro = processar_imagem(cv2_img, gabarito_selecionado)

    if erro:
        st.error(f"Erro: {erro}")
    else:
        nota_final = int(round(nota_tri))
        max_nota = int(sum([v[1] for k, v in gabarito_selecionado.items()]))
        
        col1, col2 = st.columns(2)
        col1.metric("Nota Aluno", f"{nota_final}")
        col2.metric("Valor Prova", f"{max_nota}")
        
        st.image(img_result, caption="Correção", use_column_width=True)
        
        if nota_final == max_nota:
            st.balloons()
