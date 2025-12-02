import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

# --- 1. CONFIGURAÇÃO DA PÁGINA (Deve ser a primeira linha do Streamlit) ---
st.set_page_config(page_title="Corretor TRI", page_icon="📝")

# --- 2. CONFIGURAÇÃO DA PROVA (GABARITO E PESOS) ---
# Estrutura: {Questão: (Índice_Resposta_0a4, Pontos)}
# 0=A, 1=B, 2=C, 3=D, 4=E
CONFIG_PROVA = {
    0: (1, 150.5), # Q1: B, 150.5 pts
    1: (4, 320.0), # Q2: E, 320.0 pts
    2: (0, 100.0), # Q3: A, 100.0 pts
    3: (3, 500.0), # Q4: D, 500.0 pts
    4: (1, 200.0)  # Q5: B, 200.0 pts
}
ALTERNATIVAS = 5

def processar_imagem(image):
    # 1. Converter para Cinza e achar bordas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Encontrar contornos do papel
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
        return None, None, "Não achei o papel. Use fundo escuro e enquadre bem."

    # 3. Transformação de Perspectiva (Deixar reto)
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    
    # 4. Binarização (Preto e Branco puro)
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 5. Achar as bolinhas
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Filtra por tamanho e formato quadrado
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    if not questionCnts:
        return None, None, "Não achei as bolinhas de resposta."

    # Ordenar de cima para baixo
    try:
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    except:
        return None, None, "Erro ao ordenar as questões."

    # --- LÓGICA DE CORREÇÃO ---
    correct_score = 0.0
    paper_draw = paper.copy()
    
    # Processar cada linha de questão
    for (q, i) in enumerate(np.arange(0, len(questionCnts), ALTERNATIVAS)):
        if q not in CONFIG_PROVA:
            break
            
        # Pega as 5 bolinhas da linha atual
        cnts_row = contours.sort_contours(questionCnts[i:i + ALTERNATIVAS])[0]
        bubbled = None

        # Descobre qual está marcada
        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # Verifica acerto
        k = CONFIG_PROVA[q][0]     # Resposta Certa (Índice)
        valor = CONFIG_PROVA[q][1] # Pontos TRI
        color = (0, 0, 255)        # Vermelho

        if k == bubbled[1]:
            color = (0, 255, 0)    # Verde
            correct_score += valor

        cv2.drawContours(paper_draw, [cnts_row[k]], -1, color, 3)

    return correct_score, paper_draw, None

# --- INTERFACE VISUAL ---
st.title("📸 Corretor TRI")
st.markdown("Use um **fundo escuro** para melhor resultado.")

img_file_buffer = st.camera_input("Tirar Foto do Gabarito")

if img_file_buffer is not None:
    # Ler imagem
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with st.spinner('Processando...'):
        nota_tri, img_result, erro = processar_imagem(cv2_img)

    if erro:
        st.error(f"Erro: {erro}")
    else:
        nota_final = int(round(nota_tri))
        max_nota = int(sum([v[1] for k, v in CONFIG_PROVA.items()]))
        
        col1, col2 = st.columns(2)
        col1.metric("Sua Nota", f"{nota_final}")
        col2.metric("Máximo", f"{max_nota}")
        
        st.image(img_result, caption="Correção", use_column_width=True)
        
        if nota_final == max_nota:
            st.success("Parabéns! Gabaritou!")
            st.balloons()