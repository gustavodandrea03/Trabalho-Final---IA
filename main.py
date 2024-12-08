import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(BASE_DIR, "models", "dog_cat_classifier.keras")
if not os.path.exists(CAMINHO_MODELO):
    raise FileNotFoundError(f"Modelo não encontrado no caminho: {CAMINHO_MODELO}")

NOMES_CLASSES = ['gato', 'cachorro']

LIMITE_CONFIANCA = 0.9

print("Carregando modelo...")
model = load_model(CAMINHO_MODELO)
print("Modelo carregado com sucesso!")

# Função para prever imagem
def prever_imagem(caminho_imagem, limite_confianca=LIMITE_CONFIANCA):
    print(f"🔄 Quase lá... {caminho_imagem}")
    try:
        imagem = load_img(caminho_imagem, target_size=(224, 224))
        imagem_array = img_to_array(imagem) / 255.0
        imagem_array = np.expand_dims(imagem_array, axis=0)

        predicao = model.predict(imagem_array)
        classe_prevista = NOMES_CLASSES[np.argmax(predicao)]
        confianca = np.max(predicao)

        if confianca < limite_confianca:
            print("⚠️ A confiança da predição é baixa.")
            print("\n Para obter melhores resultados, veja algumas sugestões:")
            print("- Tente uma imagem com melhor iluminação.")
            print("- Certifique-se de que o animal ocupa a maior parte da imagem.")
            print("- Use uma imagem com maior resolução.")

        return classe_prevista, confianca

    except Exception as e:
        print(f"❌ Erro ao processar a imagem. Tente novamente: {e}")
        return None, None

# Função para visualizar a previsão
def visualizar_previsao(caminho_imagem, classe_prevista, confianca):
    imagem = load_img(caminho_imagem, target_size=(224, 224))
    plt.imshow(imagem)
    plt.title(f"Pronto! É um: {classe_prevista.capitalize()} (Confiança: {confianca:.2f})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    print("=" * 50)
    print("\n  🐾   Bem-vindo!   🐾")
    print("\n Este programa utiliza técnicas de visão computacional para identificar se uma imagem contém um cachorro ou gato.")
    print("\n Instruções:")
    print(" - Basta fornecer o caminho da imagem, e o sistema fará a análise.")
    print(" - Digite 'f' para encerrar o programa.")
    print("=" * 50)

    while True:
        caminho_imagem_teste = input(" Insira o caminho da imagem (ou 'f' para sair): ").strip()

        if caminho_imagem_teste.lower() == 'f':
            print("\nEncerrando o programa...")
            print("=" * 50)
            break

        if os.path.exists(caminho_imagem_teste):
            print("🔄 Processando a imagem, aguarde...")
            classe_prevista, confianca = prever_imagem(caminho_imagem_teste)

            if classe_prevista:
                print(f" Resultado: {classe_prevista.capitalize()} com confiança de {confianca:.2f}")
                visualizar_previsao(caminho_imagem_teste, classe_prevista, confianca)
            else:
                print("⚠️ Não foi possível fazer a previsão. Verifique se a imagem está clara.")
        else:
            print("❌ Caminho inválido. Verifique o arquivo e tente novamente.")
