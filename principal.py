import cv2
from ultralytics import YOLO

#Modelo a ser usado.
model = YOLO("modelos/keruv_v1.pt")


def iniciandoprograma():
    #Conectando a câmera 
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro Não achei a camera")
    else:
        print("Camera rodando, para fechar aperte 'q'")

        while cap.isOpened():
            sucess, frame = cap.read()

            if sucess:

                #A ia está analisando oque está vendo + Confirmação necessária para mostrar
                results = model(frame, conf=0.5)

                #Nesse momento ela desenha os resultados na tela
                annotated_frame = results[0].plot()

                #Abre a janela da câmera
                cv2.imshow("Teste de IA - Gabriel", annotated_frame)

                #Aperte "Q" Para fechar a janela.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
            
                #Limpa os arquivo temporários que foram gerados durante a finalização do programa.
    cap.release()   #Desliga a camera
    cv2.destroyAllWindows() #Fecha as janelas

iniciandoprograma()