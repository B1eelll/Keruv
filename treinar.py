#Boa Sorte Gabriel! Pense no seu futuro!

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") ##Ia já pré treinada
#model = YOLO("yolov8n.yaml") #Ia do zero

train_results = model.train( #Alugns paramêtros de treinamento.
        data="epi.yaml", #é o arquivo de treinamento.
        epochs=20,  #Quantas vezes a ia vai passar por cada arquivo
        imgsz=640, #tamanho da imagem
        device="cpu", #Nesse caso o processador vai treinar a rede

    )

metrics = model.val() #Serve para ver oque a ia aprendeu durante o treinamento

results = model("path/to/image.jpg") 
results[0].show()
path = model.export(format="onnx") #Deixa a ia em um formato em que outros programas possam ler

#--
