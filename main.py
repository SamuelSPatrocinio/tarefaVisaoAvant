import cv2
import numpy as np
import os
import auxiliarFunctions as aux
from pyzbar.pyzbar import decode

#pegar a quantidade de arquivos dentro da pasta
imageList = os.listdir("Imagens")
numberOfImages = len(imageList) + 1

#realizar operacoes em cada uma das imagens
for imageNumber in range(1,numberOfImages):

    #caminho das imagens
    path = 'Imagens/photo' + str(imageNumber) + '.png'

    #pegar a imagem
    orig = cv2.imread(path)
    photo = orig.copy()                                                                             #Copia da imagem original

    #tratando a imagem
    photo = photo[1:photo.shape[0]-1,1:photo.shape[1]-1]                                            #Cortando 1 pixel das imagens para evitar erros dos prints
    masked = aux.maskOutColor(photo,[110,50,50],[130,255,255])                                      #Pegando somente a cor azul
    invertedFloodFill = aux.floodFillandInvert(masked)                                              #Aplicando um floodfill
    result = invertedFloodFill | masked                                                             #Somando a mascara e o flood fill para gerar regiões brancas onde há plataformas

    #checando se a imagem não é toda preta ou toda branca, ou seja, apresenta platafomas
    if np.mean(result) != 0 and np.mean(result) != 255: 
        #localizando as plataformas
        firstUpscales = []
        contours,hierarchy = cv2.findContours(result,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)         #Definindo contornos
        for c in contours:   
            x,y,w,h = cv2.boundingRect(c)
            cropped = photo[y:y+h,x:x+w]
            resized = cv2.resize(cropped,(w*4,h*4),interpolation=cv2.INTER_CUBIC)
            firstUpscales.append(resized)
        
        #cortando os qrcodes
        count = 0
        for image in firstUpscales:
            count += 1
            masked = aux.maskOutColor(image,[0,0,200],[0,0,255])                                    #Pegando pretos e brancos
            edged = cv2.Canny(masked,20,200)
            contours,hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>1:
                rectanglePoints = []
                for c in contours:
                    x,y,w,h = cv2.boundingRect(c)
                    rectanglePoints.append([x,y,x+w,y+h])
                rectanglePoints = np.asarray(rectanglePoints)
                left, top = np.min(rectanglePoints, axis=0)[:2]
                right, bottom = np.max(rectanglePoints, axis=0)[2:]
                qrcode = image[top-1:bottom+1,left-1:right+1]                                       #Dando uma pequena margem para garantir que o qr code não tenha sido cortado
                bigImage = aux.centerImage([800,800,3],np.uint8,100,qrcode)                         #Imagem grande cinza
                result = decode(bigImage)
                if(not result):
                    print("Couldnt Decode")
                for i in result:
                    print(i.data.decode("utf-8"))
                cv2.imshow('Image'+str(imageNumber)+'/Qrcode' + str(count),bigImage)
                cv2.waitKey(100)
cv2.waitKey(0)