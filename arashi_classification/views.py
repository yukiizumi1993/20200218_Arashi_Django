from django.shortcuts import render, redirect
from PIL import Image
import numpy as np
import base64
#import tensorflow as tf
import os
import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
#from tensorflow.python.saved_model import tag_constants

#学習モデルのロード
#model = tf.keras.models.load_model('/home/yukzum/yukzum.pythonanywhere.com/model_arashi_classification3.h5')
model = keras.models.load_model('/home/yukzum/yukzum.pythonanywhere.com/model_arashi_classification4.h5')

def predict(input):
    result = model.predict(input)
    return result

def upload(request):

    #画像データの取得
    files = request.FILES.getlist("files[]")

    #簡易エラーチェック（jpg拡張子）
    for memory_file in files:

        root, ext = os.path.splitext(memory_file.name)

        if ext != '.jpg':

            message ="【ERROR】jpg以外の拡張子ファイルが指定されています。"
            return render(request, 'arashi_classification/index.html', {
                "message": message,
                })


    if request.method =='POST' and files:
        result=[]
        labels=[]
        fruits=["相葉雅紀", "松本潤", "二宮和也", "大野智", "櫻井翔"]
        for file in files:
            Z=[]
            img = img_to_array(load_img(file, target_size=(200, 200)))
            Z.append(img)
            Z = np.asarray(Z)
            Z = Z / 255.0
            labels.append(predict(Z))

        for file, label in zip(files, labels):
            file.seek(0)
            file_name = file
            src = base64.b64encode(file.read())
            src = str(src)[2:-1]
            proba = round(label.max()*100, 1)
            result.append((src, proba, fruits[label.argmax()]))

        context = {
            'result': result
           }
        return render(request, 'arashi_classification/result.html', context)

    else:
        return redirect('index')
