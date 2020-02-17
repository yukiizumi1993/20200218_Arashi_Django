from django.shortcuts import render, redirect
from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image
import numpy as np
import base64, os, keras, cv2, io
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.preprocessing.image import array_to_img, img_to_array, load_img


#学習モデルのロード
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('/home/yukzum/yukzum.pythonanywhere.com/model_arashi_classification.h5')

#predict関数を定義
def predict(input):
    result = model.predict(input)
    return result

#画像から顔部分を抽出
def detect(image):
    cascade_path = "/home/yukzum/yukzum.pythonanywhere.com/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.array(gray, dtype='uint8')
    facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    x = facerect[0][0]
    y = facerect[0][1]
    w = facerect[0][2]
    h = facerect[0][3]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    '''
    for (x, y, w, h) in facerect:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    '''

    return [cv2.resize(image[y:y+h, x:x+w], (100, 100)) for x, y, w, h in facerect]

#画像をアップロード
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
        image_files=[]

        for file in files:
            img = np.asarray(Image.open(file))
            faces = detect(img)

            if not faces:
                message ="【ERROR】すみません。顔が認識できませんでした。"
                return render(request, 'arashi_classification/index.html', {
                    "message": message,
                    })

            faces_array = np.array(faces[0]).reshape(1, 100, 100, 3)
            #faces_array = np.array(faces).reshape((len(faces), 100, 100, 3))

            labels.append(predict(faces_array / 255.0))

            image = Image.fromarray(np.uint8(img))
            image_io = io.BytesIO()
            image.save(image_io, format='JPEG')
            image_file = InMemoryUploadedFile(image_io, None, 'foo.jpg', 'image/jpeg', image_io.getbuffer().nbytes, None)
            image_files.append(image_file)

        for file, label in zip(image_files, labels):
            file.seek(0)
            src = base64.b64encode(file.read())
            src = str(src)[2:-1]
            proba = round(label.max()*100, 1)
            arashi=["相葉雅紀", "松本潤", "二宮和也", "大野智", "櫻井翔"]
            result.append((src, proba, arashi[label.argmax()]))

        context = {'result': result}

        return render(request, 'arashi_classification/result.html', context)

    else:
        return redirect('index')
