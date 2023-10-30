import pymongo
import base64
import cv2
import numpy as np
import tensorflow as tf

# Conectar ao banco de dados
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["frutas"]
collection = db["imagens"]

# Carregar o modelo treinado
model = tf.keras.models.load_model('leaf_disease.h5')

# Ler uma nova imagem
new_image = cv2.imread("new_image.jpg")
new_image_resized = cv2.resize(new_image, (224, 224))
new_image_tensor = np.expand_dims(new_image_resized, axis=0) / 255.0

# Fazer a previsão da nova imagem
prediction = model.predict(new_image_tensor)
predicted_class = np.argmax(prediction)

# Recuperar os parâmetros do banco de dados
params = collection.find_one({"nome": "banana"})
params_pred = params["predicao"]

# Comparar a previsão da nova imagem com os parâmetros do banco de dados
if np.array_equal(prediction, params_pred):
    predicted_label = params["nome"]
else:
    predicted_label = "Desconhecido"

# Mostrar a imagem e a classificação
cv2.imshow("Nova imagem", new_image)
print("Classificação:", predicted_label)

cv2.waitKey(0)
cv2.destroyAllWindows()
