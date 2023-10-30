import pymongo
import base64
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Conectar ao banco de dados
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["frutas"]
collection = db["imagens"]

# Especificar os caminhos das pastas com as imagens
healthy_leaves_folder = "D:\\fotos_do_tcc\\saudavel"
diseased_leaves_folder = "D:\\fotos_do_tcc\\praga"

# Carregar as imagens de folhas saudáveis
healthy_images = []
for filename in os.listdir(healthy_leaves_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(healthy_leaves_folder, filename))
        img_resized = cv2.resize(img, (224, 224))
        healthy_images.append(img_resized)

# Carregar as imagens de folhas doentes
diseased_images = []
for filename in os.listdir(diseased_leaves_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(diseased_leaves_folder, filename))
        img_resized = cv2.resize(img, (224, 224))
        diseased_images.append(img_resized)

# Converter as imagens para tensores
healthy_images_tensor = np.array(healthy_images)
diseased_images_tensor = np.array(diseased_images)

# Criar rótulos para as imagens
healthy_labels = np.zeros(len(healthy_images))
diseased_labels = np.ones(len(diseased_images))

# Concatenar as imagens e rótulos
images = np.concatenate((healthy_images_tensor, diseased_images_tensor), axis=0)
labels = np.concatenate((healthy_labels, diseased_labels), axis=0)

# Dividir os dados em conjuntos de treinamento, teste e validação
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Normalizar as imagens
train_images = train_images / 255.0
test_images = test_images / 255.0
val_images = val_images / 255.0

# Definir o número de classes
num_classes = 2

# Codificar os rótulos
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
val_labels_encoded = label_encoder.transform(val_labels)

# Carregar o modelo de reconhecimento de folhas
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Congelar as camadas do modelo base
base_model.trainable = False

# Adicionar as camadas personalizadas
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Treinar o modelo
model.fit(train_images, train_labels_encoded, epochs=10, validation_data=(val_images, val_labels_encoded))

# Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(test_images, test_labels_encoded)
print("Erro do teste:", loss)
print("Precisão do Teste:", accuracy)

# Salvar o modelo treinado
model.save("leaf_disease.h5")

# Mostrar uma imagem de teste
test_image = test_images[0]
prediction = model.predict(np.expand_dims(test_image, axis=0))
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
print("Rótulo da Predição:", predicted_label[0])
