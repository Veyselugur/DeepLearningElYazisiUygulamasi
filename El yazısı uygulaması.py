"""Veri seti keras ile birlikte çağırıldı."""
from keras.datasets import mnist

"""Veri setinin doğruluğu için incelendi."""
(train_img, train_labels) = mnist.load_data()[0]
(test_img, test_labels) = mnist.load_data()[1]
print("Shape of Train Images: ",train_img.shape)
print("Shape of Test Images: ",test_img.shape)

"""Veri seti içerisindeki resimler farklı yöntemler ile incelendi."""
import matplotlib.pyplot as plt
plt.imshow(train_img[5], cmap='gray_r')
plt.show()

for row in train_img[5]:
    for i in row:
        print("%3s "%i, end='')
    print()

import numpy as np

n, k, figsize = 10, 10, (10, 10)
fig, ax = plt.subplots(n, k, figsize=figsize)
for i in range(n):
    for j in range(k):
        ax[i,j].imshow(train_img[np.random.randint(train_img.shape[0])], cmap='gray_r')
        ax[i,j].axis('off')

plt.show()

"""Eğitime başlamadan önce resimlerimizdeki
 tüm değerleri [0,1] aralığına sıkıştırıyoruz ve etiketlerinizi de kategorik olarak kodluyoruz."""
train_img = train_img.reshape((60000,28*28))
train_img = train_img.astype("float32")/255

test_img = test_img.reshape((10000,28*28))
test_img = test_img.astype("float32")/255
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

"""Veriler hazırlandıktan sonra ağ oluşturmak için 2 katmanlı bir yapı oluşturuldu."""
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(512,activation="relu", input_shape=(28*28,)))
model.add(layers.Dense(10,activation="softmax"))
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(train_img,
                    train_labels,
                    epochs=10,
                    batch_size=128)
history_dict = history.history
print("Keys: ",history_dict.keys())

import matplotlib.pyplot as plt

epochs = range(1,11)
loss = history_dict['loss']
accuracy = history_dict['accuracy']

"""Eğitim işlemi yapılıp değerler incelendi."""
plt.plot(epochs,loss)
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Eğitim Kaybı")
plt.show()

plt.plot(epochs,accuracy)
plt.title("Accu")
plt.xlabel("Epochs")
plt.ylabel("Eğitim Başarımı")
plt.show()
test_loss, test_acc = model.evaluate(test_img,test_labels)
print("Test Loss: ",test_loss)
print("Test Accuracy: ",test_acc)