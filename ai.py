# %%
import json
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# %%
with open('ai_data.json', 'r') as f:
    ai_data = json.load(f)
ai_data = np.array(ai_data)
ai_data.shape

# %%
ai_data[0]
# %%
x = ai_data[:, :-1]
y = ai_data[:, -1]
y = to_categorical(y)
y

# %%
model = Sequential()
model.add(Dense(32, input_shape=(47,), activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', \
              optimizer = 'Adam', metrics = ['accuracy'])
model.summary()

# %%
history = model.fit(x, y, epochs=1000)

# %%

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()
# %%

model.save('ai_model')

# %%
new_model = load_model("ai_model")

# %%
predict = new_model.predict(np.array([x[0]]))
predict

# %%
"RPS"[np.argmax(predict)]
# %%
