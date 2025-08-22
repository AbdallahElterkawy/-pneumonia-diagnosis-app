from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # لأن التصنيف ثنائي
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

model.save("pneumonia_model.h5")

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("pneumonia_model.h5")

st.title("تشخيص الالتهاب الرئوي بالأشعة")
uploaded_file = st.file_uploader("ارفع صورة أشعة", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="الصورة المدخلة", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)[0][0]
    result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    st.write(f"التشخيص: **{result}**")

    st.write(f"التشخيص: **{result}**")

if result == "PNEUMONIA":
    st.error("⚠️ الحالة: التهاب رئوي")
else:
    st.success("✅ الحالة: طبيعية")

@st.cache_resource
def load_model_once():
    return load_model('pneumonia_model.h5')

model = load_model_once()

