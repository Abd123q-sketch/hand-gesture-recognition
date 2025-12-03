from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer

num_classes = 5  

print("\n=== Construction du modèle LSTM adapté ===")

model = Sequential([
    InputLayer(shape=(1, 21*3)),  # 1 "frame", 63 features
    
    LSTM(64, return_sequences=False),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
