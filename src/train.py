# train_cnn_lstm.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Charger le dataset
X_train = np.load('../dataset/X_train.npy')
y_train = np.load('../dataset/y_train.npy')
X_val   = np.load('../dataset/X_val.npy')
y_val   = np.load('../dataset/y_val.npy')
X_test  = np.load('../dataset/X_test.npy')
y_test  = np.load('../dataset/y_test.npy')

# Vérifier les formes
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# S’assurer que les labels sont en one-hot
num_classes = y_train.shape[-1]

# Construction du modèle CNN + LSTM
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])),
    
    TimeDistributed(Conv2D(32, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Flatten()),

    LSTM(128, return_sequences=False),
    
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Sauvegarde automatique du meilleur modèle
checkpoint = ModelCheckpoint('hand_gesture_cnn_lstm.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=8,
    callbacks=[checkpoint]
)

# Évaluation sur le jeu de test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
