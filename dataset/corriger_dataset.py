import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, InputLayer
from tensorflow.keras.utils import to_categorical

# Charger les données
X_train = np.load('train/X.npy')
y_train = np.load('train/y.npy')
X_val = np.load('val/X.npy')
y_val = np.load('val/y.npy')
X_test = np.load('test/X.npy')
y_test = np.load('test/y.npy')

num_classes = len(np.unique(y_train))

# Si besoin, one-hot encoding
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Reshape pour LSTM : (samples, timesteps, features)
# Ici on considère 1 "frame" par sample
X_train_lstm = X_train.reshape((X_train.shape[0], 1, 21*3))
X_val_lstm = X_val.reshape((X_val.shape[0], 1, 21*3))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, 21*3))

# Construction du modèle
model = Sequential([
    InputLayer(input_shape=(1, 21*3)),  # 1 frame, 63 features
    
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

# Entraînement
model.fit(
    X_train_lstm, y_train_cat,
    validation_data=(X_val_lstm, y_val_cat),
    epochs=30,
    batch_size=16
)

# Évaluation sur test
test_loss, test_acc = model.evaluate(X_test_lstm, y_test_cat)
print(f"\nTest Accuracy: {test_acc:.4f}")
