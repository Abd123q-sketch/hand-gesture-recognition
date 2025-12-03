from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam

num_classes = 6  
seq_length, img_height, img_width, channels = 21, 64, 64, 3

print("\n=== Construction du modèle CNN + LSTM ===")

model = Sequential([
    InputLayer(input_shape=(seq_length, img_height, img_width, channels)),

    # CNN appliqué à chaque frame
    TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Flatten()),

    # LSTM pour capturer la dimension temporelle
    LSTM(128, return_sequences=False),

    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
