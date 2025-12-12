from tensorflow.keras.models import load_model

model_path =  "C:\\Users\\Hp\\Desktop\\mon_projet_deeplearning\\src\\hand_gesture_cnn_lstm.h5"

model = load_model(model_path)

print("\n=== SUMMARY ===")
model.summary()

print("\n=== INPUT SHAPE ===")
print(model.input_shape)
