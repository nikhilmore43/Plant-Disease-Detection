import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model("best_model.keras")

# ==============================
# TEST DATA
# ==============================
test_dir = r"C:\Users\moren\Documents\plant_datatset\Test"

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())
print("Class order:", class_names)

# ==============================
# EVALUATE MODEL
# ==============================
loss, accuracy = model.evaluate(test_gen)
print("\n✅ Test Accuracy:", round(accuracy * 100, 2), "%")

# ==============================
# PREDICTIONS
# ==============================
pred_probs = model.predict(test_gen)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_gen.classes

# ==============================
# CLASSIFICATION REPORT
# ==============================
print("\n📊 Classification Report:\n")
print(classification_report(true_classes, pred_classes, target_names=class_names))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()