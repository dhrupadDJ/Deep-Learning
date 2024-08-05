# main.py
from data_preprocessing.load_data import load_data
from data_preprocessing.preprocess_data import preprocess_data
from models.build_model import build_model
from evaluation.evaluate_model import evaluate_model
import tensorflow as tf

# Load and preprocess data
file_path = r"C:\Users\jaisw\Desktop\Data Science Final Project\Deep learning with tensor flow\Deep learning with tensor flow\utils\employee_attrition.csv"
df = load_data(file_path)
x_train, x_test, y_train, y_test = preprocess_data(df)

# Build and train model
model = build_model(input_shape=x_train.shape[1])
history = model.fit(x_train, y_train, epochs=50, verbose=0)

# Evaluate the model
y_preds = tf.round(model.predict(x_test))
evaluate_model(y_test, y_preds)

# Plot loss curves
pd.DataFrame(history.history).plot()
plt.title("Model training curves")
plt.show()
