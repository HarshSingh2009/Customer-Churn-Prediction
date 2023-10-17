import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pickle
import sklearn


class PredictionPipeline():
    def __init__(self) -> None:
        pass

    def predict(self, inputs):
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        model = load_model('DL model\customer_churn_prediction_model.h5')
        input_tensor = scaler.transform(tf.expand_dims(tf.constant(inputs), axis=0))
        if input_tensor.shape == (1, 12):
            y_probs = model.predict(input_tensor)
            y_preds = tf.round(y_probs)
            if y_preds == 0:
                y_probs = 1 - y_probs[0][0]
            else:
                y_probs = y_probs[0][0]
        return y_probs, tf.round(y_probs)