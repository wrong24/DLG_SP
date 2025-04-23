from tensorflow.keras.layers import Layer, MultiHeadAttention, Dropout, Dense
import tensorflow as tf
import numpy as np

class TemporalGraphAttentionLayer(Layer):
    def __init__(self, num_heads, output_dim, dropout_rate=0.1):
        super(TemporalGraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=output_dim)
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(output_dim)

    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout(attention_output)
        return self.dense(attention_output)

class TGAT(tf.keras.Model):
    def __init__(self, num_heads, output_dim, dropout_rate=0.1):
        super(TGAT, self).__init__()
        self.attention_layer = TemporalGraphAttentionLayer(num_heads, output_dim, dropout_rate)

    def call(self, inputs):
        return self.attention_layer(inputs)

    def train_model(self, x_train, y_train, epochs=100, batch_size=32):
        self.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.predict(x)