from tensorflow.keras import layers, models
import numpy as np
import networkx as nx

class HGNN:
    def __init__(self, num_nodes, num_features, num_classes):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.num_nodes, self.num_features))
        h = layers.Dense(64, activation='relu')(inputs)
        h = layers.Dense(32, activation='relu')(h)
        outputs = layers.Dense(self.num_classes, activation='softmax')(h)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_hypergraph(self, stocks, industries, market_caps):
        self.hypergraph = nx.Graph()
        for stock, industry, market_cap in zip(stocks, industries, market_caps):
            self.hypergraph.add_node(stock, industry=industry, market_cap=market_cap)
            # Create hyper edges based on industry and market cap
            self.hypergraph.add_edge(industry, stock)
            self.hypergraph.add_edge(market_cap, stock)

    def train(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def get_probability_distribution(self, X):
        predictions = self.predict(X)
        return np.max(predictions, axis=1)  # Return the max probability for each stock