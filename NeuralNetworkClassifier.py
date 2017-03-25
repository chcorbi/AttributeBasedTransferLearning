from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras import optimizers

class NeuralNetworkClassifier():                                                  
    def __init__(self, dim_features, nb_attributes=85, batch_size=10, nb_epoch=5):
        inputs = Input(shape=(dim_features,))
        x = Dense(1000, activation='relu')(inputs)
        x = Dropout(0.3)(x)

        predictions = []
        for p in range(nb_attributes):
            predictions.append(Dense(1, activation='sigmoid')(x))

        self.model = Model(inputs, predictions)

        self.model.compile(optimizer="adam", loss=['binary_crossentropy'] * nb_attributes)
        
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def fit(self, X, y):
        self.model.fit(X, list(y.T), self.batch_size, self.nb_epoch)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return 0
