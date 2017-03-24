from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras import optimizers

class NeuralNetworkClassifier():                                                  
    def __init__(self, dim_features, nb_attributes=85, batch_size=10, nb_epoch=5):
        self.model = Sequential()
        self.model.add(Dense(1000, input_dim=dim_features))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(nb_attributes))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer="adam", loss='binary_crossentropy')
        
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def fit(self, X, y):
        self.model.fit(X, y, self.batch_size, self.nb_epoch)

    def predict(self, X):
        return self.model.prediczt(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class NeuralNetworkClassifier2():                                                  
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
