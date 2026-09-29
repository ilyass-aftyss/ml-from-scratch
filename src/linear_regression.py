

import numpy as np 

class lineaireRegression : 

    # learning_rate --> controle la taille de modification apportées au modele !
    # Une epochs --> correspond a un passage complet sur les données d'entrainement --> le modele combien il regarde les données 
    def __init__(self,learning_rate = 0.01,epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None

        self.loss_history = []

    def prediction(self , X) :
        return X @ self.weights + self.bias

    # la fonction loss --> nous calculons une erreur pour savoir est ce que le modele est bonne 


    def compute_loss(self,y_true,y_pred) :
        n = len(y_true)

        return np.sum((y_true - y_pred) **2)/n


    # le coeur de ml --> le Gradient Descent --> 

    # w nouveau = w ancien - learning_rate * gradient


    # Apprends les meilleurs weights et bias possibles à partir de mes données X et y : 


    def fit(self,X,y) : 
        n_samples , n_features = X.shape

        self.weights = np.zeros((n_features,1))
        self.bias = 0 


        self.loss_history = []

        # la boucle d'apprentissage : 

        for epoch in range(self.epochs) : 

            y_pred = self.prediction(X)
            loss = self.compute_loss(y,y_pred)
            dw = (2 / n_samples) * X.T @ (y_pred - y)

            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            self.loss_history.append(loss)


            

