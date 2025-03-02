from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class CNNLSTMClassifier:
    def __init__(self, input_shape=(187, 1), num_classes=5, learning_rate=1e-4, batch_size=32, epochs=50):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        """Builds a simplified and regularized CNN + LSTM model."""
        model = keras.Sequential([
            # Simplified CNN layers with fewer filters and more dropout
            layers.Input(shape=self.input_shape),
            layers.Conv1D(32, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),

            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.4),

            # Only one LSTM layer with reduced units and dropout
            layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2),

            # Fully connected layer with L2 regularization
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.3),

            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        """Trains the CNN + LSTM model with early stopping and model checkpointing."""
        
        # Early stopping to avoid overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
    
        # Model checkpoint to save the best model
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            'saved_model.h5',               
            monitor='val_accuracy',            
            save_best_only=True,           
            mode='min',                    
            verbose=1                      
        )
    
        # Reduce learning rate if the model hits a plateau
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )    
        
        # Include all callbacks in the training process
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, model_checkpoint, lr_scheduler]
        )

    def evaluate(self, X_test, y_test):
        """Evaluates the model on the test set using the best saved model."""
        
        # Load the best model weights
        self.model = keras.models.load_model('saved_model.h5')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
    
        print(classification_report(y_test, y_pred_classes))
    
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()


    def predict(self, X):
        """
        Generates predictions using the trained model.
        
        Args:
            X (np.array): Input data for prediction.
        
        Returns:
            np.array: Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        self.model = keras.models.load_model('saved_model.h5')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model.predict(X, verbose=0)


    def plot_history(self):
        """Plots training and validation accuracy and loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig("Model Accuracy" , bbox_inches='tight', dpi=300)
        plt.show()



