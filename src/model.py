import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Initialize the sequential model
def build_and_train_model(X_train_Scaled, y_train, X_test_Scaled, y_test, save_path):
    model = Sequential()

    # Input layer and first hidden layer with Dropout
    model.add(Dense(128, input_dim=X_train_Scaled.shape[1], kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.4))

    # Second hidden layer with Dropout
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.4))

    # Third hidden layer with Dropout
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.4))

    # Output layer (binary classification with sigmoid activation)
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # Print model summary
    model.summary()

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early stopping callback (optional, to avoid overfitting)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Fit the model on the training data
    history = model.fit(X_train_Scaled, y_train, epochs=100, validation_split=0.2,
                        callbacks=[early_stop], batch_size=32)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test_Scaled, y_test)
    print(f"Test Loss: {test_loss}\n Test Accuracy: {test_accuracy}")

    # Save the trained model
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model, history
