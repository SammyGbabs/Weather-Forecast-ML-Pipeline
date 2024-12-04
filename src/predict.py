import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_evaluate_model(model_path, X_test_Scaled, y_test):
    # Load the saved model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Make predictions on the test set
    y_pred_prob = model.predict(X_test_Scaled)

    # Convert probabilities to binary predictions (threshold > 0.5)
    y_pred_model = (y_pred_prob > 0.5).astype(int)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_model)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rain', 'No-Rain'], yticklabels=['Rain', 'No-Rain'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of Optimized Weather Forecast Model')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_model, target_names=['Rain', 'No-Rain']))
