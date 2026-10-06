import joblib
import matplotlib.pyplot as plt

# Load training history
history = joblib.load(
    "models/neural_training_history.joblib"
)

# Create figure
plt.figure(figsize=(10, 6))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Neural Network Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/training_validation_loss.png",
    dpi=300
)

plt.show()

print("Training graph saved to:")
print("models/training_validation_loss.png")