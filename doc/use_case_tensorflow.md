# Use Case: Analyzing a TensorFlow/Keras Model

This tutorial illustrates how to use `WeightUsageAnalyzer` on a simple model built with TensorFlow/Keras to evaluate its structure and efficiency.

---

### Analyzing a Keras Model

#### Code

```python
import core.weightusageanalyzer as wua
import numpy as np
import tensorflow as tf

# Dataset
np.random.seed(42)
X = np.random.rand(1000, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Keras Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,), name="input_layer"),
    tf.keras.layers.Dense(8, activation='relu', name="hidden_layer"), # Hidden layer with 8 neurons
    tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")
])

# Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=70, batch_size=32, verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

# Analysis with WeightUsageAnalyzer
wua.print_flops_report(model, nb_epochs=70, dataset=X)

importance_list = wua.compute_weight_importance(model, X)
for importance, weights, name in importance_list:
    report, norm_importance = wua.generate_report(importance, weights)
    print(f"\n📌 Report for layer: {name}")
    wua.print_report(report)
    wua.plot_importance_histogram(norm_importance, report["entropy"])

print("\n================================\n")
wua.show(model, X)
```

We can as we did in the use case of pytorch analyse the model and reduce it's size