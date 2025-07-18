# Fundamental Concepts

To correctly interpret the results from `WeightUsageAnalyzer`, it's essential to understand the key metrics used.

---

## 1. Weight Importance

The analysis relies on a simple yet powerful metric to quantify the importance of each weight in the network.

#### Formula
The importance of a weight $w_{ij}$ (connecting neuron $i$ from the previous layer to neuron $j$ of the current layer) is defined as:

$$ \text{Importance}(w_{ij}) = |w_{ij}| \times \overline{a_j} $$

Where:
-   $|w_{ij}|$ is the **magnitude (absolute value) of the weight**. A weight with a large magnitude has a stronger impact on the output neuron.
-   $\overline{a_j}$ is the **average activation of the output neuron $j$** across the entire dataset. A high activation means the neuron is frequently "firing" and actively participates in the computation.
---

## 2. Entropy and "Effective Weights"

Entropy is a concept from information theory that measures the uncertainty or uniformity of a distribution.

#### Shannon Entropy
In our case, we apply it to the normalized distribution of weight importances.

$$ H = - \sum_{i} p_i \log_2(p_i) $$

Where $p_i$ is the normalized importance of the i-th weight (such that $\sum p_i = 1$).

-   **Low Entropy**: Indicates that a small subset of weights dominates and holds the majority of the total importance. The distribution is very "peaked."
-   **High Entropy**: Indicates that the importance is spread very uniformly across all weights.

#### Effective Weights
Entropy is useful but not very intuitive. We convert it into a more meaningful metric: the number of **effective weights**.

$$ N_{\text{eff}} = e^H $$

$N_{\text{eff}}$ represents the number of weights in an "ideal" model where all weights contribute equally, which would have the same entropy as our real model. It's a measure of the model's *used* complexity, as opposed to its structural complexity (the total number of weights).

---

## 3. FLOPs (Floating Point Operations)

**FLOPs** are a standard measure of the number of arithmetic operations (additions, multiplications) a model performs. It's a more reliable indicator of computational cost than just the parameter count.

-   **Inference FLOPs**: Correspond to one forward pass.
-   **Training FLOPs**: Are generally estimated to be 2-3 times the inference FLOPs, to account for the backward pass and gradient updates. Our tool uses a factor of 2 for the backward pass, leading to `2 * FLOPs_forward` per sample, then multiplied by the number of epochs and samples.

Reducing FLOPs is directly linked to:
-   **Faster inference**.
-   **Less energy-intensive training**.
-   A **reduced ecological footprint** for the model.

---
## 4. Importance Visualization

The `show()` function visually represents the network:
-   **Nodes**: The circles represent the neurons in each layer.
-   **Connections**: The lines between neurons represent the weights.
    -   The **thickness** of the line is proportional to the absolute magnitude of the weight.
    -   The **color** (red vs. gray) highlights the strongest weights (magnitude > 0.1 by default).

This visualization allows for the instant identification of "dead" neurons (few strong connections) and the preferred pathways that information takes through the network.

![A diagram generated by the `show()` function, depicting a network with lines of varying thicknesses and colors, illustrating well-defined neural pathways.](graphs/UseCaseTorch2.png)
