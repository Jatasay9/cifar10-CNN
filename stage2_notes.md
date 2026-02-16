## This is when i trained for 50000 images --->
## Experiment 1: Baseline (1 Conv Layer)

**Architecture:**
- Conv2d(3, 16, 3, padding=1)
- MaxPool2d(2, 2)
- Linear(16*16*16, 10)

**Results:**
- Train Accuracy: ~57.05%
- Test Accuracy: ~55.97%

**Observation:** The model successfully learns basic patterns but has limited representational capacity due to shallow depth. Train and test accuracy are close, indicating minimal overfitting.

---

## Experiment 2: 2 Conv Layers

**Architecture:**
- Conv2d(3, 16, 3, padding=1)
- Conv2d(16, 32, 3, padding=1)
- MaxPool
- Linear(32*8*8, 10)

**Results:**
- Train Accuracy: ~60.60%
- Test Accuracy: ~60.81%

**Observation:** Increasing depth improved both training and test accuracy. This suggests better feature extraction capability from deeper convolutional layers without significant overfitting.

---

## Experiment 3: Data Augmentation

**Transforms:**
- RandomHorizontalFlip
- RandomCrop(32, padding=4)

**Results:**
- Train Accuracy: ~51.67%
- Test Accuracy: ~58.23%

**Observation:** Training accuracy decreased due to harder augmented samples, but test accuracy remained relatively stable. This suggests improved robustness, though longer training may be required for full benefit.

---

## Experiment 4: Dropout

**Added:**
- Dropout(0.5)

**Results:**
- Train Accuracy: ~46.87%
- Test Accuracy: ~53.13%

**Observation:** Performance decreased, indicating underfitting. Combined augmentation and dropout introduced strong regularization, and 3 epochs were insufficient for convergence.

---

## Final Model Decision (Before Extended Training)

The 2-convolution architecture without dropout showed the best balance between capacity and generalization. Augmentation appears promising but requires longer training to fully evaluate its impact.

## Final Model (2 Conv + Augmentation, 10 Epochs)

Results:
- Train Accuracy: ~60.06%
- Test Accuracy: ~64.32%

Observation:
Training with augmentation required more epochs but improved generalization. The final model achieved the highest test accuracy without signs of overfitting.




