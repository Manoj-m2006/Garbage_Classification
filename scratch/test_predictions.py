import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import random
from pathlib import Path
from src.predict import predict_image, CLASSES

print("========================================")
print("PREDICTION DIAGNOSTIC TEST (10 SAMPLES PER CLASS)")
print("========================================")

correct = 0
total = 0

for cls in CLASSES:
    imgs = list(Path(f"dataset/dataset-resized/{cls}").glob("*.jpg"))
    samples = random.sample(imgs, min(10, len(imgs)))
    cls_correct = 0
    for s in samples:
        res = predict_image(str(s))
        pred = res['predicted_class']
        conf = res['confidence']
        is_match = (pred == cls)
        if is_match:
            cls_correct += 1
            correct += 1
        total += 1
        print(f"True: {cls:<10} | Pred: {pred:<10} | Match: {'YES' if is_match else 'NO ':<3} | Conf: {conf}%")
    print(f"--> Class {cls} Accuracy: {cls_correct}/{len(samples)} ({cls_correct/len(samples)*100:.1f}%)\n")

print(f"Overall Test Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
