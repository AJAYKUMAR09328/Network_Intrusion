import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config
from data_loader import get_data_loaders
from models.cnn import CNNClassifier
from models.transformer import IntrusionTransformer
from models.gnn import TabularGNN

def evaluate_models():
    
    print("⏳ Loading Test Data...")

    # Get only test loader
    _, _, test_loader, input_dim = get_data_loaders()

    # Model dictionary
    models_dict = {
        'CNN': (
            CNNClassifier(
                input_dim=input_dim,
                num_classes=config.NUM_CLASSES
            ),
            "CNN_Model.pth"
        ),

        'Transformer': (
            IntrusionTransformer(
                input_dim=input_dim,
                num_classes=config.NUM_CLASSES
            ),
            "Transformer_Model.pth"
        ),

        'GNN': (
            TabularGNN(
                input_dim=input_dim,
                hidden_dim=config.HIDDEN_DIM,  # IMPORTANT FIX
                num_classes=config.NUM_CLASSES
            ),
            "GNN_Model.pth"
        )
    }

    class_names = config.CLASS_NAMES
    results = []

    print("\n--- STARTING GLOBAL EVALUATION ---")
    
    for name, (model, weight_file) in models_dict.items():

        if not os.path.exists(weight_file):
            print(f"⚠️ WARNING: '{weight_file}' not found. Skipping {name}...")
            continue

        print(f"\nEvaluating {name}...")
        model.load_state_dict(torch.load(weight_file, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()

        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in test_loader:

                X = X.to(config.DEVICE)
                y = y.to(config.DEVICE)

                outputs = model(X)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # ---------------- METRICS ----------------
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        # ---------------- CONFUSION MATRIX ----------------
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.title(f'{name} Confusion Matrix\nAccuracy: {acc:.2%}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        plt.show()

    # ---------------- FINAL COMPARISON ----------------
    if results:

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 40)
        print("FINAL MODEL COMPARISON")
        print("=" * 40)
        print(results_df.to_string(index=False))

        # Bar chart comparison in long format
        df_melted = results_df.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        )

        plt.figure(figsize=(10, 6))

        chart = sns.barplot(
            data=df_melted,
            x="Model",
            y="Score",
            hue="Metric"
        )

        plt.ylim(0, 1.05)

        plt.title('Model Performance Comparison')

        for container in chart.containers:
            chart.bar_label(container, fmt='%.2f', padding=3)

        plt.tight_layout()
        plt.show()

    else:
        print("❌ No models evaluated. Check weight files.")


if __name__ == "__main__":
    evaluate_models()