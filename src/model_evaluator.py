import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

class ModelEvaluator:
    def __init__(self):
        """model değerlendirme sınıfı"""
        self.baseline = None
        self.auc_score = None
        self.lift_score = None

    def validate_predictions(self,
                           y_true: np.ndarray,
                           y_pred_proba: np.ndarray,
                           quiet: bool = False) -> dict:
        """
        model performansı
        """
        self.baseline = np.mean(y_true)

        y_pred = (y_pred_proba > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        #AUC 
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        self.auc_score = auc(fpr, tpr)

        self.lift_score = accuracy / self.baseline

        if not quiet:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {self.auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

        # Sonuçları hazırla
        metrics = {
            'baseline': self.baseline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': self.auc_score,
            'lift': self.lift_score,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn,
            'correct_predictions': tp + tn,
            'total_predictions': tp + tn + fp + fn
        }

        if not quiet:
            print(f"\nModel Değerlendirme Sonuçları:")
            print(f"Baseline (Ortalama Kazanma Oranı): {self.baseline:.3f}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {precision:.3f}")
            print(f"AUC: {self.auc_score:.3f}")
            print(f"Lift: {self.lift_score:.3f}")
            print(f"\nTahmin Detayları:")
            print(f"Doğru Pozitif (TP): {tp}")
            print(f"Yanlış Pozitif (FP): {fp}")
            print(f"Doğru Negatif (TN): {tn}")
            print(f"Yanlış Negatif (FN): {fn}")
            print(f"\nToplam Doğru Tahmin: {tp + tn}")
            print(f"Toplam Tahmin: {tp + tn + fp + fn}")

        return metrics

def evaluate_model_performance(model, X_test, y_test):
    """
    test verisi üzerinde model performansını değerlendirir
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator()

    metrics = evaluator.validate_predictions(y_test, y_pred_proba)

    return metrics