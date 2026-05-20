import numpy as np
import optuna
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import os

# Modified BSSF class that supports multiple genuine matches via probe and gallery labels.
class GalleryScoreBSSF:
    def __init__(self, n_matchers=3, n_trials=100, random_state=42):
        self.n_matchers = n_matchers
        self.n_trials = n_trials
        self.random_state = random_state
        self.fusion_weights = None
        
    def get_eer(self, genuine_scores, impostor_scores):
        n_genuine = len(genuine_scores)
        n_impostor = len(impostor_scores)
        y_true = np.concatenate([np.ones(n_genuine), np.zeros(n_impostor)])
        y_score = np.concatenate([genuine_scores, impostor_scores])
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute(fpr - fnr))]
        return eer
    
    def objective(self, trial, score_matrices, probe_labels, gallery_labels):
        weights = np.array([trial.suggest_float(f'w{i}', 0.0, 1.0) for i in range(self.n_matchers)])
        weights = weights / np.sum(weights)
        
        fused_scores = np.zeros_like(score_matrices[0])
        for i in range(self.n_matchers):
            fused_scores += weights[i] * score_matrices[i]
        
        genuine_scores = []
        impostor_scores = []
        n_probes = fused_scores.shape[0]
        for probe_idx in range(n_probes):
            genuine_mask = (gallery_labels == probe_labels[probe_idx])
            # If no genuine match is found, skip the probe
            if not genuine_mask.any():
                continue
            genuine_scores.extend(fused_scores[probe_idx, genuine_mask].tolist())
            impostor_scores.extend(fused_scores[probe_idx, ~genuine_mask].tolist())
        
        eer = self.get_eer(np.array(genuine_scores), np.array(impostor_scores))
        return eer

    def fit(self, score_matrices, probe_labels, gallery_labels):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial, score_matrices, probe_labels, gallery_labels),
            n_trials=self.n_trials
        )
        best_params = study.best_params
        self.fusion_weights = np.array([best_params[f'w{i}'] for i in range(self.n_matchers)])
        self.fusion_weights = self.fusion_weights / np.sum(self.fusion_weights)
        return self

    def predict(self, score_matrices):
        if self.fusion_weights is None:
            raise ValueError("Model not fitted. Call 'fit' first.")
        fused_scores = np.zeros_like(score_matrices[0])
        for i in range(self.n_matchers):
            fused_scores += self.fusion_weights[i] * score_matrices[i]
        return fused_scores

    def evaluate(self, score_matrices, probe_labels, gallery_labels):
        fused_scores = self.predict(score_matrices)
        individual_eers = []
        for i in range(self.n_matchers):
            genuine_scores_i = []
            impostor_scores_i = []
            n_probes = score_matrices[i].shape[0]
            for probe_idx in range(n_probes):
                genuine_mask = (gallery_labels == probe_labels[probe_idx])
                if not genuine_mask.any():
                    continue
                genuine_scores_i.extend(score_matrices[i][probe_idx, genuine_mask].tolist())
                impostor_scores_i.extend(score_matrices[i][probe_idx, ~genuine_mask].tolist())
            eer_i = self.get_eer(np.array(genuine_scores_i), np.array(impostor_scores_i))
            individual_eers.append(eer_i)
        
        genuine_scores = []
        impostor_scores = []
        n_probes = fused_scores.shape[0]
        for probe_idx in range(n_probes):
            genuine_mask = (gallery_labels == probe_labels[probe_idx])
            if not genuine_mask.any():
                continue
            genuine_scores.extend(fused_scores[probe_idx, genuine_mask].tolist())
            impostor_scores.extend(fused_scores[probe_idx, ~genuine_mask].tolist())
        eer = self.get_eer(np.array(genuine_scores), np.array(impostor_scores))
        
        max_rank = min(20, fused_scores.shape[1])
        cmc = np.zeros(max_rank)
        for probe_idx in range(n_probes):
            sorted_indices = np.argsort(-fused_scores[probe_idx])
            genuine_indices = np.where(gallery_labels == probe_labels[probe_idx])[0]
            genuine_ranks = np.where(np.isin(sorted_indices, genuine_indices))[0]
            if genuine_ranks.size > 0:
                min_rank = genuine_ranks.min()
                if min_rank < max_rank:
                    cmc[min_rank:] += 1
        cmc = cmc / n_probes
        
        return {
            'eer': eer,
            'individual_eers': individual_eers,
            'fused_scores': fused_scores,
            'cmc': cmc
        }

    def save(self, filepath):
        if self.fusion_weights is None:
            raise ValueError("Model not fitted. Call 'fit' first.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        model_params = {
            'n_matchers': self.n_matchers,
            'n_trials': self.n_trials,
            'random_state': self.random_state,
            'fusion_weights': self.fusion_weights.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(model_params, f, indent=4)
        print(f"Model parameters saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            model_params = json.load(f)
        model = cls(
            n_matchers=model_params['n_matchers'],
            n_trials=model_params['n_trials'],
            random_state=model_params['random_state']
        )
        model.fusion_weights = np.array(model_params['fusion_weights'])
        print(model.fusion_weights)
        print(f"Model parameters loaded from {filepath}")
        return model

# --- Demo Example ---

def demo_train_model():
    np.random.seed(42)
    
    # Parameters
    n_probes = 100       # Number of probe samples
    n_gallery = 200      # Number of gallery samples
    n_matchers = 3       # Number of score matrices (matchers)
    
    # Generate synthetic probe and gallery labels (e.g., IDs from 0 to 49)
    probe_labels = np.random.randint(0, 50, size=n_probes)
    gallery_labels = np.random.randint(0, 50, size=n_gallery)
    
    # Create synthetic score matrices for each matcher
    score_matrices = []
    for matcher_idx in range(n_matchers):
        score_matrix = np.zeros((n_probes, n_gallery))
        # Different distributions for each matcher
        if matcher_idx == 0:
            genuine_mean, genuine_std = 0.8, 0.1
            impostor_mean, impostor_std = 0.2, 0.1
        elif matcher_idx == 1:
            genuine_mean, genuine_std = 0.7, 0.15
            impostor_mean, impostor_std = 0.3, 0.15
        else:
            genuine_mean, genuine_std = 0.6, 0.2
            impostor_mean, impostor_std = 0.4, 0.2
        
        for i in range(n_probes):
            # Generate impostor scores for the probe
            score_matrix[i, :] = np.random.normal(impostor_mean, impostor_std, n_gallery)
            # For genuine matches, find all gallery entries with the same label as the probe
            genuine_indices = np.where(gallery_labels == probe_labels[i])[0]
            if genuine_indices.size > 0:
                for idx in genuine_indices:
                    score_matrix[i, idx] = np.random.normal(genuine_mean, genuine_std)
        score_matrix = np.clip(score_matrix, 0, 1)
        score_matrices.append(score_matrix)
    
    # Split probes into train and test sets (gallery remains the same)
    indices = np.arange(n_probes)
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
    
    train_matrices = [mat[train_indices] for mat in score_matrices]
    test_matrices = [mat[test_indices] for mat in score_matrices]
    train_probe_labels = probe_labels[train_indices]
    test_probe_labels = probe_labels[test_indices]
    
    # Create and train the BSSF model using training data and labels
    bssf = GalleryScoreBSSF(n_matchers=n_matchers, n_trials=50)
    bssf.fit(train_matrices, train_probe_labels, gallery_labels)
    
    # Save the model
    model_path = "bssf_model.json"
    bssf.save(model_path)
    bssf = GalleryScoreBSSF.load(model_path)
    
    # Evaluate the model on the test set
    results = bssf.evaluate(test_matrices, test_probe_labels, gallery_labels)
    
    print("Optimized Fusion Weights:", bssf.fusion_weights)
    print("Fused EER: {:.4f}".format(results['eer']))
    print("Individual Matchers EERs:")
    for i, eer in enumerate(results['individual_eers']):
        print(f"  Matcher {i+1}: {eer:.4f}")
    print("CMC (Rank-1 to Rank-5):", results['cmc'][:5])
    
if __name__ == "__main__":
    demo_train_model()
    
    from BSSF import GalleryScoreBSSF
    bssf = GalleryScoreBSSF(n_matchers=3, n_trials=50)
    bssf.fit([res['concat_scores'][..., 0].cpu().numpy(), res['concat_scores'][..., 1].cpu().numpy(), res['concat_scores'][..., 2].cpu().numpy()], 
             q_pids.cpu().numpy(), center_pids.cpu().numpy())
    # Save the model
    model_path = "bssf_model_ltcc.json"
    bssf.save(model_path)
    bssf = GalleryScoreBSSF.load(model_path)
