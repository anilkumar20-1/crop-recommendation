import os
import io
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


class DiseasePredictor:
    def __init__(self, dataset_dir: str = 'Dataset', image_size: Tuple[int, int] = (128, 128)):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.model: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_names: List[str] = []

    def scan_classes(self) -> List[str]:
        if not os.path.isdir(self.dataset_dir):
            return []
        classes = [d for d in os.listdir(self.dataset_dir)
                   if os.path.isdir(os.path.join(self.dataset_dir, d))]
        classes.sort()
        self.class_names = classes
        return classes

    def _load_image_paths(self) -> List[Tuple[str, str]]:
        image_paths: List[Tuple[str, str]] = []
        if not os.path.isdir(self.dataset_dir):
            return image_paths
        for class_name in self.scan_classes():
            class_dir = os.path.join(self.dataset_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append((os.path.join(class_dir, fname), class_name))
        return image_paths

    def _extract_features_from_image(self, image: Image.Image) -> np.ndarray:
        image_rgb = image.convert('RGB').resize(self.image_size)
        image_np = np.asarray(image_rgb, dtype=np.float32) / 255.0

        # Convert to grayscale using luminance
        gray = (0.2989 * image_np[:, :, 0] + 0.5870 * image_np[:, :, 1] + 0.1140 * image_np[:, :, 2]).astype(np.float32)

        # Compute simple gradients (Sobel-like approximation)
        # Horizontal and vertical gradients using central differences
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
        magnitude = np.sqrt(gx * gx + gy * gy) + 1e-8
        orientation = (np.arctan2(gy, gx) + np.pi) % np.pi  # [0, pi)

        # Global orientation histogram (9 bins)
        num_bins = 9
        bin_edges = np.linspace(0.0, np.pi, num_bins + 1, dtype=np.float32)
        orient_hist = np.zeros(num_bins, dtype=np.float32)
        inds = np.digitize(orientation.ravel(), bin_edges) - 1
        inds = np.clip(inds, 0, num_bins - 1)
        # Accumulate magnitudes per bin
        for b in range(num_bins):
            orient_hist[b] = float(magnitude.ravel()[inds == b].sum())
        # Normalize orientation histogram
        orient_sum = float(orient_hist.sum())
        if orient_sum > 0:
            orient_hist /= orient_sum

        # Grayscale intensity histogram (16 bins)
        gray_hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
        gray_hist = gray_hist.astype(np.float32)

        # RGB color histogram (16 bins per channel)
        hist_r, _ = np.histogram(image_np[:, :, 0], bins=16, range=(0.0, 1.0), density=True)
        hist_g, _ = np.histogram(image_np[:, :, 1], bins=16, range=(0.0, 1.0), density=True)
        hist_b, _ = np.histogram(image_np[:, :, 2], bins=16, range=(0.0, 1.0), density=True)
        color_hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)

        # Downsampled grayscale patch (16x16) for coarse texture/shape
        small_size = (16, 16)
        small_gray = np.asarray(image_rgb.convert('L').resize(small_size), dtype=np.float32) / 255.0
        small_gray_flat = small_gray.flatten()

        features = np.concatenate([orient_hist, gray_hist, color_hist, small_gray_flat])
        return features.astype(np.float32)

    def _extract_features_from_path(self, image_path: str) -> Optional[np.ndarray]:
        try:
            with Image.open(image_path) as img:
                return self._extract_features_from_image(img)
        except Exception:
            return None

    def load_dataset(self, per_class_limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        paths_and_labels = self._load_image_paths()
        if not paths_and_labels:
            return np.array([]), np.array([]), []

        # Group by class and optionally limit
        class_to_items: Dict[str, List[str]] = {}
        for path, label in paths_and_labels:
            class_to_items.setdefault(label, []).append(path)

        X_list: List[np.ndarray] = []
        y_list: List[str] = []
        for label, items in class_to_items.items():
            selected = items
            if per_class_limit is not None and per_class_limit > 0:
                selected = items[:per_class_limit]
            for path in selected:
                feats = self._extract_features_from_path(path)
                if feats is not None:
                    X_list.append(feats)
                    y_list.append(label)

        if not X_list:
            return np.array([]), np.array([]), []

        X = np.vstack(X_list)
        y = np.array(y_list)
        self.class_names = sorted(list(set(y_list)))
        return X, y, self.class_names

    def train(self, per_class_limit: Optional[int] = 200, test_size: float = 0.1, random_state: int = 42) -> Dict:
        X, y, classes = self.load_dataset(per_class_limit=per_class_limit)
        if X.size == 0:
            raise ValueError("No training data found in dataset directory")

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        clf = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('logreg', LogisticRegression(max_iter=1000))
        ])

        clf.fit(X_train, y_train)
        self.model = clf

        accuracy = float(clf.score(X_test, y_test))
        return {
            'num_samples': int(X.shape[0]),
            'num_features': int(X.shape[1]),
            'num_classes': int(len(classes)),
            'accuracy': accuracy
        }

    def save_model(self, model_path: str) -> bool:
        if self.model is None or self.label_encoder is None:
            return False
        to_save = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'image_size': self.image_size
        }
        joblib.dump(to_save, model_path)
        return True

    def load_model(self, model_path: str) -> bool:
        if not os.path.exists(model_path):
            return False
        data = joblib.load(model_path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.class_names = data.get('class_names', [])
        self.image_size = tuple(data.get('image_size', self.image_size))
        return True

    def predict_image(self, image_bytes: bytes) -> Tuple[Optional[str], Optional[float]]:
        if self.model is None or self.label_encoder is None:
            return None, None
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                feats = self._extract_features_from_image(img)
            probs = self._predict_proba(feats.reshape(1, -1))
            if probs is None:
                return None, None
            pred_idx = int(np.argmax(probs[0]))
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[0][pred_idx])
            return pred_label, confidence
        except Exception:
            return None, None

    def _predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        # LogisticRegression supports predict_proba
        try:
            return self.model.predict_proba(X)
        except Exception:
            return None


