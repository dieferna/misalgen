import hashlib
import json
import os
import matplotlib.pyplot as plt
from statistics import mean
import torch
import pickle
import torch.nn as nn
from einops import einsum
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import yaml
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


from utils.classifiers import ContrastiveProbe, LinearProbe, BilinearProbe, PolyProbe, EEMLP, CenteredLogisticRegression
from utils.train_probes import train 

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency

from sklearn.base import clone
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def _get_sklearn_scores(clf, X):
    try:
        probs = clf.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
    except Exception:
        pass

    try:
        df = clf.decision_function(X)
        if df.ndim == 1:
            return df
        else:
            if df.shape[1] >= 2:
                return df[:, 1]
            else:
                return df.ravel()
    except Exception:
        pass

    return clf.predict(X)


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = {i: "40GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "160GiB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory,
        dtype=torch.bfloat16,
        output_hidden_states=True,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.eval()
    return model, tokenizer
   

def load_data_final(deceptive_path, honest_path):
    base_dir = os.path.dirname(os.path.dirname(__file__)) 
    if deceptive_path == honest_path:
        files = [deceptive_path]
    else:
        files = [deceptive_path, honest_path]

    deceptive_data = []
    honest_data = []

    for path in files:
        if not os.path.exists(path):
            print(f"Warning: file not found: {path}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {path}")
                    continue

                if "label" not in entry:
                    print(f"Missing label in {path}, skipping entry.")
                    continue

                if entry["label"]:
                    honest_data.append(entry)
                else:
                    deceptive_data.append(entry)

    print(f"Loaded {len(deceptive_data)} deceptive and {len(honest_data)} honest entries.")
    return deceptive_data, honest_data


def compute_stats(deceptive_data, honest_data, cfg, output_dir="output_plots", file_name="length_distributions.png"):
    downsample_rate = cfg["data"].get("downsample_rate", 1)
    if downsample_rate > 1:
        deceptive_sampled = deceptive_data[::downsample_rate]
    else:
        deceptive_sampled = deceptive_data

    print(f"After downsampling: {len(deceptive_sampled)} deceptive, {len(honest_data)} honest.")

    all_data = deceptive_sampled + honest_data
    print(f"Total combined: {len(all_data)} entries.")

    lengths = {0: {'analysis': [], 'final': []},
               1: {'analysis': [], 'final': []}}

    for e in all_data:
        label = e['label']
        if 'analysis' in e and isinstance(e['analysis'], str):
            lengths[label]['analysis'].append(len(e['analysis']))
        if 'final' in e and isinstance(e['final'], str):
            lengths[label]['final'].append(len(e['final']))

    for label in [0, 1]:
        for key in ['analysis', 'final']:
            avg = mean(lengths[label][key]) if lengths[label][key] else 0
            print(f"Average length of '{key}' for label {label}: {avg:.2f}")

    fields = ['final']
    colors = {0: 'steelblue', 1: 'orange'}

    plt.figure(figsize=(8, 5))

    for i, field in enumerate(fields):
        plt.subplot(1, 1, i + 1)
        plt.hist(lengths[0][field], bins=40, alpha=0.6, color=colors[0], label='Honest (label 0)')
        plt.hist(lengths[1][field], bins=40, alpha=0.6, color=colors[1], label='Deceptive (label 1)')
        plt.title(f"Distribution of '{field}' lengths")
        plt.xlabel('Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()

    # ---- Save plot ----
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output_plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {os.path.abspath(plot_path)}")

    plt.close()  


def prepare_data(deceptive_sampled, honest_data, tokenizer, crop=0):
    """
    Prepares the text inputs for activation extraction.

    Supports two formats:
      1) {"final": "...", "label": 0/1}
      2) {"prompt": "...", "output": "...", "label": 0/1}

    For (prompt, output), we build a Qwen-compatible chat transcript.
    """

    all_data = deceptive_sampled + honest_data

    filtered_data = [
        e for e in all_data
        if (
            ("final" in e and isinstance(e["final"], str) and e["final"].strip()) or
            ("prompt" in e and "output" in e
             and isinstance(e["prompt"], str)
             and isinstance(e["output"], str))
        )
    ]

    texts = []
    labels = []

    for e in filtered_data:

        # --- FORMAT: prompt + output ---
        if "prompt" in e and "output" in e:

            chat_str = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": e["prompt"]},
                    {"role": "assistant", "content": e["output"]},
                ],
                tokenize=False
            )

            texts.append(chat_str)
            labels.append(e["label"])
            continue

        # --- FORMAT: final ---
        if "final" in e and isinstance(e["final"], str):
            texts.append(e["final"])
            labels.append(e["label"])
            continue

    print(f"Total usable texts: {len(texts)}")
    return texts, labels



def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logging.info(
            f"{prefix} | GPU mem allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB"
        )


def extract_activations(
    texts,
    tokenizer,
    model,
    batch_size=8,
    max_length=512,
    use_mean=False,
    return_both=False,
):
    model.eval()
    model.config.output_hidden_states = True

    total_batches = (len(texts) + batch_size - 1) // batch_size
    start_time = time.time()

    logging.info(f"[ACT] Starting activation extraction")
    logging.info(f"[ACT] Total samples: {len(texts)} | Batch size: {batch_size} | Total batches: {total_batches}")

    all_layers_activations = []
    all_layers_last = []
    all_layers_mean = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
        batch_start = time.time()
        batch = texts[i:i + batch_size]

        logging.info(f"[ACT] Batch {batch_idx+1}/{total_batches} — Tokenizing")

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        logging.info(
            f"[ACT] Batch {batch_idx+1} — input shape: {inputs['input_ids'].shape}"
        )

        logging.info(f"[ACT] Batch {batch_idx+1} — Forward pass")
        with torch.no_grad():
            outputs = model(**inputs)

        logging.info(
            f"[ACT] Batch {batch_idx+1} — Forward done | "
            f"{len(outputs.hidden_states)-1} layers"
        )

        log_gpu_memory(f"[ACT] Batch {batch_idx+1}")

        hidden_states = outputs.hidden_states[1:]

        # ---- last token ----
        last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
        batch_last = torch.stack(
            [h[torch.arange(h.size(0)), last_token_indices] for h in hidden_states],
            dim=1
        )

        # ---- mean ----
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        batch_mean = []
        for h in hidden_states:
            masked_sum = (h * attention_mask).sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            batch_mean.append(masked_sum / lengths)
        batch_mean = torch.stack(batch_mean, dim=1)

        if return_both:
            all_layers_last.append(batch_last.cpu().float())
            all_layers_mean.append(batch_mean.cpu().float())
        else:
            batch_processed = batch_mean if use_mean else batch_last
            all_layers_activations.append(batch_processed.cpu().float())

        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time

        logging.info(
            f"[ACT] Batch {batch_idx+1} complete | "
            f"{batch_time:.2f}s batch | {elapsed/60:.2f} min total"
        )

    if return_both:
        return {
            "last_token": torch.cat(all_layers_last, dim=0).numpy(),
            "mean": torch.cat(all_layers_mean, dim=0).numpy(),
        }

    return torch.cat(all_layers_activations, dim=0).numpy()


def _get_activation_cache_paths(model_name, dataset_id, use_mean, base_cache_dir="cache/activations"):
    """
    Build paths for cached activations and labels.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    safe_model = model_name.replace("/", "_")
    mode = "mean" if use_mean else "last_token"

    cache_dir = os.path.join(base_dir, base_cache_dir, safe_model, dataset_id, mode)
    os.makedirs(cache_dir, exist_ok=True)

    activations_path = os.path.join(cache_dir, "activations.npy")
    labels_path = os.path.join(cache_dir, "labels.npy")
    return activations_path, labels_path


def extract_activations_cached(
    texts,
    labels,
    tokenizer,
    model,
    batch_size=8,
    use_mean=False,
    save_both=True,
    model_name="model",
    dataset_id="default",
    force_load=False,
):
    activations_path, labels_path = _get_activation_cache_paths(
        model_name=model_name,
        dataset_id=dataset_id,
        use_mean=use_mean,
    )

    mean_path, mean_labels_path = _get_activation_cache_paths(
        model_name=model_name,
        dataset_id=dataset_id,
        use_mean=True,
    )
    last_path, last_labels_path = _get_activation_cache_paths(
        model_name=model_name,
        dataset_id=dataset_id,
        use_mean=False,
    )

    if save_both:
        if os.path.exists(mean_path) and os.path.exists(last_path):
            print(f"[Cache] Loaded cached activations (both modes) for '{dataset_id}'")
            return np.load(mean_path if use_mean else last_path)

    if force_load:
        if not os.path.exists(activations_path):
            raise FileNotFoundError(
                f"[Cache] force_load=True, but cached activations missing:\n{activations_path}"
            )
        return np.load(activations_path)

    if os.path.exists(activations_path) and os.path.exists(labels_path):
        acts = np.load(activations_path)
        cached_labels = np.load(labels_path)
        if acts.shape[0] == len(texts) and len(cached_labels) == len(labels):
            print(f"[Cache] Loaded activations for '{dataset_id}'")
            return acts

    if model is None:
        raise RuntimeError("Cache missing and model=None (train_only mode).")

    # ---- compute ----
    if save_both:
        acts = extract_activations(
            texts,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=512,
            return_both=True,
        )

        np.save(mean_path, acts["mean"])
        np.save(last_path, acts["last_token"])
        np.save(mean_labels_path, np.array(labels))
        np.save(last_labels_path, np.array(labels))

        return acts["mean"] if use_mean else acts["last_token"]

    else:
        acts = extract_activations(
            texts,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=512,
            use_mean=use_mean,
        )
        np.save(activations_path, acts)
        np.save(labels_path, np.array(labels))
        return acts

def get_probe_cache_status(
    model_name,
    train_dataset_id,
    use_mean,
    classifier_names,
    num_layers,
    base_cache_dir="cache/probes"
):
    """
      - Builds the cache directory path for probes
      - Checks if all expected probe files exist
      - Returns: (cache_dir, exists_flag)
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    safe_model = model_name.replace("/", "_")
    mode = "mean" if use_mean else "last_token"

    cache_dir = os.path.join(base_dir, base_cache_dir, safe_model, train_dataset_id, mode)
    os.makedirs(cache_dir, exist_ok=True)

    meta_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return cache_dir, False

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except:
        return cache_dir, False

    print("DEBUG → Expected classifiers:", classifier_names)
    print("DEBUG → Metadata classifiers:", meta.get("classifiers", []))
    print("DEBUG → Expected num_layers:", num_layers)
    print("DEBUG → Metadata num_layers:", meta.get("num_layers"))

    # Check consistency with expected classifiers + layers
    if meta.get("num_layers") != num_layers:
        return cache_dir, False
    if not set(classifier_names).issubset(set(meta.get("classifiers", []))):
        return cache_dir, False

    # Check each model file exists
    for clf in classifier_names:
        for layer in range(num_layers):
            pkl_path = os.path.join(cache_dir, f"{clf}_layer{layer}.pkl")
            pt_path = os.path.join(cache_dir, f"{clf}_layer{layer}.pt")
            if not (os.path.exists(pkl_path) or os.path.exists(pt_path)):
                return cache_dir, False

    return cache_dir, True

def _save_probes(cache_dir, classifier_names, trained_models, val_results):
    # Save metadata
    meta = {
        "classifiers": classifier_names,
        "num_layers": len(next(iter(trained_models.values()))),
        "val_results": val_results
    }
    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save each classifier per layer
    for clf_name, models in trained_models.items():
        for layer_idx, model in enumerate(models):

            # sklearn classifier
            if hasattr(model, "predict"):
                path = os.path.join(cache_dir, f"{clf_name}_layer{layer_idx}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(model, f)

            # torch probe ensemble
            elif isinstance(model, list):
                path = os.path.join(cache_dir, f"{clf_name}_layer{layer_idx}.pt")
                torch.save(model, path)

def _load_probes(cache_dir, classifier_names):
    trained_models = {}

    with open(os.path.join(cache_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    num_layers = meta["num_layers"]
    val_results = meta.get("val_results", None)

    # Filter val_results to only requested classifiers
    if val_results is not None:
        val_results = {
            k: v for k, v in val_results.items()
            if k in classifier_names
        }

    for clf_name in classifier_names:
        models_per_layer = []
        for layer_idx in range(num_layers):
    
            pkl_path = os.path.join(cache_dir, f"{clf_name}_layer{layer_idx}.pkl")
            pt_path  = os.path.join(cache_dir, f"{clf_name}_layer{layer_idx}.pt")
    
            if os.path.exists(pkl_path):
                # sklearn model → pickle load
                with open(pkl_path, "rb") as f:
                    models_per_layer.append(pickle.load(f))
    
            else:
                models_per_layer.append(
                    torch.load(
                        pt_path,
                        map_location=torch.device("cpu"),
                        weights_only=False
                    )
                )
    
        trained_models[clf_name] = models_per_layer
    
    return trained_models, val_results



def train_classifiers(all_layers_activations, labels, cfg, random_seed, train_dataset_id):

    base_dir = os.path.dirname(os.path.dirname(__file__))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    X = all_layers_activations
    y = np.array(labels)
    num_layers = X.shape[1]

    # --- Split data ---
    idx_train, idx_val, y_train, y_val = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y,
    )

    # --- scikit-learn models ---
    sklearn_models = {
        "lr": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "lr_centered": CenteredLogisticRegression(C=1.0),
        "rf": RandomForestClassifier(
            n_estimators=1000,
            max_depth=30,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=42,
            verbose=0
        ),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "svm": SVC(kernel="rbf", C=2, gamma="scale", probability=True),
    }

    # --- Optional torch-based probes (custom classes) ---
    probe_map = {
        "linearprobe": LinearProbe,
        "bilinearprobe": BilinearProbe,
        "polyprobe": PolyProbe,
        "eemlp": EEMLP,
        "contrastiveprobe": ContrastiveProbe,
    }

    selected = [c.lower() for c in cfg["classifiers"]]
    print(f"\nTraining {len(selected)} classifiers across {num_layers} layers...\n")

    model_name = cfg["model"]["name"]
    use_mean = cfg["experiment"]["mean"]
    classifier_names = [c.upper() for c in selected]

    # --- Probe caching: check if we already trained these probes ---
    cache_dir, cache_exists = get_probe_cache_status(
        model_name=model_name,
        train_dataset_id=train_dataset_id,
        use_mean=use_mean,
        classifier_names=classifier_names,
        num_layers=num_layers,
    )

    if cache_exists:
        print(f"[Cache] Loaded cached probes from {cache_dir}")
        trained_models, val_results = _load_probes(cache_dir, classifier_names)
        results = val_results
        return {"results": results, "models": trained_models}
    
    else:
        print(f"Probes for '{train_dataset_id}' not found → training new probes...")


    # Storage for metrics and trained models
    results = {name.upper(): {"acc": [], "auc": []} for name in selected}
    trained_models = {name.upper(): [] for name in selected}

    # === Train on each layer ===
    for layer_id in range(num_layers):
        X_layer = X[:, layer_id, :]
        X_train, X_val = X_layer[idx_train], X_layer[idx_val]

        

        # --- sklearn models ---
        for name, model in sklearn_models.items():
            if name in selected:
                try:
                    clf = clone(model)  # create a fresh copy
                    clf.fit(X_train, y_train)
                    y_pred_val = clf.predict(X_val)
                    acc_val = accuracy_score(y_val, y_pred_val)
                    auc_val = (
                        roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
                        if len(np.unique(y_val)) >= 2 else np.nan
                    )
                except Exception as e:
                    print(f"⚠️ {name.upper()} failed on layer {layer_id}: {e}")
                    acc_val, auc_val = np.nan, np.nan

                results[name.upper()]["acc"].append(acc_val)
                results[name.upper()]["auc"].append(auc_val)
                trained_models[name.upper()].append(clf)

        # --- Torch-based probes (if defined) ---
        for name, probe_cls in probe_map.items():
            if name in selected:
                probe_cfg = cfg["probe_training"]
                in_dim = X_layer.shape[1]
                epochs = probe_cfg.get("epochs", 10)
                num_seeds = probe_cfg.get("num_seeds", 1)

                X_t = torch.tensor(X_train, dtype=torch.float32)
                y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                Xv_t = torch.tensor(X_val, dtype=torch.float32)
                yv_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X_t, y_t), batch_size=16, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(Xv_t, yv_t), batch_size=16
                )

                cls_fn = lambda **kwargs: probe_cls(in_dim, 1, **{k: v for k, v in kwargs.items() if k in probe_cfg})

                cls_all, _, _, _ = train(
                    cls_fn,
                    train_loader,
                    val_loader,
                    num_epochs=epochs,
                    train_mode=probe_cfg.get("train_mode", "train_all"),
                    lr=1e-4,
                    weight_decay=probe_cfg.get("weight_decay", 0.1),
                    d_prob=probe_cfg.get("dropout", 0.0),
                    device=device,
                    num_seeds=num_seeds,
                )

                # Evaluate probe ensemble on validation
                acc_list, auc_list = [], []
                for probe in cls_all:
                    probe.to(device)
                    probe.eval()
                    with torch.no_grad():
                        # Ensure Xv_t is a tensor on the correct device
                        if not isinstance(Xv_t, torch.Tensor):
                            Xv_t_t = torch.tensor(Xv_t, dtype=torch.float32, device=device)
                        else:
                            Xv_t_t = Xv_t.detach().clone().to(device).float()

                        # Forward pass
                        out = probe(Xv_t_t)

                        # Handle cases where probe returns list/tuple
                        if isinstance(out, (list, tuple)):
                            out = out[0]

                        # Ensure 'out' is a tensor
                        if not isinstance(out, torch.Tensor):
                            out = torch.as_tensor(out, dtype=torch.float32, device=device)

                        # Convert to probabilities
                        probs = torch.sigmoid(out).cpu().numpy().ravel()
                        preds = (probs > 0.5).astype(int)

                        # Metrics
                        acc_list.append(accuracy_score(y_val, preds))

                        if len(np.unique(y_val)) >= 2:
                            try:
                                auc_list.append(float(roc_auc_score(y_val, probs)))
                            except Exception:
                                auc_list.append(np.nan)
                        else:
                            auc_list.append(np.nan)

                acc_mean, auc_mean = np.nanmean(acc_list), np.nanmean(auc_list)
                results[name.upper()]["acc"].append(acc_mean)
                results[name.upper()]["auc"].append(auc_mean)
                trained_models[name.upper()].append(cls_all)

        # --- Print layer summary ---
        summary = [
            f"Layer {layer_id:2d}: {n}: ACC={results[n]['acc'][-1]:.3f}, AUC={np.nan_to_num(results[n]['auc'][-1], nan=0):.3f}"
            for n in results
        ]
        print(f"{' | '.join(summary)}")

    # === Plot Validation Performance ===
    layers = np.arange(num_layers)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for name in results:
        plt.plot(layers, results[name]["acc"], marker="o", label=name)
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Layer")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for name in results:
        plt.plot(layers, results[name]["auc"], marker="o", label=name)
    plt.xlabel("Layer Index")
    plt.ylabel("ROC AUC")
    plt.title("Validation AUC per Layer")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.join(base_dir, "output_plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "output_plots", "layerwise_val_performance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nValidation performance plot saved to: {out_path}\n")

    # --- Save trained probes + validation metrics to cache ---
    _save_probes(cache_dir, classifier_names, trained_models, results)
    print(f"[Cache] Saved trained probes + val results to: {cache_dir}")

    return {"results": results, "models": trained_models}



def evaluate_classifiers(trained_models, all_layers_activations_test, labels_test):
    num_layers = all_layers_activations_test.shape[1]
    results_test = {name: {"acc": [], "auc": []} for name in trained_models}

    print(f"\nEvaluating {len(trained_models)} classifier types across {num_layers} layers...\n")

    for name, model_list in trained_models.items():
        print(f"Classifier: {name}")
        for layer_id in range(num_layers):
            clf = model_list[layer_id]
            X_test = all_layers_activations_test[:, layer_id, :]

            # --- sklearn classifiers ---
            if hasattr(clf, "predict"):
                try:
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(labels_test, y_pred)
                    if hasattr(clf, "predict_proba"):
                        scores = clf.predict_proba(X_test)[:, 1]
                        auc = float(roc_auc_score(labels_test, scores)) if len(np.unique(labels_test)) >= 2 else np.nan
                    else:
                        auc = np.nan
                except Exception as e:
                    print(f"  Layer {layer_id}: sklearn model failed ({e})")
                    acc, auc = np.nan, np.nan

            # --- torch models (list of seeds per layer) ---
            elif isinstance(clf, list) and hasattr(clf[0], "forward"):
                acc_list, auc_list = [], []
                
                # Ensure X_test is a proper tensor (avoid warning)
                if not isinstance(X_test, torch.Tensor):
                    X_t = torch.as_tensor(X_test, dtype=torch.float32)
                else:
                    X_t = X_test.detach().clone().float()

                y_t = np.array(labels_test)
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                X_t = X_t.to(device)

                for net in clf:
                    net.eval()
                    with torch.no_grad():
                        try:
                            # Forward pass
                            out = net(X_t)

                            # Handle possible list/tuple outputs
                            if isinstance(out, (list, tuple)):
                                out = out[0]

                            # Ensure it's a tensor
                            if not isinstance(out, torch.Tensor):
                                out = torch.as_tensor(out, dtype=torch.float32, device=device)

                            # Apply sigmoid to get probabilities
                            probs = torch.sigmoid(out).detach().cpu().numpy().ravel()

                            # Compute predictions and metrics
                            preds = (probs > 0.5).astype(int)
                            acc_list.append(accuracy_score(y_t, preds))

                            if len(np.unique(y_t)) >= 2:
                                auc_list.append(float(roc_auc_score(y_t, probs)))
                            else:
                                auc_list.append(np.nan)

                        except Exception as e:
                            print(f"  Layer model failed ({e})")
                            acc_list.append(np.nan)
                            auc_list.append(np.nan)

                acc = np.nanmean(acc_list)
                auc = np.nanmean(auc_list)

            else:
                print(f"  Layer {layer_id}: Unrecognized model type")
                acc, auc = np.nan, np.nan

            results_test[name]["acc"].append(acc)
            results_test[name]["auc"].append(auc)
            print(f"  Layer {layer_id:2d} | ACC={acc:.3f} | AUC={auc:.3f}")

        # Summary per classifier
        acc_mean = np.nanmean(results_test[name]["acc"])
        auc_mean = np.nanmean(results_test[name]["auc"])
        print(f"  Summary → ACC={acc_mean:.3f} | AUC={auc_mean:.3f}\n")

    print("Testing complete.\n")

    return results_test

def analyze_layers(all_layers_activations, labels, all_layers_activations_target, labels_target, cfg, random_seed):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    X = all_layers_activations
    y = np.array(labels)
    X_target = all_layers_activations_target
    y_target = np.array(labels_target)

    num_layers = X.shape[1]

    # --- Train/test split for source domain (indices) ---
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(y)),
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y
    )

    sklearn_models = {
        "lr": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "svm": SVC(kernel="rbf", C=2, gamma="scale", probability=True),
    }

    probe_map = {
        "linearprobe": LinearProbe,
        "bilinearprobe": BilinearProbe,
        "polyprobe": PolyProbe,
        "eemlp": EEMLP,
    }

    selected = [c.lower() for c in cfg["classifiers"]]
    # results structure: results[name] = {'acc': [.. per layer ..], 'auc': [.. per layer ..]}
    results = {name.upper(): {'acc': [], 'auc': []} for name in selected}
    results_target = {name.upper(): {'acc': [], 'auc': []} for name in selected}

    print(f"Training {len(selected)} classifiers on {num_layers} layers...\n")

    for layer_id in range(num_layers):
        X_layer = X[:, layer_id, :]
        X_train, X_test = X_layer[X_train_idx], X_layer[X_test_idx]
        X_layer_target = X_target[:, layer_id, :]

        # --- sklearn models ---
        for name, model in sklearn_models.items():
            if name in selected:
                model.fit(X_train, y_train)

                # accuracy on source val (X_test) and target (X_layer_target)
                y_pred_val = model.predict(X_test)
                acc_val = accuracy_score(y_test, y_pred_val)

                # On target: model.predict might require same feature dim, we assume same dim
                try:
                    y_pred_tgt = model.predict(X_layer_target)
                    acc_tgt = accuracy_score(y_target, y_pred_tgt)
                except Exception:
                    acc_tgt = np.nan

                # AUC computation (safe: only if both classes present)
                if len(np.unique(y_test)) >= 2:
                    scores_val = _get_sklearn_scores(model, X_test)
                    try:
                        auc_val = float(roc_auc_score(y_test, scores_val))
                    except Exception:
                        auc_val = np.nan
                else:
                    auc_val = np.nan

                if len(np.unique(y_target)) >= 2:
                    try:
                        scores_tgt = _get_sklearn_scores(model, X_layer_target)
                        auc_tgt = float(roc_auc_score(y_target, scores_tgt))
                    except Exception:
                        auc_tgt = np.nan
                else:
                    auc_tgt = np.nan

                results[name.upper()]['acc'].append(acc_val)
                results[name.upper()]['auc'].append(auc_val)
                results_target[name.upper()]['acc'].append(acc_tgt)
                results_target[name.upper()]['auc'].append(auc_tgt)

        # --- torch probes  ---
        for name, probe_cls in probe_map.items():
            if name in selected:
                in_dim = X_layer.shape[1]
                probe_train_cfg = cfg["probe_training"]
                epochs = probe_train_cfg.get("epochs", 10)

                # Prepare tensors
                X_t = torch.tensor(X_train, dtype=torch.float32)
                y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                Xv_t = torch.tensor(X_test, dtype=torch.float32)
                yv_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

                Xtgt_t = torch.tensor(X_layer_target, dtype=torch.float32)
                ytgt_t = torch.tensor(y_target, dtype=torch.float32).unsqueeze(1)

                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X_t, y_t), batch_size=16, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(Xv_t, yv_t), batch_size=16
                )

                classifier = lambda **kwargs: probe_cls(
                    in_dim, 1,
                    **{k:v for k,v in kwargs.items() if k in probe_train_cfg}
                )

                cls_all, _, best_epochs, val_accs = train(
                    classifier,
                    train_loader,
                    val_loader,
                    num_epochs=epochs,
                    train_mode=probe_train_cfg.get("train_mode", "train_all"),
                    min_order=probe_train_cfg.get("min_order", 1),
                    max_order=probe_train_cfg.get("max_order", 3),
                    metric=lambda p, l: (torch.sigmoid(p) > 0.5).eq(l.bool()).float().mean().item(),
                    lr=probe_train_cfg.get("lr", 1e-4),
                    weight_decay=probe_train_cfg.get("weight_decay", 0.1),
                    d_prob=probe_train_cfg.get("dropout", 0.0),
                    device=device,
                    num_seeds=probe_train_cfg.get("num_seeds", 1),
                    early_stopping_patience=probe_train_cfg.get("early_stopping_patience"),
                    metric_min_delta=probe_train_cfg.get("metric_min_delta", 1e-4),
                )

                acc_val_list = []
                auc_val_list = []
                acc_tgt_list = []
                auc_tgt_list = []

                Xv_np = Xv_t.numpy()
                Xtgt_np = Xtgt_t.numpy()
                yv_np = yv_t.numpy().ravel()
                ytgt_np = ytgt_t.numpy().ravel()

                for trained_model in cls_all:
                    trained_model.to(device)
                    trained_model.eval()
                    with torch.no_grad():
                        out_val = trained_model(Xv_t.to(device))
                        if isinstance(out_val, (list, tuple)):
                            out_val = out_val[0]
                        probs_val = torch.sigmoid(out_val).cpu().numpy().ravel()
                        preds_val = (probs_val > 0.5).astype(int)
                        acc_val_list.append(accuracy_score(yv_np, preds_val))
                        if len(np.unique(yv_np)) >= 2:
                            try:
                                auc_val_list.append(float(roc_auc_score(yv_np, probs_val)))
                            except Exception:
                                auc_val_list.append(np.nan)
                        else:
                            auc_val_list.append(np.nan)

                        out_tgt = trained_model(Xtgt_t.to(device))
                        if isinstance(out_tgt, (list, tuple)):
                            out_tgt = out_tgt[0]
                        probs_tgt = torch.sigmoid(out_tgt).cpu().numpy().ravel()
                        preds_tgt = (probs_tgt > 0.5).astype(int)
                        acc_tgt_list.append(accuracy_score(ytgt_np, preds_tgt))
                        if len(np.unique(ytgt_np)) >= 2:
                            try:
                                auc_tgt_list.append(float(roc_auc_score(ytgt_np, probs_tgt)))
                            except Exception:
                                auc_tgt_list.append(np.nan)
                        else:
                            auc_tgt_list.append(np.nan)

                # average across seeds/models
                acc_val_mean = float(np.nanmean(acc_val_list)) if acc_val_list else np.nan
                auc_val_mean = float(np.nanmean(auc_val_list)) if auc_val_list else np.nan
                acc_tgt_mean = float(np.nanmean(acc_tgt_list)) if acc_tgt_list else np.nan
                auc_tgt_mean = float(np.nanmean(auc_tgt_list)) if auc_tgt_list else np.nan

                results[name.upper()]['acc'].append(acc_val_mean)
                results[name.upper()]['auc'].append(auc_val_mean)
                results_target[name.upper()]['acc'].append(acc_tgt_mean)
                results_target[name.upper()]['auc'].append(auc_tgt_mean)

        # --- Print summary for this layer ---
        summary = []
        for n in results:
            val_acc = results[n]['acc'][-1]
            val_auc = results[n]['auc'][-1]
            tgt_acc = results_target[n]['acc'][-1]
            tgt_auc = results_target[n]['auc'][-1]
            summary.append(f"{n}: acc {val_acc:.3f} (tgt {tgt_acc:.3f}) | auc {np.nan_to_num(val_auc, nan=0):.3f} (tgt {np.nan_to_num(tgt_auc, nan=0):.3f})")
        print(f"Layer {layer_id:2d} → " + " | ".join(summary))

    # --- Plot accuracy and AUC in two stacked subplots ---
    plt.figure(figsize=(12, 8))
    layers = range(num_layers)

    # Accuracy subplot
    plt.subplot(2, 1, 1)
    for name in results:
        plt.plot(layers, results[name]['acc'], marker='o', label=f"{name} (val)")
        plt.plot(layers, results_target[name]['acc'], marker='x', linestyle='--', label=f"{name} (tgt)")
    plt.ylabel("Accuracy")
    plt.title("Validation vs Target Accuracy by Layer")
    plt.legend()
    plt.grid(True)

    # AUC subplot
    plt.subplot(2, 1, 2)
    for name in results:
        plt.plot(layers, results[name]['auc'], marker='o', label=f"{name} (val)")
        plt.plot(layers, results_target[name]['auc'], marker='x', linestyle='--', label=f"{name} (tgt)")
    plt.xlabel("Layer index")
    plt.ylabel("ROC AUC")
    plt.title("Validation vs Target ROC AUC by Layer")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(os.path.join(base_dir, "output_plots"), exist_ok=True)
    plot_file = os.path.join(base_dir, "output_plots", "layerwise_probe_acc_auc_transfer.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print(f"\nSaved plot to {plot_file}\n")

    # Return both metrics
    return {"val": results, "target": results_target}



from sklearn.decomposition import PCA

def plot_class_activations_and_pca(
    all_layers_activations, labels,
    all_layers_activations_target, labels_target,
    base_dir, max_layers_to_plot=29,
    scale_by_std=False
):

    num_layers = all_layers_activations.shape[1]
    os.makedirs(os.path.join(base_dir, "output_plots"), exist_ok=True)

    # --- Select layers to visualize ---
    layers_to_plot = np.linspace(0, num_layers - 1, max_layers_to_plot, dtype=int)

    # --- Prepare subplot grid ---
    ncols = 4
    nrows = int(np.ceil(len(layers_to_plot) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()

    # --- Loop through each selected layer ---
    for i, layer_id in enumerate(layers_to_plot):
        ax = axes[i]

        X = all_layers_activations[:, layer_id, :]
        X_tgt = all_layers_activations_target[:, layer_id, :]

        # Optional per-sample standardization
        if scale_by_std:
            X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
            X_tgt = (X_tgt - X_tgt.mean(axis=1, keepdims=True)) / (X_tgt.std(axis=1, keepdims=True) + 1e-8)

        # PCA fit on source
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X)
        X_pca = pca.transform(X)
        X_tgt_pca = pca.transform(X_tgt)

        # Scatter plots
        ax.scatter(X_pca[np.array(labels) == 0, 0], X_pca[np.array(labels) == 0, 1],
                   c='blue', alpha=0.6, s=8, label="Honest (Src)")
        ax.scatter(X_pca[np.array(labels) == 1, 0], X_pca[np.array(labels) == 1, 1],
                   c='red', alpha=0.6, s=8, label="Deceptive (Src)")
        ax.scatter(X_tgt_pca[np.array(labels_target) == 0, 0], X_tgt_pca[np.array(labels_target) == 0, 1],
                   c='blue', marker='x', label="Honest (Tgt)")
        ax.scatter(X_tgt_pca[np.array(labels_target) == 1, 0], X_tgt_pca[np.array(labels_target) == 1, 1],
                   c='red', marker='x', label="Deceptive (Tgt)")

        ax.set_title(f"PCA Layer {layer_id}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)

        # Only show legend once
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots if any
    for j in range(len(layers_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(" PCA of Layer Activations (Source vs Target)", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- Save combined figure ---
    combined_path = os.path.join(base_dir, "output_plots", "pca_all_layers_summary.png")
    plt.savefig(combined_path, dpi=300)
    plt.close(fig)

    print(f"\nCombined PCA summary plot saved to: {combined_path}\n")
