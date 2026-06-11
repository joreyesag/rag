"""
S-RAG Auditing Data Provenance - Evaluation Module
Calcula KPIs de auditoría sobre predicciones de AutoGluon.
"""

import os
import argparse
import numpy as np
import pandas as pd
import jsonlines
from rich.console import Console
from rich.panel import Panel
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

console = Console()

def evaluate_model(result_store_path: str, feature_file: str, bin_num: int, audit_model_path: str):
    console.print(f"[cyan]Iniciando evaluación del modelo: {audit_model_path}[/cyan]")

    # Carga de probabilidades generadas
    probabilities = []
    with jsonlines.open(os.path.join(result_store_path, feature_file)) as reader:
        for obj in reader:
            probabilities.append(obj.get('probability', []))

    # Transformación a histogramas (Features)
    bins = np.arange(0, 1.1, 1 / bin_num)
    hists = [np.histogram(prob, bins)[0] for prob in probabilities]

    # Ground truth: La mitad son miembros (clase 1), la otra mitad no (clase 0)
    # Nota: Si tu set de evaluación no mantiene el balance 50/50, ajusta esto según el origen.
    n = len(hists)
    y_true = [0] * (n // 2) + [1] * (n // 2)

    feature_columns = [f'feature{i + 1}' for i in range(bin_num)]
    test_data = pd.DataFrame(
        {feature_columns[i]: [hist[i] for hist in hists] for i in range(bin_num)}
    )

    # Inferencia con AutoGluon
    loaded_predictor = TabularPredictor.load(audit_model_path)
    
    y_pred = loaded_predictor.predict(test_data)
    y_prob = loaded_predictor.predict_proba(test_data).iloc[:, 1]

    # Reporte de métricas con Rich
    console.print(Panel.fit(
        f"[bold]Métricas de Auditoría (N={n})[/bold]\n\n"
        f"AUC: [green]{roc_auc_score(y_true, y_prob):.4f}[/green]\n"
        f"ACC: [green]{accuracy_score(y_true, y_pred):.4f}[/green]\n"
        f"PRE: [green]{precision_score(y_true, y_pred):.4f}[/green]\n"
        f"REC: [green]{recall_score(y_true, y_pred):.4f}[/green]\n"
        f"F1 : [green]{f1_score(y_true, y_pred):.4f}[/green]",
        title="Resultados Finales"
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='HealthCare')
    parser.add_argument('--method', type=str, default='Audit')
    parser.add_argument('--llm', type=str, default='llama3')
    parser.add_argument('--data_store_path', type=str, default='Data')
    parser.add_argument('--result_store_path', type=str, default='Result')
    parser.add_argument('--defence', choices=['wo', 'prompt_modify', 'paraphrasing'], default='wo')
    parser.add_argument('--bin_num', type=int, default=10)
    parser.add_argument('--audit_model', type=str, default='./Model/AutoGluon', help="Path to saved model")
    
    args = parser.parse_args()

    # Resolución de rutas consistente con Audit.py
    model_id = "./Model/llama-3-8b-Instruct" if 'llama' in args.llm else "gpt-4o-mini"
    prefix = f"-{args.defence}"
    feature_file = f"{args.dataset_name}{model_id.replace('/', '-').replace('.', '-')}{prefix}-Audit_feature.jsonl"

    evaluate_model(args.result_store_path, feature_file, args.bin_num, args.audit_model)