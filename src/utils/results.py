import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import log

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)


def _slugify(name: str) -> str:
    """Cria um nome de pasta/arquivo seguro a partir do nome do dataset."""
    return "".join(
        c if c.isalnum() or c in [" ", "-", "_"] else "_" for c in name
    ).replace(" ", "_")


def summarize_results_in_all_csvs(results_root: str = "results"):
    log.info("-------------- Calculating mean and std for all Results")

    for root, _, files in os.walk(results_root):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(csv_path, sep=";")
                    numeric_cols = df.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()

                    if not numeric_cols:
                        log.warning(
                            f"No numeric columns found in {csv_path}, skipping."
                        )
                        continue

                    mean_row = df[numeric_cols].mean()
                    std_row = df[numeric_cols].std()

                    mean_row_full = {
                        col: "MEAN"
                        if col == "dataset_name"
                        else mean_row.get(col, None)
                        for col in df.columns
                    }
                    std_row_full = {
                        col: "STD" if col == "dataset_name" else std_row.get(col, None)
                        for col in df.columns
                    }

                    df = pd.concat(
                        [df, pd.DataFrame([mean_row_full, std_row_full])],
                        ignore_index=True,
                    )
                    df.to_csv(csv_path, sep=";", index=False)

                    log.info(f"Summarized {csv_path} with mean and std.")
                except Exception as e:
                    log.error(f"Error summarizing {csv_path}: {e}")

    log.info("-------------- All Result files summarized.")


def consolidate_results(results_root="results"):
    """Percorre todas as subpastas de 'results' e concatena todos os CSVs."""
    log.info(f"Procurando arquivos CSV em: {results_root}")
    dfs = []
    for root, _, files in os.walk(results_root):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(csv_path, sep=";")
                    df["source_file"] = file
                    dfs.append(df)
                except Exception as e:
                    log.error(f"Erro ao ler {csv_path}: {e}")
    if not dfs:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado em 'results/'.")
    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total de resultados combinados: {len(combined)} linhas")
    return combined


def preprocess_results(df: pd.DataFrame):
    """Remove linhas de média/STD e garante tipos numéricos."""
    # Criar uma cópia para evitar SettingWithCopyWarning
    df_filtered = df[~df["dataset_name"].isin(["MEAN", "STD"])].copy()
    numeric_cols = ["accuracy", "precision", "recall", "f1_score"]

    # Adiciona coluna de tempo de vetorização se existir
    if "total_vectorization_time" in df_filtered.columns:
        numeric_cols.append("total_vectorization_time")

    for col in numeric_cols:
        if col in df_filtered.columns:
            df_filtered.loc[:, col] = pd.to_numeric(df_filtered[col], errors="coerce")

    # Garantir que total_vectorization_time seja float e arredondado a 4 casas decimais
    if "total_vectorization_time" in df_filtered.columns:
        df_filtered["total_vectorization_time"] = pd.to_numeric(
            df_filtered["total_vectorization_time"], errors="coerce"
        ).round(4)
    return df_filtered


def generate_visualizations(df: pd.DataFrame, output_dir="results_visuals"):
    """Gera boxplots e gráficos comparativos agrupados por dataset, modelo e vetorizador.
    Agora inclui plots separados por modelo dentro de cada dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Gerando visualizações em: {output_dir}")

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    # --- (opcionais) plots globais que você já tinha (mantive) ---
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="model_name", y=metric, hue="vectorizer_name")
        plt.title(
            f"{metric.capitalize()} — Comparação entre Modelos e Vetorizadores (Global)"
        )
        plt.xticks(rotation=30, ha="right")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/boxplot_model_{metric}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="dataset_name", y=metric, hue="vectorizer_name")
        plt.title(
            f"{metric.capitalize()} — Comparação entre Datasets e Vetorizadores (Global)"
        )
        plt.xticks(rotation=30, ha="right")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/boxplot_dataset_{metric}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Média por modelo+vetorizador (global)
    mean_df = (
        df.groupby(["model_name", "vectorizer_name"])[
            ["accuracy", "precision", "recall", "f1_score"]
        ]
        .mean()
        .reset_index()
    )

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=mean_df, x="model_name", y=metric, hue="vectorizer_name")
        plt.title(f"Média de {metric.capitalize()} por Modelo e Vetorizador (Global)")
        plt.xticks(rotation=30, ha="right")
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.2f",
                label_type="center",
                fontsize=7,
                fontweight="bold",
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/barplot_mean_{metric}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # ----- Per-dataset plots: one folder per dataset -----
    datasets = df["dataset_name"].dropna().unique().tolist()
    for ds in datasets:
        ds_slug = _slugify(str(ds))
        ds_out = os.path.join(output_dir, ds_slug)
        os.makedirs(ds_out, exist_ok=True)

        df_ds = df[df["dataset_name"] == ds].copy()
        if df_ds.empty:
            log.info(f"No data for dataset {ds}; skipping.")
            continue

        # summary: how many models and vectorizers
        models = df_ds["model_name"].dropna().unique().tolist()
        vects = df_ds["vectorizer_name"].dropna().unique().tolist()
        log.info(
            f"Dataset '{ds}': models={models}, vectorizers={vects}, rows={len(df_ds)}"
        )

        # 1) Per-dataset overall boxplots (models vs vectorizers) - optional
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_ds, x="model_name", y=metric, hue="vectorizer_name")
            plt.title(f"{metric.capitalize()} — {ds} — Modelos x Vetorizadores")
            plt.xticks(rotation=30, ha="right")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)
            plt.tight_layout()
            plt.savefig(
                f"{ds_out}/boxplot_model_{metric}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # 2) === NOVO: separar por MODELO dentro de cada dataset ===
        for model in models:
            model_slug = _slugify(str(model))
            df_ds_model = df_ds[df_ds["model_name"] == model]
            if df_ds_model.empty:
                continue

            # boxplot: x = vectorizer_name, y = metric  (separado por modelo)
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_ds_model, x="vectorizer_name", y=metric)
                plt.title(f"{metric.capitalize()} — {ds} — Modelo: {model}")
                plt.xlabel("Vetorizador")
                plt.ylabel(metric.capitalize())
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(
                    f"{ds_out}/boxplot_{model_slug}_{metric}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # mean barplot per vectorizer for this model
            mean_df_model = (
                df_ds_model.groupby("vectorizer_name")[
                    ["accuracy", "precision", "recall", "f1_score"]
                ]
                .mean()
                .reset_index()
            )
            for metric in metrics:
                if mean_df_model.empty:
                    continue
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(data=mean_df_model, x="vectorizer_name", y=metric)
                plt.title(f"Média de {metric.capitalize()} — {ds} — Modelo: {model}")
                plt.xlabel("Vetorizador")
                plt.ylabel(metric.capitalize())
                plt.xticks(rotation=30, ha="right")
                # labels inside bars
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f", fontsize=8, label_type="center")
                plt.tight_layout()
                plt.savefig(
                    f"{ds_out}/barplot_mean_{model_slug}_{metric}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        # 3) Heatmap (model x vectorizer) usando médias já calculadas
        mean_df_ds = (
            df_ds.groupby(["model_name", "vectorizer_name"])[
                ["accuracy", "precision", "recall", "f1_score"]
            ]
            .mean()
            .reset_index()
        )
        pivot_ds = mean_df_ds.pivot(
            index="model_name", columns="vectorizer_name", values="f1_score"
        )
        if pivot_ds is not None and not pivot_ds.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_ds,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                cbar_kws={"label": "F1-Score"},
            )
            plt.title(f"F1-score Médio — {ds}")
            plt.tight_layout()
            plt.savefig(
                f"{ds_out}/heatmap_f1score_{ds_slug}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    # (rest of vectorization time visualizations kept as you had them)
    if "total_vectorization_time" in df.columns:
        log.info("Gerando visualizações para tempos de vetorização...")
        time_df = df[df["total_vectorization_time"] > 0].dropna(
            subset=["total_vectorization_time"]
        )
        if len(time_df) > 0:
            # Debug: confirmar tipo e amostra dos tempos lidos
            log.info(
                f"total_vectorization_time dtype: {time_df['total_vectorization_time'].dtype}"
            )
            log.info(
                f"Amostra total_vectorization_time:\n{time_df['total_vectorization_time'].head(10).to_list()}"
            )
            # ... copia exatamente o bloco que você já tinha para tempos (mantive inalterado)
            datasets_time = time_df["dataset_name"].dropna().unique().tolist()
            for ds in datasets_time:
                ds_slug = _slugify(str(ds))
                ds_out = os.path.join(output_dir, ds_slug)
                os.makedirs(ds_out, exist_ok=True)

                df_time_ds = time_df[time_df["dataset_name"] == ds]
                if df_time_ds.empty:
                    continue

                # Linear-scale boxplot per dataset
                plt.figure(figsize=(12, 8))
                sns.boxplot(
                    data=df_time_ds, x="vectorizer_name", y="total_vectorization_time"
                )
                plt.title(f"Tempo de Vetorização por Vetorizador — {ds} (Linear)")
                plt.xlabel("Vetorizador")
                plt.ylabel("Tempo de Vetorização (segundos)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(
                    f"{ds_out}/boxplot_vectorization_time_linear.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Log-scale boxplot per dataset
                plt.figure(figsize=(12, 8))
                sns.boxplot(
                    data=df_time_ds, x="vectorizer_name", y="total_vectorization_time"
                )
                plt.yscale("log")
                plt.title(f"Tempo de Vetorização por Vetorizador — {ds} (Log)")
                plt.xlabel("Vetorizador")
                plt.ylabel("Tempo de Vetorização (segundos) - Escala Log")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(
                    f"{ds_out}/boxplot_vectorization_time_log.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            log.info("Visualizações de tempo de vetorização geradas com sucesso!")
        else:
            log.warning("Nenhum dado válido de tempo de vetorização encontrado.")
    else:
        log.info(
            "Coluna 'total_vectorization_time' não encontrada. Pulando visualizações de tempo."
        )

    log.info(f"Todas as visualizações foram salvas em {output_dir}")


if __name__ == "__main__":
    try:
        df = consolidate_results("./final_results/results")
        df = preprocess_results(df)
        generate_visualizations(df, "results_plots")
        log.info("Visualizações geradas com sucesso!")
    except Exception as e:
        log.error(f"Erro ao gerar visualizações: {e}")
