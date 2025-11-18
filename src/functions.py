# Matriz de dispersión para una columna objetivo
def plot_matriz_dispersion(df, target_col="Addiction_Level", excluir_cols=None, n_cols=3):
    if excluir_cols is None:
        excluir_cols = [target_col, "AddiccionBinaria"]

    variables_numericas = df.select_dtypes(include="number").columns
    variables_numericas = [col for col in variables_numericas if col not in excluir_cols]

    n_vars = len(variables_numericas)
    if n_vars == 0:
        return

    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    unique_levels = sorted(df[target_col].unique())
    unique_levels_int = sorted({int(round(x)) for x in unique_levels})

    for idx, col in enumerate(variables_numericas):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].scatter(df[col], df[target_col], alpha=0.5)
        axes[r][c].set_title(col)
        axes[r][c].set_xlabel(col)
        axes[r][c].set_ylabel(target_col)
        axes[r][c].set_yticks(unique_levels_int)

    for idx in range(n_vars, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].set_visible(False)

    fig.suptitle(f"Matriz de gráficos de dispersión ({target_col} en eje Y)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
