# -*- coding: utf-8 -*-
"""
Лабораторная работа 2. Классификация и нейронная сеть на наборе Wine Quality.

Цель: хорошее вино (quality >= 7, класс 1) против обычного (класс 0).
Запуск из папки со скриптом (рядом winequalityN.csv):  python lab2_classification.py
Графики -> figures/, таблицы -> results/, журнал TensorBoard -> <SystemDrive>/lr2_wine_tensorboard
(путь только ASCII: иначе TensorFlow на Windows не пишет event-файлы).
Подробные выводы — в ЛР2_Классификация.ipynb.
"""

import os
import shutil
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110, "font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

HERE = Path(__file__).resolve().parent
FIGURES, RESULTS = HERE / "figures", HERE / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

NEG_NAME, POS_NAME = "обычное", "хорошее"

_FIG_NAMES = [
    "01_баланс_классов.png",
    "02_ящики_с_усами.png",
    "03_корреляция_признаков.png",
    "04_gaussian_nb.png",
    "05_multinomial_nb.png",
    "06_complement_nb.png",
    "07_bernoulli_порог.png",
    "08_bernoulli_alpha.png",
    "09_дерево_глубина.png",
    "10_lda_shrinkage.png",
    "11_svm_линейное_ядро.png",
    "12_svm_rbf.png",
    "13_knn.png",
    "14_прирост_настройки.png",
    "15_roc_кривые.png",
    "16_столбчатые_метрики.png",
    "17_матрицы_ошибок.png",
]
_NN_FIG_NAMES = [
    "nn_tb_01_скорость_обучения.png",
    "nn_tb_02_число_нейронов.png",
    "nn_tb_03_число_слоёв.png",
    "nn_tb_04_активация.png",
    "nn_tb_05_размер_батча.png",
    "nn_tb_06_число_эпох.png",
    "nn_tb_07_dropout.png",
    "nn_tb_08_матрицы_ошибок.png",
    "nn_tb_09_roc_сеть_и_классика.png",
]
_fig_i = 0
_nn_fig_i = 0


def _save(names, idx, prefix):
    name = names[idx] if idx < len(names) else f"{prefix}{idx + 1:02d}.png"
    plt.gcf().savefig(FIGURES / name, bbox_inches="tight")
    print("Сохранён график:", name)
    plt.close()


def save_show():
    """Сохраняет текущий рисунок под очередным именем из _FIG_NAMES."""
    global _fig_i
    _save(_FIG_NAMES, _fig_i, "fig_")
    _fig_i += 1


def save_show_nn():
    """То же для рисунков сети (nn_tb_*.png)."""
    global _nn_fig_i
    _save(_NN_FIG_NAMES, _nn_fig_i, "nn_tb_")
    _nn_fig_i += 1


def h(title):
    print(f"\n{title}")


def show(obj):
    print(obj)
    print()


# ---------------------------------------------------------------------------
# 1. Данные
# ---------------------------------------------------------------------------
h("## 1. Данные: Wine Quality")
csv_path = HERE / "winequalityN.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Не найден файл с данными: {csv_path}")
raw = pd.read_csv(csv_path)
print("Файл:", csv_path.name, "| исходный размер:", raw.shape)

df = raw.copy()
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
type_clean = df["type"].astype(str).str.strip().str.lower()
df["is_red"] = type_clean.map({"white": 0, "red": 1})
df = df.drop(columns=["type"])
assert df["quality"].notna().all(), "В quality не должно быть пропусков"
df["good"] = (df["quality"] >= 7).astype(int)

na_total = int(df.isna().sum().sum())
df = df.fillna(df.median(numeric_only=True))
assert int(df.isna().sum().sum()) == 0

n_before_dedup = len(df)
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
quality_counts = df["quality"].value_counts().sort_index()
df = df.drop(columns=["quality"])
df["is_red"] = df["is_red"].astype(int)

feature_cols = [c for c in df.columns if c != "good"]
X = df[feature_cols].copy()
y = df["good"].copy()
n = len(df)
n_pos = int(y.sum())
n_neg = n - n_pos
base_rate = float(y.mean())
n_features = X.shape[1]
assert set(y.unique()) == {0, 1} and int(X.isna().sum().sum()) == 0 and n_features == 12

print(f"Пропусков заполнено медианой: {na_total}; дубликатов удалено: {n_dup} из {n_before_dedup}")
print(f"Размер после очистки: {df.shape}; хороших {n_pos} ({base_rate:.1%}), обычных {n_neg}")
print("Распределение quality:", quality_counts.to_dict())

feature_meaning = {
    "is_red": "тип вина: 1 — красное, 0 — белое",
    "fixed_acidity": "нелетучие кислоты, г/дм³",
    "volatile_acidity": "летучая (уксусная) кислота, г/дм³",
    "citric_acid": "лимонная кислота, г/дм³",
    "residual_sugar": "остаточный сахар, г/дм³",
    "chlorides": "хлориды (соли), г/дм³",
    "free_sulfur_dioxide": "свободный диоксид серы, мг/дм³",
    "total_sulfur_dioxide": "общий диоксид серы, мг/дм³",
    "density": "плотность, г/см³",
    "pH": "водородный показатель",
    "sulphates": "сульфаты, г/дм³",
    "alcohol": "объёмная доля спирта, %",
}

# Эффект признаков (Cohen's d: хорошее минус обычное)
effect_rows = []
for col in feature_cols:
    x1, x0 = X.loc[y == 1, col], X.loc[y == 0, col]
    pooled = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
    effect_rows.append({
        "Признак": col,
        "Пояснение": f"{col} — {feature_meaning.get(col, col)}",
        "Cohen_d": (x1.mean() - x0.mean()) / pooled if pooled > 0 else 0.0,
        "Среднее обычное": x0.mean(),
        "Среднее хорошее": x1.mean(),
    })
effect = pd.DataFrame(effect_rows).sort_values("Cohen_d", key=lambda s: s.abs(), ascending=False)
show(effect[["Признак", "Cohen_d", "Среднее обычное", "Среднее хорошее"]].round(3).reset_index(drop=True))

# График 1. Баланс классов
fig, ax = plt.subplots(figsize=(6.2, 4.2))
counts = pd.Series({"Обычное (0)": n_neg, "Хорошее (1)": n_pos})
bars = ax.bar(counts.index, counts.values, color=["#4C78A8", "#E45756"])
for bar, value in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + n * 0.01, f"{value}\n({value / n:.1%})",
            ha="center", va="bottom")
ax.set_ylim(0, max(counts.values) * 1.22)
ax.set_ylabel("Число образцов")
ax.set_title("Баланс классов: хорошее и обычное вино")
fig.tight_layout()
save_show()

# График 2. Ящики с усами: топ-4 признака по |d| (без is_red)
top_features = effect[effect["Признак"] != "is_red"]["Признак"].head(4).tolist()
plot_df = df.copy()
plot_df["Класс"] = plot_df["good"].map({0: "Обычное", 1: "Хорошее"})
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
for ax, col in zip(axes.ravel(), top_features):
    sns.boxplot(data=plot_df, x="Класс", y=col, hue="Класс", order=["Обычное", "Хорошее"],
                hue_order=["Обычное", "Хорошее"], palette={"Обычное": "#4C78A8", "Хорошее": "#E45756"},
                legend=False, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
fig.suptitle("Признаки с самым сильным сдвигом средних между классами", y=1.02)
fig.tight_layout()
save_show()

# График 3. Корреляция признаков
corr = X.corr()
fig, ax = plt.subplots(figsize=(10, 8.4))
sns.heatmap(corr, cmap="vlag", center=0, square=True, annot=True, fmt=".2f", annot_kws={"fontsize": 7},
            ax=ax, cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
ax.tick_params(labelsize=8)
ax.set_title(f"Корреляция {n_features} признаков")
fig.tight_layout()
save_show()

# ---------------------------------------------------------------------------
# 2. Разбиение выборки
# ---------------------------------------------------------------------------
h("## 2. Разбиение на обучение и тест")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def split_line(name, target):
    return {"Выборка": name, "Объектов": int(len(target)), "Хороших": int(target.sum()),
            "Обычных": int((target == 0).sum()), "Доля хороших": float(target.mean())}


split_table = pd.DataFrame([split_line("Вся таблица", y), split_line("Обучение", y_train), split_line("Тест", y_test)])
show(split_table.round(4))
n_train, n_test = int(len(y_train)), int(len(y_test))
n_pos_test = int(y_test.sum())
majority_acc = (n_test - n_pos_test) / n_test
print(f"Константа «всегда обычное» на тесте: Accuracy {majority_acc:.3f}, Recall 0")

# ---------------------------------------------------------------------------
# 3-4. Кросс-валидация, метрики, правило выбора
# ---------------------------------------------------------------------------
h("## 3-4. Кросс-валидация и правило выбора параметров")
CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
}
best_configs = {}
tuning_summary = {}
print(f"StratifiedKFold({N_SPLITS}, shuffle=True, random_state={RANDOM_STATE}) на обучающей части ({n_train}).")
print("Правило: AUC в пределах 0.005 от максимума -> минимальный разрыв train-val -> больше AUC и F1.")


def format_param(value):
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value == 0:
            return "0"
        return f"{value:.0e}" if abs(value) < 0.01 else f"{value:g}"
    return str(value)


def evaluate_cv(estimator, X, y):
    result = cross_validate(estimator, X, y, cv=CV, scoring=SCORING, n_jobs=1,
                            return_train_score=True, error_score="raise")
    out = {}
    for key in SCORING:
        out[key] = float(result[f"test_{key}"].mean())
        out[f"{key}_std"] = float(result[f"test_{key}"].std())
        out[f"{key}_train"] = float(result[f"train_{key}"].mean())
    return out


def select_best(frame):
    pool = frame.copy()
    pool = pool[pool["roc_auc"] >= pool["roc_auc"].max() - 0.005].copy()
    pool["gap"] = pool["roc_auc_train"] - pool["roc_auc"]
    return pool.sort_values(["gap", "roc_auc", "f1"], ascending=[True, False, False]).iloc[0]


def sweep(make, combos, label, extra=None):
    """Считает CV для каждой комбинации параметров фабрики make; extra(kw) переопределяет столбцы строки."""
    rows = []
    for kw in combos:
        metrics = evaluate_cv(make(**kw), X_train, y_train)
        rows.append({**kw, **(extra(kw) if extra else {}), "param_label": label(kw), **metrics})
    return pd.DataFrame(rows)


def at(frame, **cond):
    mask = pd.Series(True, index=frame.index)
    for key, value in cond.items():
        mask &= frame[key] == value
    return frame[mask].iloc[0]


def plot_lines(frame, group_col, group_order, xlabel, title, ylabel="AUC-ROC", metric="roc_auc"):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for group in group_order:
        sub = frame[frame[group_col] == group]
        x = np.arange(len(sub))
        ax.plot(x, sub[metric], marker="o", label=str(group))
        if metric == "roc_auc" and f"{metric}_std" in sub.columns:
            ax.fill_between(x, sub[metric] - sub[f"{metric}_std"], sub[metric] + sub[f"{metric}_std"], alpha=0.12)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["param_label"], rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="")
    fig.tight_layout()
    save_show()


# ---------------------------------------------------------------------------
# Фабрики моделей
# ---------------------------------------------------------------------------
def _wrap(clf, scaler, scale):
    return Pipeline([("scaler", scaler), ("clf", clf)]) if scale else clf


def make_gaussian(var_smoothing=1e-9, scale=True):
    return _wrap(GaussianNB(var_smoothing=float(var_smoothing)), StandardScaler(), scale)


def make_multinomial(alpha=1.0, scale=True):
    return _wrap(MultinomialNB(alpha=float(alpha)), MinMaxScaler(), scale)


def make_complement(alpha=1.0, scale=True):
    return _wrap(ComplementNB(alpha=float(alpha)), MinMaxScaler(), scale)


def make_bernoulli(alpha=1.0, threshold=0.5, scale=True):
    return _wrap(BernoulliNB(alpha=float(alpha), binarize=float(threshold)), MinMaxScaler(), scale)


def make_tree(max_depth=None, min_samples_leaf=1, criterion="gini"):
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=int(min_samples_leaf),
                                  criterion=criterion, random_state=RANDOM_STATE)


def make_lda(solver="svd", shrinkage=None, scale=True):
    kwargs = {"solver": solver}
    if shrinkage is not None:
        kwargs["shrinkage"] = shrinkage
    return _wrap(LinearDiscriminantAnalysis(**kwargs), StandardScaler(), scale)


def make_svm(C=1.0, kernel="rbf", gamma="scale", class_weight=None, probability=False, scale=True):
    clf = SVC(C=float(C), kernel=kernel, gamma="scale" if kernel == "linear" else gamma,
              class_weight=class_weight, probability=probability, random_state=RANDOM_STATE)
    return _wrap(clf, StandardScaler(), scale)


def make_knn(n_neighbors=5, weights="uniform", metric="euclidean", scale=True):
    clf = KNeighborsClassifier(n_neighbors=int(n_neighbors), weights=weights, metric=metric)
    return _wrap(clf, StandardScaler(), scale)


FACTORIES = {
    "gaussian": make_gaussian, "multinomial": make_multinomial, "complement": make_complement,
    "bernoulli": make_bernoulli, "tree": make_tree, "lda": make_lda, "svm": make_svm, "knn": make_knn,
}


def build(cfg, probability=False):
    params = dict(cfg["params"])
    if cfg["factory"] == "svm":
        params["probability"] = probability
    return FACTORIES[cfg["factory"]](**params)


def remember(name, factory, params, params_text, selected, default_auc, default_text):
    best_configs[name] = {
        "factory": factory, "params": params, "params_text": params_text,
        "cv_roc_auc": float(selected["roc_auc"]), "cv_roc_auc_std": float(selected["roc_auc_std"]),
    }
    tuning_summary[name] = {
        "Метод": name,
        "Конфигурация по умолчанию": default_text,
        "AUC по умолчанию": float(default_auc),
        "Выбранная конфигурация": params_text,
        "AUC после выбора": float(selected["roc_auc"]),
        "Прирост AUC": float(selected["roc_auc"] - default_auc),
    }
    gap = float(selected["roc_auc_train"] - selected["roc_auc"])
    print(f"Выбрано [{name}]: {params_text}")
    print(f"  CV AUC {selected['roc_auc']:.4f}±{selected['roc_auc_std']:.4f}, F1 {selected['f1']:.4f}, "
          f"P {selected['precision']:.4f}, R {selected['recall']:.4f}, Acc {selected['accuracy']:.4f}, "
          f"разрыв train-val {gap:.4f}; база {default_auc:.4f}, прирост {selected['roc_auc'] - default_auc:+.4f}")


def best_nb(frame):
    """Лучшая точка в каждой ветке масштаба, затем лучшая среди веток."""
    candidates = [select_best(sub.copy()) for _, sub in frame.groupby("scale")]
    return select_best(pd.DataFrame(candidates).infer_objects())


h("## 5. Настройка методов классификации")

# --- 5.1 GaussianNB ---
h("### 5.1. GaussianNB")
gaussian_df = sweep(
    make_gaussian,
    [{"var_smoothing": float(v), "scale": s} for s in (False, True) for v in np.logspace(-12, 0, 13)],
    label=lambda kw: format_param(kw["var_smoothing"]),
    extra=lambda kw: {"group": "StandardScaler" if kw["scale"] else "без масштабирования"},
)
plot_lines(gaussian_df, "group", ["StandardScaler", "без масштабирования"], "var_smoothing",
           "GaussianNB: AUC-ROC при сглаживании дисперсии")
scaled = gaussian_df[gaussian_df["scale"]].copy()
selected = select_best(scaled)
default_row = scaled.loc[(scaled["var_smoothing"] - 1e-9).abs().idxmin()]
raw_default = gaussian_df[(~gaussian_df["scale"]) & ((gaussian_df["var_smoothing"] - 1e-9).abs() < 1e-15)].iloc[0]
remember("GaussianNB", "gaussian", {"var_smoothing": float(selected["var_smoothing"]), "scale": True},
         f"StandardScaler, var_smoothing={format_param(selected['var_smoothing'])}",
         selected, default_row["roc_auc"], "StandardScaler, var_smoothing=1e-9")
print(f"  без масштабирования при 1e-9: AUC {raw_default['roc_auc']:.4f}")

# --- 5.2 MultinomialNB / 5.3 ComplementNB ---
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def nb_alpha_sweep(make, title, key, name, fig_title):
    frame = sweep(make, [{"alpha": a, "scale": s} for s in (False, True) for a in alphas],
                  label=lambda kw: format_param(kw["alpha"]),
                  extra=lambda kw: {"group": "MinMaxScaler" if kw["scale"] else "исходный масштаб"})
    plot_lines(frame, "group", ["MinMaxScaler", "исходный масштаб"], "alpha", fig_title)
    selected = best_nb(frame)
    use_scale = bool(selected["scale"])
    default_row = at(frame, scale=True, alpha=1.0)
    raw_at_1 = at(frame, scale=False, alpha=1.0)
    scale_name = "MinMaxScaler" if use_scale else "исходный масштаб"
    remember(name, key, {"alpha": float(selected["alpha"]), "scale": use_scale},
             f"{scale_name}, alpha={format_param(selected['alpha'])}",
             selected, default_row["roc_auc"], "MinMaxScaler, alpha=1")
    print(f"  при alpha=1: исходный масштаб {raw_at_1['roc_auc']:.4f}, MinMax {default_row['roc_auc']:.4f}")
    return frame


h("### 5.2. MultinomialNB")
multi_df = nb_alpha_sweep(make_multinomial, "5.2", "multinomial", "MultinomialNB",
                          "MultinomialNB: AUC-ROC при сглаживании")
h("### 5.3. ComplementNB")
comp_df = nb_alpha_sweep(make_complement, "5.3", "complement", "ComplementNB",
                         "ComplementNB: AUC-ROC при сглаживании")

# --- 5.4 BernoulliNB ---
h("### 5.4. BernoulliNB")
thresholds = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
bern_df = sweep(
    make_bernoulli,
    [{"alpha": a, "threshold": t, "scale": True} for t in thresholds for a in alphas],
    label=lambda kw: f"порог={format_param(kw['threshold'])}, alpha={format_param(kw['alpha'])}",
)
selected = select_best(bern_df)

alpha_slice = bern_df[bern_df["alpha"] == 1.0].sort_values("threshold").copy()
fig, ax = plt.subplots(figsize=(8.2, 4.6))
x_thr = np.arange(len(alpha_slice))
ax.plot(x_thr, alpha_slice["roc_auc"], marker="o", color="#4C78A8", label="AUC-ROC")
ax.fill_between(x_thr, alpha_slice["roc_auc"] - alpha_slice["roc_auc_std"],
                alpha_slice["roc_auc"] + alpha_slice["roc_auc_std"], alpha=0.15, color="#4C78A8")
ax.set_xticks(x_thr)
ax.set_xticklabels(alpha_slice["threshold"].map(format_param))
ax.set_xlabel("Порог бинаризации после MinMax при alpha=1")
ax.set_ylabel("AUC-ROC")
ax.set_title("BernoulliNB: как порог 0/1 меняет качество")
fig.tight_layout()
save_show()

thr_best = float(selected["threshold"])
alpha_at_best = bern_df[np.isclose(bern_df["threshold"], thr_best)].sort_values("alpha")
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.plot(alpha_at_best["alpha"].map(format_param), alpha_at_best["roc_auc"], marker="o", color="#E45756")
ax.set_xlabel(f"alpha при пороге {format_param(thr_best)}")
ax.set_ylabel("AUC-ROC")
ax.set_title("BernoulliNB: сглаживание при выбранном пороге")
fig.tight_layout()
save_show()

raw_zero = evaluate_cv(make_bernoulli(alpha=1.0, threshold=0.0, scale=False), X_train, y_train)
default_row = bern_df[np.isclose(bern_df["threshold"], 0.5) & np.isclose(bern_df["alpha"], 1.0)].iloc[0]
remember("BernoulliNB", "bernoulli", {"alpha": float(selected["alpha"]), "threshold": thr_best, "scale": True},
         f"MinMaxScaler, порог={format_param(thr_best)}, alpha={format_param(selected['alpha'])}",
         selected, default_row["roc_auc"], "MinMaxScaler, порог=0.5, alpha=1")
print(f"  без MinMax (порог 0, alpha=1): AUC {raw_zero['roc_auc']:.4f}")

# --- 5.5 Дерево решений ---
h("### 5.5. Дерево решений")
depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, None]
tree_df = sweep(
    make_tree,
    [{"max_depth": d, "min_samples_leaf": leaf, "criterion": c}
     for d in depths for leaf in [1, 5, 10, 20] for c in ["gini", "entropy"]],
    label=lambda kw: f"depth={format_param(kw['max_depth'])}, leaf={kw['min_samples_leaf']}, {kw['criterion']}",
    extra=lambda kw: {"max_depth": kw["max_depth"] if kw["max_depth"] is not None else -1},
)
selected = select_best(tree_df)
slice_df = tree_df[(tree_df["min_samples_leaf"] == 1) & (tree_df["criterion"] == "gini")].sort_values("max_depth")
fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(len(slice_df))
ax.plot(x, slice_df["roc_auc_train"], marker="s", label="AUC-ROC, обучение фолда")
ax.plot(x, slice_df["roc_auc"], marker="o", label="AUC-ROC, проверка")
ax.set_xticks(x)
ax.set_xticklabels(slice_df["max_depth"].map(lambda v: "без ограничения" if v < 0 else str(int(v))))
ax.set_xlabel("max_depth при min_samples_leaf=1 и критерии Джини")
ax.set_ylabel("AUC-ROC")
ax.set_title("Дерево: недообучение и запоминание выборки")
ax.legend()
fig.tight_layout()
save_show()

default_row = at(tree_df, max_depth=-1, min_samples_leaf=1, criterion="gini")
params = {
    "max_depth": None if int(selected["max_depth"]) < 0 else int(selected["max_depth"]),
    "min_samples_leaf": int(selected["min_samples_leaf"]),
    "criterion": selected["criterion"],
}
remember("Дерево", "tree", params, selected["param_label"], selected,
         default_row["roc_auc"], "depth=None, leaf=1, gini")
tree_importance = pd.Series(make_tree(**params).fit(X_train, y_train).feature_importances_,
                            index=feature_cols).sort_values(ascending=False)
print("  важность признаков (топ-3):", ", ".join(f"{k} {v:.2f}" for k, v in tree_importance.head(3).items()))

# --- 5.6 LDA ---
h("### 5.6. LDA")
lda_combos = [{"solver": "lsqr", "shrinkage": s} for s in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, "auto"]]
lda_combos.append({"solver": "svd", "shrinkage": None})
lda_df = sweep(
    make_lda, lda_combos,
    label=lambda kw: (f"lsqr, shrinkage={format_param(kw['shrinkage'])}" if kw["solver"] == "lsqr"
                      else "svd, без shrinkage"),
)
selected = select_best(lda_df)
svd_scaled = lda_df[lda_df["solver"] == "svd"].iloc[0]
svd_raw = evaluate_cv(make_lda("svd", None, scale=False), X_train, y_train)

plot_df_lda = lda_df[lda_df["solver"] == "lsqr"].copy()
fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(len(plot_df_lda))
ax.plot(x, plot_df_lda["roc_auc"], marker="o", color="#4C78A8", label="lsqr + shrinkage")
ax.fill_between(x, plot_df_lda["roc_auc"] - plot_df_lda["roc_auc_std"],
                plot_df_lda["roc_auc"] + plot_df_lda["roc_auc_std"], alpha=0.15, color="#4C78A8")
ax.axhline(svd_scaled["roc_auc"], color="#E45756", linestyle="--",
           label=f"svd без сжатия ({svd_scaled['roc_auc']:.3f})")
ax.set_xticks(x)
ax.set_xticklabels([format_param(v) for v in plot_df_lda["shrinkage"]], rotation=45, ha="right")
ax.set_xlabel("shrinkage")
ax.set_ylabel("AUC-ROC")
ax.set_title("LDA: сжатие ковариации на стандартизованных признаках")
ax.legend()
fig.tight_layout()
save_show()

if selected["solver"] == "svd":
    params = {"solver": "svd", "shrinkage": None, "scale": True}
else:
    shrink_param = selected["shrinkage"]
    params = {"solver": "lsqr", "shrinkage": "auto" if isinstance(shrink_param, str) else float(shrink_param),
              "scale": True}
remember("LDA", "lda", params, selected["param_label"] + ", StandardScaler", selected,
         svd_scaled["roc_auc"], "svd, без shrinkage, StandardScaler")
print(f"  svd без стандартизации: AUC {svd_raw['roc_auc']:.4f}")

# --- 5.7 SVM ---
h("### 5.7. SVM (линейное и RBF-ядро)")
gammas = [0.01, 0.1, 1.0, "scale"]
svm_combos = []
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    for cw in (None, "balanced"):
        svm_combos.append({"C": C, "kernel": "linear", "gamma": "scale", "class_weight": cw})
        svm_combos += [{"C": C, "kernel": "rbf", "gamma": g, "class_weight": cw} for g in gammas]


def svm_label(kw):
    weight = "balanced" if kw["class_weight"] == "balanced" else "обычные"
    if kw["kernel"] == "linear":
        return f"linear, C={format_param(kw['C'])}, веса={weight}"
    return f"rbf, C={format_param(kw['C'])}, gamma={format_param(kw['gamma'])}, веса={weight}"


svm_df = sweep(
    make_svm, svm_combos, label=svm_label,
    extra=lambda kw: {
        "gamma": "—" if kw["kernel"] == "linear" else format_param(kw["gamma"]),
        "class_weight": "balanced" if kw["class_weight"] == "balanced" else "обычные",
    },
)
selected = select_best(svm_df)

linear = svm_df[svm_df["kernel"] == "linear"]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
for weight, sub in linear.groupby("class_weight"):
    sub = sub.sort_values("C")
    ax.plot(sub["C"].map(format_param), sub["roc_auc"], marker="o", label=f"веса: {weight}")
ax.set_xlabel("C, линейное ядро")
ax.set_ylabel("AUC-ROC")
ax.set_title("Линейный SVM: плата за ошибку и баланс классов")
ax.legend()
fig.tight_layout()
save_show()

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
for ax, weight in zip(axes, ["обычные", "balanced"]):
    sub = svm_df[(svm_df["kernel"] == "rbf") & (svm_df["class_weight"] == weight)]
    pivot = sub.pivot(index="gamma", columns="C", values="roc_auc")
    pivot.columns = [format_param(v) for v in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=float(svm_df["roc_auc"].min()))
    ax.set_title(f"RBF, веса: {weight}")
    ax.set_xlabel("C")
    ax.set_ylabel("gamma")
fig.suptitle("SVM с ядром RBF: AUC-ROC на кросс-валидации", y=1.03)
fig.tight_layout()
save_show()

raw_svm = evaluate_cv(make_svm(C=1.0, kernel="rbf", gamma="scale", class_weight=None, scale=False),
                      X_train, y_train)
default_row = at(svm_df, kernel="rbf", C=1.0, gamma="scale", class_weight="обычные")
gamma_value = selected["gamma"]
if gamma_value not in ("—", "scale"):
    gamma_value = float(gamma_value)
params = {
    "C": float(selected["C"]),
    "kernel": selected["kernel"],
    "gamma": "scale" if selected["kernel"] == "linear" else gamma_value,
    "class_weight": None if selected["class_weight"] == "обычные" else "balanced",
    "scale": True,
}
remember("SVM", "svm", params, selected["param_label"] + ", StandardScaler", selected,
         default_row["roc_auc"], "StandardScaler, rbf, C=1, gamma=scale, обычные веса")
best_linear = select_best(linear.copy())
best_rbf = select_best(svm_df[svm_df["kernel"] == "rbf"].copy())
print(f"  лучший linear {best_linear['roc_auc']:.4f}, лучший rbf {best_rbf['roc_auc']:.4f}; "
      f"rbf без стандартизации {raw_svm['roc_auc']:.4f}")

# --- 5.8 kNN ---
h("### 5.8. kNN")
knn_df = sweep(
    make_knn,
    [{"n_neighbors": k, "weights": w, "metric": m}
     for k in range(1, 26) for w in ("uniform", "distance") for m in ("euclidean", "manhattan")],
    label=lambda kw: f"k={kw['n_neighbors']}, {kw['weights']}, {kw['metric']}",
)
selected = select_best(knn_df)

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
for weights, metric in (("uniform", "euclidean"), ("distance", "euclidean"),
                        ("uniform", "manhattan"), ("distance", "manhattan")):
    sub = knn_df[(knn_df["weights"] == weights) & (knn_df["metric"] == metric)].sort_values("n_neighbors")
    axes[0].plot(sub["n_neighbors"], sub["roc_auc"], marker="o", label=f"{weights}, {metric}")
    axes[1].plot(sub["n_neighbors"], sub["recall"], marker="o", label=f"{weights}, {metric}")
axes[0].set_title("AUC-ROC")
axes[1].set_title("Recall хорошего класса")
for ax in axes:
    ax.set_xlabel("Число соседей k")
    ax.legend(fontsize=8)
fig.suptitle("kNN на стандартизованных признаках", y=1.03)
fig.tight_layout()
save_show()

raw_knn = evaluate_cv(make_knn(n_neighbors=5, weights="uniform", metric="euclidean", scale=False),
                      X_train, y_train)
default_row = at(knn_df, n_neighbors=5, weights="uniform", metric="euclidean")
params = {"n_neighbors": int(selected["n_neighbors"]), "weights": selected["weights"],
          "metric": selected["metric"], "scale": True}
remember("kNN", "knn", params, selected["param_label"] + ", StandardScaler", selected,
         default_row["roc_auc"], "StandardScaler, k=5, uniform, euclidean")
print(f"  k=5 без стандартизации: AUC {raw_knn['roc_auc']:.4f}")

# ---------------------------------------------------------------------------
# Сводка настройки
# ---------------------------------------------------------------------------
order = ["GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB", "Дерево", "LDA", "SVM", "kNN"]
tuning_table = pd.DataFrame([tuning_summary[name] for name in order]).set_index("Метод")
h("### Сводка настройки гиперпараметров")
show(tuning_table.round(4))

fig, ax = plt.subplots(figsize=(8.4, 4.6))
gains = tuning_table["Прирост AUC"]
colors = ["#2CA02C" if v >= 0.01 else "#4C78A8" if v >= 0 else "#E45756" for v in gains]
ax.bar(gains.index, gains.values, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Прирост CV AUC-ROC")
ax.set_title("Что дала настройка относительно обычного старта")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
save_show()

# ---------------------------------------------------------------------------
# 6. Сравнение на тесте
# ---------------------------------------------------------------------------
h("## 6. Финальное сравнение методов на тесте")
test_rows, scores, cms = [], {}, {}
for name in order:
    model = build(best_configs[name], probability=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    scores[name] = score
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    cms[name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    test_rows.append({
        "Метод": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "F1-Score": f1_score(y_test, pred, pos_label=1, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, score),
        "CV AUC-ROC": best_configs[name]["cv_roc_auc"],
    })

report = pd.DataFrame(test_rows).set_index("Метод").loc[order]
report_sorted = report.sort_values(["AUC-ROC", "Recall", "F1-Score"], ascending=False)
h("### Метрики на тестовой выборке (по AUC-ROC, затем Recall и F1)")
show(report_sorted.round(4))

params_view = pd.DataFrame({"Параметры": [best_configs[nm]["params_text"] for nm in report_sorted.index]},
                           index=report_sorted.index)
show(params_view)

# График 15. ROC-кривые
fig, ax = plt.subplots(figsize=(8.2, 6.2))
for color, name in zip(sns.color_palette("tab10", n_colors=len(report_sorted)), report_sorted.index):
    fpr, tpr, _ = roc_curve(y_test, scores[name])
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} ({report_sorted.loc[name, 'AUC-ROC']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Случайное ранжирование")
ax.set_xlabel("Доля ложных тревог (1 − специфичность)")
ax.set_ylabel("Recall хорошего класса")
ax.set_title("ROC-кривые на тестовой выборке")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save_show()

# График 16. Столбчатые метрики
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
for ax, metric in zip(axes.ravel(), ["Accuracy", "Precision", "Recall", "F1-Score"]):
    values = report.loc[order, metric]
    ax.bar(np.arange(len(order)), values.values, color=sns.color_palette("tab10", n_colors=len(order)))
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_title(metric)
    for i, value in enumerate(values.values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Тестовые метрики. Ось начинается с нуля, чтобы не преувеличивать разрыв", y=1.02)
fig.tight_layout()
save_show()


def draw_cm(ax, matrix, title):
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                xticklabels=[f"прогноз {NEG_NAME}", f"прогноз {POS_NAME}"],
                yticklabels=[f"истина {NEG_NAME}", f"истина {POS_NAME}"])
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def cm_matrix(name):
    c = cms[name]
    return np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])


# График 17. Матрицы ошибок
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
for ax, name in zip(axes.ravel(), order):
    draw_cm(ax, cm_matrix(name), name)
fig.suptitle("Матрицы ошибок на тесте. Нижняя левая клетка — пропущенные хорошие вина", y=1.03)
fig.tight_layout()
save_show()

# Рекомендация: лидер по AUC; в полосе AUC 0.01 лучший F1 (затем Recall, AUC);
# если он сложный (SVM-rbf/kNN/дерево) и простая модель в полосе уступает по F1 <= 0.01 — берём простую.
leader_name = report_sorted.index[0]
leader = report_sorted.iloc[0]
close = report[report["AUC-ROC"] >= leader["AUC-ROC"] - 0.01].copy()
practical = close.sort_values(["F1-Score", "Recall", "AUC-ROC"], ascending=False)
practical_name = practical.index[0]

simplicity = {
    "LDA": 0, "GaussianNB": 1, "SVM": 2 if best_configs["SVM"]["params"]["kernel"] == "linear" else 5,
    "MultinomialNB": 3, "ComplementNB": 3, "BernoulliNB": 4, "kNN": 6, "Дерево": 6,
}
simple_pool = close.copy()
simple_pool["простота"] = [simplicity[nm] for nm in simple_pool.index]
simple_name = simple_pool.sort_values(["простота", "F1-Score", "AUC-ROC"],
                                      ascending=[True, False, False]).index[0]

recommend_name = practical_name
reason = "лучший F1 в верхней группе"
if simple_name != practical_name and simplicity[practical_name] >= 5:
    f1_gap_simple = float(report.loc[practical_name, "F1-Score"] - report.loc[simple_name, "F1-Score"])
    auc_gap_simple = abs(float(report.loc[simple_name, "AUC-ROC"] - report.loc[practical_name, "AUC-ROC"]))
    if f1_gap_simple <= 0.01 and auc_gap_simple <= 0.01:
        recommend_name = simple_name
        reason = f"проще {practical_name}, F1 отстаёт не более чем на 0.01"

cm = cms[recommend_name]
print(f"Лидер по AUC: {leader_name} ({leader['AUC-ROC']:.4f}); "
      f"полоса 0.01: {', '.join(close.sort_values('AUC-ROC', ascending=False).index)}")
print(f"Лучший F1 в полосе: {practical_name}; самый простой: {simple_name}")
print(f"Рекомендация: {recommend_name} ({reason}), параметры: {best_configs[recommend_name]['params_text']}")
print("  тест: " + ", ".join(f"{k} {report.loc[recommend_name, k]:.4f}" for k in report.columns[:5]))
print(f"  tp {cm['tp']}, fn {cm['fn']} из {n_pos_test}, fp {cm['fp']}, tn {cm['tn']}")

report_sorted.round(4).to_csv(RESULTS / "метрики_тест.csv", encoding="utf-8-sig")
tuning_table.round(4).to_csv(RESULTS / "настройка_гиперпараметров.csv", encoding="utf-8-sig")
split_table.round(4).to_csv(RESULTS / "разбиение_выборки.csv", encoding="utf-8-sig", index=False)
params_view.to_csv(RESULTS / "параметры_моделей.csv", encoding="utf-8-sig")
effect.round(4).to_csv(RESULTS / "эффект_признаков.csv", encoding="utf-8-sig", index=False)
print("Таблицы сохранены в", RESULTS)

# ---------------------------------------------------------------------------
# 7. Нейронная сеть (Keras Sequential MLP + TensorBoard)
# ---------------------------------------------------------------------------
h("## 7. Нейронная сеть для классификации вина")
try:
    import tensorflow as tf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as error:
    print("TensorFlow/TensorBoard не установлены:", error)
    print("Установите зависимости: pip install -r requirements.txt")
    print("Классическая часть работы выполнена, таблицы и графики сохранены.")
    raise SystemExit(0)

# Журнал только в ASCII-пути (кириллица в профиле пользователя ломает event-файлы).
LOG_ROOT = Path(os.environ.get("SystemDrive", "C:") + os.sep) / "lr2_wine_tensorboard"
tf.keras.utils.set_random_seed(RANDOM_STATE)
tf.get_logger().setLevel("ERROR")

# Проверочная часть из обучения для подбора архитектуры; тест не используется.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)
nn_scaler = StandardScaler()
X_fit_s = nn_scaler.fit_transform(X_fit)
X_val_s = nn_scaler.transform(X_val)
y_fit_f = np.asarray(y_fit).astype(np.float32)
y_val_f = np.asarray(y_val).astype(np.float32)
y_val_i = np.asarray(y_val).astype(int)

NN_BASE = {"units": 64, "learning_rate": 1e-3, "dropout": 0.2, "n_layers": 2,
           "activation": "relu", "batch_size": 32, "epochs": 50}


def class_weight_of(target):
    values = np.asarray(target).astype(int)
    return {label: float(len(values) / (2.0 * np.sum(values == label))) for label in (0, 1)}


def merge_cfg(**cfg):
    p = {**NN_BASE, **cfg}
    for key in ("units", "n_layers", "batch_size", "epochs"):
        p[key] = int(p[key])
    p["learning_rate"] = float(p["learning_rate"])
    p["dropout"] = float(p["dropout"])
    p["activation"] = str(p["activation"])
    return p


def run_name(p):
    return (f"L{p['n_layers']}_U{p['units']}_{p['activation']}_lr{p['learning_rate']:.0e}"
            f"_d{p['dropout']:.1f}_bs{p['batch_size']}_ep{p['epochs']}")


def build_mlp(n_feat, units, dropout, learning_rate, n_layers=2, activation="relu"):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    layers = [tf.keras.layers.Input(shape=(n_feat,))]
    width = int(units)
    for i in range(int(n_layers)):
        layers.append(tf.keras.layers.Dense(width, activation=activation))
        if i == 0:
            layers.append(tf.keras.layers.Dropout(float(dropout)))
        width = max(width // 2, 8)
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_from(p, n_feat):
    return build_mlp(n_feat, p["units"], p["dropout"], p["learning_rate"],
                     n_layers=p["n_layers"], activation=p["activation"])


def tb_callback(log_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0,
                                          write_graph=True, update_freq="epoch")


def read_tb(log_dir, split, tag):
    acc = EventAccumulator(str(Path(log_dir) / split), size_guidance={"tensors": 0})
    acc.Reload()
    return [float(tf.make_ndarray(e.tensor_proto)) for e in acc.Tensors(tag)]


def train_run(**cfg):
    p = merge_cfg(**cfg)
    name = run_name(p)
    log_dir = LOG_ROOT / name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    model = build_from(p, X_fit_s.shape[1])
    model.fit(X_fit_s, y_fit_f, validation_data=(X_val_s, y_val_f), epochs=p["epochs"],
              batch_size=p["batch_size"], class_weight=class_weight_of(y_fit),
              callbacks=[tb_callback(log_dir)], verbose=0)
    proba = model.predict(X_val_s, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    tl, ta = read_tb(log_dir, "train", "epoch_loss"), read_tb(log_dir, "train", "epoch_accuracy")
    vl, va = read_tb(log_dir, "validation", "epoch_loss"), read_tb(log_dir, "validation", "epoch_accuracy")
    row = {
        "Прогон": name, **p,
        "val_auc": float(roc_auc_score(y_val_i, proba)),
        "val_recall": float(recall_score(y_val_i, pred, zero_division=0)),
        "loss": float(tl[-1]), "accuracy": float(ta[-1]),
        "val_loss": float(vl[-1]), "val_accuracy": float(va[-1]),
        "gap_acc": float(ta[-1] - va[-1]),
        "min_val_loss": float(np.min(vl)), "min_val_epoch": int(np.argmin(vl) + 1),
        "val_loss_first": float(vl[0]), "val_loss_last": float(vl[-1]),
    }
    print(f"{name}: val loss {row['val_loss']:.4f}, val accuracy {row['val_accuracy']:.4f}, AUC {row['val_auc']:.4f}")
    return row


def pick_row(rows):
    frame = pd.DataFrame(rows)
    pool = frame[frame["val_auc"] >= frame["val_auc"].max() - 0.005].copy()
    return pool.sort_values(["val_loss", "gap_acc"], ascending=[True, True]).iloc[0]


def sweep_table(rows, column, title):
    frame = pd.DataFrame(rows)
    shown = pd.DataFrame({
        column: frame[column].map(format_param) if column == "learning_rate" else frame[column],
        "AUC проверки": frame["val_auc"], "validation loss": frame["val_loss"],
        "validation accuracy": frame["val_accuracy"], "Разрыв accuracy": frame["gap_acc"],
        "Эпоха мин. val loss": frame["min_val_epoch"],
    })
    h(title)
    show(shown.round(4))


def plot_tb_metrics(rows, label_of, title):
    """Кривые из event-файлов TensorBoard -> nn_tb_*.png."""
    panels = (("train", "epoch_loss", "loss"), ("train", "epoch_accuracy", "accuracy"),
              ("validation", "epoch_loss", "validation loss"),
              ("validation", "epoch_accuracy", "validation accuracy"))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    for row in rows:
        log_dir = LOG_ROOT / row["Прогон"]
        label = label_of(row)
        for ax, (split, tag, ylabel) in zip(axes.ravel(), panels):
            values = read_tb(log_dir, split, tag)
            ax.plot(np.arange(1, len(values) + 1), values, marker="o", markersize=2.5, label=label)
            ax.set_xlabel("Эпоха")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
    for ax in axes.ravel():
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_show_nn()


if LOG_ROOT.exists():
    shutil.rmtree(LOG_ROOT)
LOG_ROOT.mkdir(parents=True)

print(f"MLP: вход {n_features}, скрытые слои (ширина вдвое меньше, не менее 8), dropout после первого, "
      f"sigmoid; binary_crossentropy, Adam; веса классов при подборе; проверка 20% обучения.")
print("Жадный подбор: lr -> units -> layers -> activation -> batch -> epochs -> dropout.")

nn_cache = {}


def cached_run(**cfg):
    p = merge_cfg(**cfg)
    key = tuple(sorted(p.items()))
    if key not in nn_cache:
        nn_cache[key] = train_run(**p)
    return nn_cache[key]


# (параметр, значения, подпись кривой, заголовок рисунка)
NN_SWEEPS = [
    ("learning_rate", (1e-4, 1e-3, 1e-2), lambda r: f"lr={format_param(r['learning_rate'])}",
     "TensorBoard: loss и accuracy при трёх скоростях обучения"),
    ("units", (16, 64, 128), lambda r: f"{int(r['units'])} нейронов",
     "TensorBoard: число нейронов в первом слое"),
    ("n_layers", (1, 2, 3), lambda r: f"{int(r['n_layers'])} скрытых слоя",
     "TensorBoard: число скрытых слоёв"),
    ("activation", ("relu", "tanh", "elu"), lambda r: str(r["activation"]),
     "TensorBoard: функции активации скрытых слоёв"),
    ("batch_size", (16, 32, 64), lambda r: f"batch {int(r['batch_size'])}",
     "TensorBoard: размер батча"),
    ("epochs", (20, 50, 100), lambda r: f"{int(r['epochs'])} эпох",
     "TensorBoard: число эпох"),
    ("dropout", (0.0, 0.2, 0.5), lambda r: f"dropout {format_param(r['dropout'])}",
     "TensorBoard: dropout"),
]
current = dict(NN_BASE)
for param, values, label_of, fig_title in NN_SWEEPS:
    rows = [cached_run(**{**current, param: v}) for v in values]
    best_row = pick_row(rows)
    current[param] = best_row[param]
    sweep_table(rows, param, f"### Подбор: {param}")
    plot_tb_metrics(rows, label_of, fig_title)
    print(f"Выбрано {param} = {format_param(best_row[param])}")
chosen_nn = best_row  # последний шаг (dropout) несёт все выбранные параметры
nn_choice = {k: merge_cfg(**current)[k]
             for k in ("units", "learning_rate", "dropout", "n_layers", "activation", "batch_size", "epochs")}

print("Выбрано:", nn_choice)
print(f"Проверочный AUC {float(chosen_nn['val_auc']):.4f}, Recall {float(chosen_nn['val_recall']):.4f}, "
      f"разрыв accuracy {float(chosen_nn['gap_acc']):.4f}, мин. val loss на эпохе "
      f"{int(chosen_nn['min_val_epoch'])} из {int(chosen_nn['epochs'])}; конфигураций: {len(nn_cache)}")
print("Журнал TensorBoard:", LOG_ROOT)
print(f'tensorboard --logdir "{LOG_ROOT}"')

nn_search = pd.DataFrame(list(nn_cache.values()))
nn_search.round(4).to_csv(RESULTS / "нейросеть_подбор.csv", encoding="utf-8-sig", index=False)

# --- Финальное обучение на всём X_train ---
h("### Финальное обучение выбранной сети на всей обучающей выборке")
nn_full_scaler = StandardScaler()
X_train_s = nn_full_scaler.fit_transform(X_train)
X_test_s = nn_full_scaler.transform(X_test)
build_from(nn_choice, X_train_s.shape[1]).summary()


def fit_final(class_weight, log_name):
    log_dir = LOG_ROOT / log_name
    if log_dir.exists():
        shutil.rmtree(log_dir)
    model = build_from(nn_choice, X_train_s.shape[1])
    kwargs = dict(epochs=nn_choice["epochs"], batch_size=nn_choice["batch_size"], verbose=0,
                  callbacks=[tb_callback(log_dir)])
    if class_weight is not None:
        kwargs["class_weight"] = class_weight
    model.fit(X_train_s, np.asarray(y_train).astype(np.float32), **kwargs)
    return model


def score_test(model):
    proba = model.predict(X_test_s, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    y_true = np.asarray(y_test).astype(int)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "F1-Score": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "AUC-ROC": float(roc_auc_score(y_true, proba)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "matrix": matrix, "proba": proba,
    }


balanced_test = score_test(fit_final(class_weight_of(y_train), "final_balanced"))
plain_test = score_test(fit_final(None, "final_plain"))

metric_cols = ("Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC")
compare_rows = {
    "Сеть, веса классов": {m: balanced_test[m] for m in metric_cols},
    "Сеть, без весов": {m: plain_test[m] for m in metric_cols},
    f"{leader_name} (лучший классический по AUC)": {m: float(report.loc[leader_name, m]) for m in metric_cols},
}
if recommend_name != leader_name:
    compare_rows[f"{recommend_name} (рекомендованный)"] = {m: float(report.loc[recommend_name, m]) for m in metric_cols}
compare = pd.DataFrame(compare_rows).T
h("### Тест: нейронная сеть и лучший классический метод")
show(compare.round(4))
compare.round(4).to_csv(RESULTS / "нейросеть_тест.csv", encoding="utf-8-sig")

best_cls_name = recommend_name
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
draw_cm(axes[0], balanced_test["matrix"], "Сеть с весами классов")
draw_cm(axes[1], plain_test["matrix"], "Сеть без весов")
draw_cm(axes[2], cm_matrix(best_cls_name), f"{best_cls_name} (классика)")
fig.suptitle("Матрицы ошибок на том же тесте, что и в разделе 6", y=1.05)
fig.tight_layout()
save_show_nn()

fig, ax = plt.subplots(figsize=(7.4, 5.6))
for proba_values, label, color in (
    (balanced_test["proba"], f"Сеть, веса ({balanced_test['AUC-ROC']:.3f})", "#E45756"),
    (plain_test["proba"], f"Сеть, без весов ({plain_test['AUC-ROC']:.3f})", "#F58518"),
    (scores[best_cls_name], f"{best_cls_name} ({report.loc[best_cls_name, 'AUC-ROC']:.3f})", "#4C78A8"),
):
    fpr, tpr, _ = roc_curve(np.asarray(y_test).astype(int), proba_values)
    ax.plot(fpr, tpr, linewidth=2, label=label, color=color)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Случайное ранжирование")
ax.set_xlabel("Доля ложных тревог (1 − специфичность)")
ax.set_ylabel("Recall хорошего класса")
ax.set_title("ROC на тесте: нейронная сеть и классический метод")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save_show_nn()

for label, res in (("Сеть с весами", balanced_test), ("Сеть без весов", plain_test)):
    print(f"{label}: верно хороших {res['tp']}, пропусков {res['fn']}, "
          f"ложных тревог {res['fp']}, верно обычных {res['tn']}.")

best_nn_label = ("Сеть, веса классов"
                 if (balanced_test["AUC-ROC"], balanced_test["F1-Score"]) >= (plain_test["AUC-ROC"], plain_test["F1-Score"])
                 else "Сеть, без весов")
best_nn_test = balanced_test if best_nn_label == "Сеть, веса классов" else plain_test
cls_ref = report.loc[best_cls_name]
d_auc = float(best_nn_test["AUC-ROC"] - cls_ref["AUC-ROC"])
d_f1 = float(best_nn_test["F1-Score"] - cls_ref["F1-Score"])
if abs(d_auc) < 0.01 and abs(d_f1) < 0.02:
    verdict = "практически равны"
elif d_auc > 0 and d_f1 > 0:
    verdict = "сеть лучше по AUC и F1"
elif d_auc < 0 and d_f1 < 0:
    verdict = "сеть уступает по AUC и F1"
else:
    verdict = "неоднозначно (AUC и F1 расходятся)"
print(f"Лучшая сеть ({best_nn_label}) против {best_cls_name}: dAUC {d_auc:+.4f}, dF1 {d_f1:+.4f} — {verdict}.")
print("Готово: графики сохранены в", FIGURES, ", таблицы — в", RESULTS)
