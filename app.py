r"""Gradio demo for the Wildfire binary classifiers in EXPERIMENT.

Run from the repository root:
    .venv\Scripts\python.exe app_gradio.py

The app loads the final Exp2 checkpoint for each selected run.  It intentionally
does not retrain models and uses the same 224px resize and optional LAB-CLAHE
preprocessing as the experiment notebook.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "EXPERIMENT"
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _available_runs() -> list[str]:
    runs = []
    for path in sorted(EXPERIMENT_DIR.glob("*_img224_seed*")):
        if (path / "exp2_best.weights.h5").is_file() and (
            path / "config.csv"
        ).is_file():
            runs.append(path.name)
    return runs


RUNS = _available_runs()
if not RUNS:
    raise RuntimeError(f"Tidak ada checkpoint Exp2 di {EXPERIMENT_DIR}")


def _run_config(run_name: str) -> dict[str, Any]:
    config = pd.read_csv(EXPERIMENT_DIR / run_name / "config.csv").iloc[0].to_dict()
    return config


def _arch_from_run(run_name: str) -> str:
    return run_name.rsplit("_img224_seed", 1)[0]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _backbone_arch(config: dict[str, Any], run_name: str) -> str:
    """Return the Keras backbone key from old and new config formats."""
    arch = str(config.get("arch") or _arch_from_run(run_name).removesuffix("_clahe"))
    if arch.startswith("mnv3l"):
        return "mnv3l"
    if arch.startswith("mnv3s"):
        return "mnv3s"
    if arch.startswith("mobilenetv2"):
        return "mobilenetv2"
    raise ValueError(
        f"Arsitektur backbone tidak dikenali pada run {run_name!r}: {arch!r}"
    )


def _format_run(run_name: str) -> str:
    return f"{_arch_from_run(run_name)} | seed {_run_config(run_name).get('seed', '?')}"


def _import_runtime() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Dependensi inference belum terpasang. Jalankan "
            "`python -m pip install -r requirements-app.txt`."
        ) from exc
    return cv2, tf, tf.keras.layers


def _preprocess(image: np.ndarray, use_clahe: bool, cv2: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Gambar belum diberikan.")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Gambar harus RGB/RGBA.")
    rgb = image[:, :, :3].astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    if use_clahe:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        bgr = cv2.cvtColor(
            cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _build_model(run_name: str, tf: Any, layers: Any) -> Any:
    config = _run_config(run_name)
    arch = _backbone_arch(config, run_name)
    head = str(config["head"])
    backbone_name = {
        "mobilenetv2": "MobileNetV2",
        "mnv3s": "MobileNetV3Small",
        "mnv3l": "MobileNetV3Large",
    }[arch]
    backbone = getattr(tf.keras.applications, backbone_name)(
        include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights=None
    )

    def cba(x: Any, channels: int, kernel: int, name: str) -> Any:
        x = layers.Conv2D(
            channels, kernel, padding="same", use_bias=False, name=f"{name}_conv"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        return layers.Activation("hard_swish", name=f"{name}_act")(x)

    def stride_tap(stride: int) -> Any:
        candidates = [
            layer
            for layer in backbone.layers
            if isinstance(layer, layers.Add)
            and layer.output.shape[1] == IMAGE_SIZE // stride
        ]
        if not candidates:
            raise RuntimeError(f"Tap stride-{stride} tidak ditemukan pada {arch}.")
        return candidates[-1].output

    def top_down(fine: Any, coarse: Any, name: str) -> Any:
        fuse_dim = int(config.get("fuse_dim", 64))
        projected_fine = cba(fine, fuse_dim, 1, f"{name}_pf")
        projected_coarse = cba(coarse, fuse_dim, 1, f"{name}_pc")
        upsampled = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(
            projected_coarse
        )
        fused = layers.Add(name=f"{name}_add")([projected_fine, upsampled])
        smoothed = layers.DepthwiseConv2D(
            3, padding="same", use_bias=False, name=f"{name}_dw"
        )(fused)
        return cba(smoothed, fuse_dim, 1, f"{name}_smooth")

    input_tensor = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = input_tensor
    if arch == "mobilenetv2":
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)

    if head == "fuse3":
        c8, c16, c32 = tf.keras.Model(
            backbone.input,
            [stride_tap(8), stride_tap(16), backbone.output],
            name="backbone_multiscale",
        )(x, training=False)
    elif head in {"fuse", "fmlp"}:
        c16, c32 = tf.keras.Model(
            backbone.input,
            [stride_tap(16), backbone.output],
            name="backbone_multiscale",
        )(x, training=False)
        c8 = None
    else:
        c32 = backbone(x, training=False)
        c16 = c8 = None

    features = c32
    if head in {"fuse", "fmlp", "fuse3"}:
        features = top_down(c16, c32, "fuse16")
        if head == "fuse3":
            features = top_down(c8, features, "fuse8")
    pooled = layers.GlobalAveragePooling2D()(features)
    pooled = layers.Dropout(float(config.get("dropout", 0.3)))(pooled)
    if head in {"mlp", "fmlp"}:
        pooled = layers.Dense(
            int(config.get("mlp_hidden", 64)),
            activation="hard_swish",
            name="mlp_hidden",
        )(pooled)
    output = layers.Dense(1, activation="sigmoid", name="p_fire")(pooled)
    model = tf.keras.Model(input_tensor, output)
    model.load_weights(EXPERIMENT_DIR / run_name / "exp2_best.weights.h5")
    return model


_MODEL_CACHE: dict[str, Any] = {}


def _predict(run_name: str, processed: np.ndarray, tf: Any) -> tuple[float, float]:
    if run_name not in _MODEL_CACHE:
        _MODEL_CACHE[run_name] = _build_model(run_name, tf, tf.keras.layers)
    batch = np.expand_dims(processed, axis=0).astype(np.float32)
    started = time.perf_counter()
    probability = float(
        _MODEL_CACHE[run_name](batch, training=False).numpy().reshape(-1)[0]
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return probability, elapsed_ms


def _threshold(run_name: str) -> float:
    result_path = EXPERIMENT_DIR / run_name / "results.csv"
    results = pd.read_csv(result_path)
    selected = results[
        (results["exp"] == "Exp2")
        & (results["split"] == "test")
        & (results["thr_type"].astype(str) == "val")
    ]
    return float(selected.iloc[0]["thr"]) if not selected.empty else 0.5


def _image_stats(original: np.ndarray, processed: np.ndarray, cv2: Any) -> pd.DataFrame:
    gray = cv2.cvtColor(original[:, :, :3], cv2.COLOR_RGB2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1)
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    hsv = cv2.cvtColor(original[:, :, :3], cv2.COLOR_RGB2HSV)
    return pd.DataFrame(
        [
            ["Resolusi asli", f"{original.shape[1]} x {original.shape[0]} px"],
            ["Resolusi input", f"{processed.shape[1]} x {processed.shape[0]} px"],
            ["Rata-rata brightness", f"{gray.mean():.2f} / 255"],
            ["Kontras (std grayscale)", f"{gray.std():.2f}"],
            ["Saturasi rata-rata", f"{hsv[:, :, 1].mean():.2f} / 255"],
            ["Entropy histogram", f"{entropy:.3f} bit"],
            ["Mode warna", "RGB"],
        ],
        columns=["Statistik gambar", "Nilai"],
    )


def _benchmark() -> pd.DataFrame:
    paths = [
        (EXPERIMENT_DIR / "reports" / "summary_exp2_mean_sd.csv", "Tanpa CLAHE"),
        (EXPERIMENT_DIR / "reports" / "clahe_summary_exp2_mean_sd.csv", "Dengan CLAHE"),
    ]
    tables = []
    for path, preprocessing in paths:
        if path.is_file():
            table = pd.read_csv(path).rename(columns={"model": "Model"})
            table.insert(1, "Preprocessing", preprocessing)
            tables.append(table)
    if not tables:
        return pd.DataFrame(
            {"Info": ["File benchmark tidak ditemukan di EXPERIMENT/reports."]}
        )
    return pd.concat(tables, ignore_index=True).sort_values(["Model", "Preprocessing"])


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def classify(image: np.ndarray | None, selected_runs: list[str] | None):
    import gradio as gr

    if image is None:
        raise gr.Error("Upload satu gambar terlebih dahulu.")
    if not selected_runs:
        raise gr.Error("Pilih minimal satu model.")

    try:
        cv2, tf, _ = _import_runtime()
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    original = np.asarray(image)
    rows = []
    processed_preview = None
    image_stats = None
    for run_name in selected_runs:
        config = _run_config(run_name)
        use_clahe = _as_bool(config.get("clahe", False))
        try:
            processed = _preprocess(original, use_clahe, cv2)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        probability, latency_ms = _predict(run_name, processed, tf)
        threshold = _threshold(run_name)
        rows.append(
            {
                "Model": _format_run(run_name),
                "Prediksi": "FIRE" if probability >= threshold else "NO FIRE",
                "Probabilitas fire": probability,
                "Threshold val": threshold,
                "Confidence": max(probability, 1 - probability),
                "Latensi (ms)": latency_ms,
                "CLAHE": "Ya" if use_clahe else "Tidak",
            }
        )
        if processed_preview is None:
            processed_preview = processed
            image_stats = _image_stats(original, processed, cv2)

    predictions = pd.DataFrame(rows).sort_values("Probabilitas fire", ascending=False)
    n_total = len(predictions)
    n_fire = int((predictions["Prediksi"] == "FIRE").sum())
    verdict = "FIRE" if n_fire > 0 and n_fire * 2 >= n_total else "NO FIRE"
    vote_ratio = f"{n_fire} / {n_total} Model"
    avg_prob = f"{predictions['Probabilitas fire'].mean():.2%}"

    label_probs = {
        html.escape(str(r["Model"])): float(r["Probabilitas fire"])
        for _, r in predictions.iterrows()
    }

    table = predictions.copy()
    table["Probabilitas fire"] = table["Probabilitas fire"].map("{:.2%}".format)
    table["Threshold val"] = table["Threshold val"].map("{:.3f}".format)
    table["Confidence"] = table["Confidence"].map("{:.2%}".format)
    table["Latensi (ms)"] = table["Latensi (ms)"].map("{:.1f}".format)

    return (
        verdict,
        vote_ratio,
        avg_prob,
        label_probs,
        processed_preview,
        image_stats,
        table,
    )


def build_demo() -> Any:
    import gradio as gr

    default_runs = [
        run
        for run in RUNS
        if _arch_from_run(run) in {"mnv3l_mlp_clahe", "mnv3l_fmlp_clahe", "mnv3l_clahe"}
    ][:3] or RUNS[:1]
    choices = [(_format_run(run), run) for run in RUNS]

    # Hilangkan border putus-putus pada upload image & beri margin horizontal lebih tebal
    custom_css = """
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 8rem;
    }
    @media (max-width: 768px) {
        .main-container {
            padding: 0 1rem;
        }
    }
    /* Hapus border dashed pada komponen Image input/upload */
    .image-container, .upload-container, div[data-testid="image"] {
        border-style: solid !important;
    }
    """

    with gr.Blocks(title="Wildfire FireDet", css=custom_css) as demo:
        with gr.Column(elem_classes="main-container"):
            gr.Markdown(
                "# Wildfire FireDet\n"
                "Sistem klasifikasi **fire / no fire** berbasis citra digital dengan pembandingan multi-checkpoint Exp2."
            )

            with gr.Row(equal_height=False):
                # Panel Kiri: Input & Kontrol
                with gr.Column(scale=4):
                    with gr.Group():
                        image = gr.Image(
                            type="numpy",
                            label="Upload Gambar",
                            sources=["upload", "clipboard"],
                        )

                    with gr.Accordion("Pengaturan Model Comparison", open=True):
                        models = gr.CheckboxGroup(
                            choices=choices,
                            value=default_runs,
                            label="Pilih Checkpoint Model",
                        )
                        with gr.Row():
                            select_all = gr.Button("Pilih Semua", size="sm")
                            reset_sel = gr.Button("Default", size="sm")

                    with gr.Row():
                        reset = gr.Button("Reset Input", variant="secondary")
                        button = gr.Button("Klasifikasikan", variant="primary")

                # Panel Kanan: Hasil & Analisis
                with gr.Column(scale=6):
                    # 3 Card Statistik Terpisah
                    with gr.Row():
                        card_verdict = gr.Textbox(
                            label="Hasil Utama",
                            value="-",
                            interactive=False,
                        )
                        card_votes = gr.Textbox(
                            label="Voting Mayoritas",
                            value="-",
                            interactive=False,
                        )
                        card_avg = gr.Textbox(
                            label="Rata-rata Probabilitas",
                            value="-",
                            interactive=False,
                        )

                    with gr.Tabs():
                        with gr.Tab("Ringkasan & Detail Prediksi"):
                            probs = gr.Label(
                                label="Probabilitas Fire per Model",
                                num_top_classes=len(RUNS),
                            )
                            predictions = gr.Dataframe(
                                label="Tabel Hasil Deteksi",
                                interactive=False,
                            )

                        with gr.Tab("Preprocessing Citra"):
                            with gr.Row():
                                processed = gr.Image(
                                    label="Hasil Preprocessing (Model Pertama)",
                                    interactive=False,
                                )
                                image_stats = gr.Dataframe(
                                    label="Statistik Gambar Asli",
                                    interactive=False,
                                )

                        with gr.Tab("Benchmark Exp2"):
                            benchmark = gr.Dataframe(
                                label="Ringkasan Kinerja (Mean ± SD antar Seed)",
                                interactive=False,
                            )

        # Event Handlers
        button.click(
            classify,
            inputs=[image, models],
            outputs=[
                card_verdict,
                card_votes,
                card_avg,
                probs,
                processed,
                image_stats,
                predictions,
            ],
        )
        select_all.click(lambda: RUNS, outputs=models)
        reset_sel.click(lambda: default_runs, outputs=models)
        reset.click(
            lambda: (None, "-", "-", "-", None, None, None, None),
            outputs=[
                image,
                card_verdict,
                card_votes,
                card_avg,
                probs,
                processed,
                image_stats,
                predictions,
            ],
        )
        demo.load(_benchmark, outputs=benchmark)

    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
