"""Single-page Streamlit UI for the botnet detection pipeline."""
import html
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_loader import inspect_dataset
from train import DEFAULT_DATASET_PATH, PIPELINE_STEPS, ensure_output_directories, run_training_pipeline

DEFAULT_UPLOAD_PATH = ROOT_DIR / "data" / "uploaded_dataset.csv"
RESULTS_CSV_PATH = ROOT_DIR / "results" / "performance_metrics.csv"
STATUS_STYLES = {
    "pending": ("Pending", "#64748b"),
    "running": ("Running", "#b45309"),
    "done": ("Done", "#166534"),
    "failed": ("Failed", "#b91c1c"),
}
STATUS_ICONS = {
    "pending": "[ ]",
    "running": "[~]",
    "done": "[x]",
    "failed": "[!]",
}


def initialize_state():
    """Initialize session state used by the app."""
    step_map = {step_key: {"label": label, "status": "pending", "message": ""} for step_key, label in PIPELINE_STEPS}
    st.session_state.setdefault("step_status", step_map)
    st.session_state.setdefault("log_entries", [])
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("run_error", None)
    st.session_state.setdefault("active_dataset_path", None)


def reset_run_state():
    """Reset progress and result state before a new run."""
    st.session_state.step_status = {
        step_key: {"label": label, "status": "pending", "message": ""}
        for step_key, label in PIPELINE_STEPS
    }
    st.session_state.log_entries = []
    st.session_state.pipeline_result = None
    st.session_state.run_error = None


def append_log_entry(kind, message):
    """Append a structured entry to the run log."""
    st.session_state.log_entries.append({"kind": kind, "message": message})


def save_uploaded_dataset(uploaded_file):
    """Persist the uploaded dataset under a predictable local path."""
    ensure_output_directories()
    DEFAULT_UPLOAD_PATH.write_bytes(uploaded_file.getbuffer())
    return str(DEFAULT_UPLOAD_PATH)


def validate_dataset_path(dataset_path):
    """Validate dataset availability and required columns."""
    try:
        inspection = inspect_dataset(dataset_path)
    except Exception as exc:
        return {
            "is_valid": False,
            "message": str(exc),
            "path": dataset_path,
            "columns": [],
            "missing_columns": [],
        }

    if inspection["missing_columns"]:
        return {
            "is_valid": False,
            "message": "Missing required columns: " + ", ".join(inspection["missing_columns"]),
            "path": dataset_path,
            "columns": inspection["columns"],
            "missing_columns": inspection["missing_columns"],
        }

    return {
        "is_valid": True,
        "message": f"Dataset is ready with {inspection['column_count']} detected columns.",
        "path": dataset_path,
        "columns": inspection["columns"],
        "missing_columns": [],
    }


def render_log_panel(container, log_entries):
    """Render a single scrollable live log panel."""
    rendered_lines = []
    for entry in log_entries:
        escaped_message = html.escape(entry["message"])
        if entry["kind"] == "step":
            rendered_lines.append(
                f'<div style="font-weight:700;color:#111827;margin:0 0 6px 0;">{escaped_message}</div>'
            )
        elif entry["kind"] == "error":
            rendered_lines.append(
                f'<div style="color:#b91c1c;margin:0 0 6px 22px;">{escaped_message}</div>'
            )
        else:
            rendered_lines.append(
                f'<div style="color:#4b5563;margin:0 0 6px 22px;">{escaped_message}</div>'
            )

    content = "".join(rendered_lines)
    container.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:14px;background:#ffffff;height:320px;overflow-y:auto;padding:14px 16px;font-family:monospace;font-size:0.93rem;white-space:pre-wrap;color:#111827;">
{content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_bar(container, steps):
    """Render overall progress based on completed steps."""
    total = len(PIPELINE_STEPS)
    done_count = sum(1 for step in steps.values() if step["status"] == "done")
    failed = any(step["status"] == "failed" for step in steps.values())
    progress = done_count / total if total else 0
    container.progress(progress, text="Run failed." if failed else f"{done_count}/{total} steps completed")


def handle_progress_update(log_placeholder, progress_placeholder, payload):
    """Apply a progress event to session state and repaint the live UI."""
    step_key = payload["step_key"]
    st.session_state.step_status[step_key]["status"] = payload["status"]
    st.session_state.step_status[step_key]["message"] = payload["message"]
    if payload["status"] == "running":
        append_log_entry("step", payload["step_label"])
    elif payload["status"] == "failed":
        append_log_entry("error", payload["message"])
    render_progress_bar(progress_placeholder, st.session_state.step_status)
    render_log_panel(log_placeholder, st.session_state.log_entries)


def handle_log_update(log_placeholder, message):
    """Append a log line and repaint the live UI."""
    append_log_entry("detail", message)
    render_log_panel(log_placeholder, st.session_state.log_entries)


def display_results():
    """Display the final outputs after a completed run."""
    result = st.session_state.pipeline_result
    if not result:
        return

    st.subheader("Run Summary")
    dataset_info = result["dataset_info"]
    left, middle, right = st.columns(3)
    left.metric("Rows", dataset_info["rows"])
    middle.metric("Columns", dataset_info["columns"])
    right.metric("Missing Values", dataset_info["missing_values"])

    best_model = result["best_model"]
    st.markdown(
        f"""
        <div style="border:1px solid #dbeafe;background:#eff6ff;border-radius:14px;padding:1rem 1.2rem;margin:1rem 0;">
            <div style="font-size:0.9rem;color:#1d4ed8;font-weight:700;">BEST MODEL</div>
            <div style="font-size:1.3rem;font-weight:700;color:#0f172a;margin-top:0.2rem;">{best_model['Algorithm']}</div>
            <div style="color:#334155;margin-top:0.2rem;">Accuracy: {best_model['Accuracy (%)']}% | Precision: {best_model['Precision (%)']}% | Recall: {best_model['Recall (%)']}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model Comparison")
    st.dataframe(result["results_df"], use_container_width=True)

    if RESULTS_CSV_PATH.exists():
        st.caption(f"Saved metrics: `{RESULTS_CSV_PATH}`")

    image_paths = [Path(path) for path in result["artifact_paths"] if Path(path).exists()]
    if image_paths:
        st.subheader("Generated Visuals")
        for image_path in image_paths:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def main():
    """Run the Streamlit app."""
    st.set_page_config(page_title="Botnet Detection", layout="wide")
    ensure_output_directories()
    initialize_state()

    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
            background: #ff3b46;
            color: white;
            border: 0;
            border-radius: 8px;
            min-height: 42px;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button:hover {
            background: #e0313b;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    validation = {"is_valid": False, "message": "Select a dataset to begin.", "columns": []}
    dataset_path = None
    header_left, header_right = st.columns([4, 1.2], vertical_alignment="bottom")
    with header_left:
        st.write(
            "Choose a dataset source, run the full detection pipeline, and watch each training step update live on this page."
        )
        st.markdown("## Dataset Source")
        dataset_source = st.radio(
            "Select one option",
            ("Upload your dataset", "Use a predefined dataset"),
            horizontal=True,
            index=1,
        )
    run_button_placeholder = header_right.empty()

    if dataset_source == "Upload your dataset":
        uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
        if uploaded_file is not None:
            dataset_path = save_uploaded_dataset(uploaded_file)
            validation = validate_dataset_path(dataset_path)
            st.session_state.active_dataset_path = dataset_path
            if validation["is_valid"]:
                st.success(f"{uploaded_file.name} uploaded successfully.")
            else:
                st.error(validation["message"])
        else:
            st.info("Upload a CSV file to enable the run button.")
    else:
        dataset_path = str(ROOT_DIR / DEFAULT_DATASET_PATH)
        validation = validate_dataset_path(dataset_path)
        st.session_state.active_dataset_path = dataset_path
        if validation["is_valid"]:
            st.success(f"Using predefined dataset: `{dataset_path}`")
        else:
            st.warning(
                "Predefined dataset not found or invalid. Place your file at "
                f"`{dataset_path}` to use this option."
            )
            st.caption(validation["message"])

    if validation["columns"]:
        preview_columns = ", ".join(validation["columns"][:12])
        st.caption(f"Detected columns: {preview_columns}")

    with header_right:
        st.write("")
        st.write("")
        run_clicked = run_button_placeholder.button(
            "Run",
            type="primary",
            disabled=not validation["is_valid"],
            use_container_width=True,
        )

    if not validation["is_valid"]:
        st.caption("Run becomes active once a valid dataset source is selected.")

    progress_placeholder = st.empty()
    log_title_placeholder = st.empty()
    log_placeholder = st.empty()

    render_progress_bar(progress_placeholder, st.session_state.step_status)
    log_title_placeholder.markdown("### Run Log")
    render_log_panel(log_placeholder, st.session_state.log_entries)

    if run_clicked and validation["is_valid"] and dataset_path:
        reset_run_state()
        render_progress_bar(progress_placeholder, st.session_state.step_status)
        render_log_panel(log_placeholder, st.session_state.log_entries)

        try:
            result = run_training_pipeline(
                dataset_path=dataset_path,
                show_visualizations=True,
                save_results=True,
                sample_size=50000,
                progress_callback=lambda payload: handle_progress_update(
                    log_placeholder, progress_placeholder, payload
                ),
                log_callback=lambda message: handle_log_update(
                    log_placeholder, message
                ),
            )
            st.session_state.pipeline_result = result
        except Exception as exc:
            st.session_state.run_error = str(exc)
            st.error(f"Run failed: {exc}")

    if st.session_state.run_error:
        st.error(f"Last run error: {st.session_state.run_error}")

    display_results()


if __name__ == "__main__":
    main()
