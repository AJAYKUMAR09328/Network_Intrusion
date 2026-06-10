import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import sys
import config
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_loader import get_data_loaders
from models.transformer import IntrusionTransformer
from models.cnn import CNNClassifier
from models.gnn import TabularGNN

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="IDS - Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# MODEL CONFIG
# -------------------------------------------------
MODEL_CONFIG = {
    "GNN": {
        "class": TabularGNN,
        "file": "GNN_Model.pth"
    },
    "CNN": {
        "class": CNNClassifier,
        "file": "CNN_Model.pth"
    },
    "Transformer": {
        "class": IntrusionTransformer,
        "file": "Transformer_Model.pth"
    }
}

# -------------------------------------------------
# LABELS
# -------------------------------------------------
CLASS_LABELS = {
    0: "Normal Traffic",
    1: "Denial of Service (DoS)",
    2: "Probe / Port Scan"
}

CLASS_DEFINITIONS = {
    0: "Legitimate network communication.",
    1: "Traffic flooding attack.",
    2: "Scanning activity."
}

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_resource
def load_resources():

    train_loader, val_loader, test_loader, input_dim = get_data_loaders()

    data = torch.load(
        config.PROCESSED_FILE,
        map_location=DEVICE
    )

    feature_names = data.get("feature_names", [])
    features_original = data.get("features_original")

    return (
        test_loader.dataset,
        input_dim,
        feature_names,
        features_original
    )

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model(model_name, input_dim):

    conf = MODEL_CONFIG[model_name]

    model_class = conf["class"]
    weight_file = conf["file"]

    if model_name == "GNN":

        model = model_class(
            input_dim=input_dim,
            hidden_dim=64,
            num_classes=config.NUM_CLASSES
        )

    else:

        model = model_class(
            input_dim=input_dim,
            num_classes=config.NUM_CLASSES
        )

    if not os.path.exists(weight_file):
        return None

    model.load_state_dict(
        torch.load(weight_file, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🛡️ Intrusion Detection System")
st.markdown("---")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_CONFIG.keys())
)

# -------------------------------------------------
# LOAD RESOURCES
# -------------------------------------------------
test_dataset, input_dim, feature_names, features_original = load_resources()

model = load_model(selected_model, input_dim)

if model is None:
    st.error("Model file not found.")
    st.stop()

# -------------------------------------------------
# PREPARE TEST DATA
# -------------------------------------------------
all_features = []
all_labels = []

for X, y in test_dataset:

    all_features.append(X)
    all_labels.append(y.item())

all_features = torch.stack(all_features)
all_labels = np.array(all_labels)

traffic_map = {
    "Normal Traffic": 0,
    "Probe / Port Scan": 2,
    "Denial of Service (DoS)": 1
}

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2 = st.tabs([
    "Test Instance",
    "Manual Input"
])

# =================================================
# TAB 1 — TEST INSTANCE
# =================================================
with tab1:

    left, right = st.columns([1, 1.6])

    # ---------------------------------------------
    # LEFT PANEL
    # ---------------------------------------------
    with left:

        st.subheader("Select Traffic Type")

        selected_type = st.radio(
            "Category",
            list(traffic_map.keys())
        )

        target_label = traffic_map[selected_type]

        if st.button("Evaluate Test Instance"):

            indices = np.where(
                all_labels == target_label
            )[0]

            if len(indices) == 0:

                st.warning("No samples available.")

            else:

                idx = np.random.choice(indices)

                feature_tensor = (
                    all_features[idx]
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                with torch.no_grad():

                    logits = model(feature_tensor)

                    probs = torch.softmax(
                        logits,
                        dim=1
                    )

                    pred = torch.argmax(
                        probs,
                        dim=1
                    ).item()

                    confidence = (
                        probs[0][pred].item() * 100
                    )

                st.session_state["test_result"] = {
                    "pred": pred,
                    "confidence": confidence,
                    "probs": probs,
                    "idx": idx
                }

    # ---------------------------------------------
    # RIGHT PANEL
    # ---------------------------------------------
    with right:

        if "test_result" in st.session_state:

            res = st.session_state["test_result"]

            pred = res["pred"]
            confidence = res["confidence"]
            probs = res["probs"]
            idx = res["idx"]

            st.success(CLASS_LABELS[pred])

            col1, col2 = st.columns(2)

            col1.metric(
                "Prediction",
                CLASS_LABELS[pred]
            )

            col2.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write(CLASS_DEFINITIONS[pred])

            # -----------------------------------------
            # PROBABILITY DISTRIBUTION
            # -----------------------------------------
            st.markdown("---")
            st.subheader("Class Probability Distribution")

            prob_values = probs[0].cpu().numpy()

            classes = list(CLASS_LABELS.values())

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(12, 3)
            )

            for i, ax in enumerate(axes):

                ax.bar(
                    [classes[i]],
                    [prob_values[i]]
                )

                ax.set_title(classes[i])
                ax.set_ylim(0, 1)

            st.pyplot(fig)

            # -----------------------------------------
            # FEATURE VALUES
            # -----------------------------------------
            st.markdown("---")
            st.subheader("Feature Values")

            feature_values = features_original[idx]

            names = (
                feature_names
                if feature_names
                else [
                    f"Feature_{i}"
                    for i in range(len(feature_values))
                ]
            )

            df = pd.DataFrame({
                "Feature": names,
                "Value": np.round(feature_values, 4)
            })

            st.dataframe(
                df,
                use_container_width=True
            )

            # -----------------------------------------
            # PROBABILITY TREND
            # -----------------------------------------
            st.markdown("---")
            st.subheader("Probability Trend")

            fig2 = plt.figure()

            x = np.arange(len(classes))

            plt.plot(
                x,
                prob_values,
                marker='o'
            )

            plt.xticks(
                x,
                classes,
                rotation=30
            )

            plt.ylabel("Probability")

            plt.title(
                "Class Probability Trend"
            )

            st.pyplot(fig2)

        else:

            st.info(
                "Click Evaluate Test Instance."
            )

# =================================================
# TAB 2 — MANUAL INPUT
# =================================================
with tab2:

    st.subheader("Manual Feature Input")

    manual_values = []

    names = (
        feature_names
        if feature_names
        else [
            f"Feature_{i}"
            for i in range(input_dim)
        ]
    )

    cols = st.columns(3)

    for i, feature in enumerate(names):

        with cols[i % 3]:

            value = st.number_input(
                feature,
                value=0.0,
                format="%.4f",
                key=f"manual_{i}"
            )

            manual_values.append(value)

    st.markdown("")

    if st.button("Classify Manual Input"):

        input_array = np.array(
            manual_values,
            dtype=np.float32
        )

        input_tensor = torch.tensor(
            input_array
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            logits = model(input_tensor)

            probs = torch.softmax(
                logits,
                dim=1
            )

            pred = torch.argmax(
                probs,
                dim=1
            ).item()

            confidence = (
                probs[0][pred].item() * 100
            )

        st.success(CLASS_LABELS[pred])

        col1, col2 = st.columns(2)

        col1.metric(
            "Prediction",
            CLASS_LABELS[pred]
        )

        col2.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.write(CLASS_DEFINITIONS[pred])

        # -----------------------------------------
        # PROBABILITY DISTRIBUTION
        # -----------------------------------------
        st.markdown("---")
        st.subheader("Class Probability Distribution")

        prob_values = probs[0].cpu().numpy()

        classes = list(CLASS_LABELS.values())

        fig3, axes = plt.subplots(
            1,
            3,
            figsize=(12, 3)
        )

        for i, ax in enumerate(axes):

            ax.bar(
                [classes[i]],
                [prob_values[i]]
            )

            ax.set_title(classes[i])
            ax.set_ylim(0, 1)

        st.pyplot(fig3)

        # -----------------------------------------
        # PROBABILITY TREND
        # -----------------------------------------
        st.markdown("---")
        st.subheader("Probability Trend")

        fig4 = plt.figure()

        x = np.arange(len(classes))

        plt.plot(
            x,
            prob_values,
            marker='o'
        )

        plt.xticks(
            x,
            classes,
            rotation=30
        )

        plt.ylabel("Probability")

        plt.title(
            "Class Probability Trend"
        )

        st.pyplot(fig4)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")

st.caption(
    f"Device: {DEVICE} | "
    f"Model: {selected_model}"
)