import streamlit as st
import torch
import pandas as pd
import json
import pickle
import torch.nn.functional as F
import shap
import torch
import torch.nn.functional as F
from src.dataset.hf_loader import load_mutag_from_hf
from src.models.base import get_model
from src.utils.graph_viz import draw_molecule_graph
from src.utils.smiles_to_graph import smiles_to_graph
from src.utils.adversarial import evaluate_robustness
from src.utils.lrp import compute_saliency_scores, compute_grad_input_scores
import matplotlib.pyplot as plt
import streamlit as st
from src.explainability.explain import compute_node_importance, normalize_scores
from src.explainability.explain import get_minimal_subgraph
from src.utils.graph_viz import ATOM_COLOR, ATOM_MAP 

from src.utils.calibrate import get_predictions_and_labels, compute_ece, plot_calibration_curve
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

st.set_page_config(page_title="Mutagenicity Predictor", layout="wide")

st.title("Mutagenicity Predictor (Graph Neural Networks)")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():

    checkpoint = torch.load("experiments/best_model.pt", map_location="cpu")

    config = checkpoint["config"]

    data_list, in_channels, num_classes = load_mutag_from_hf()

    model = get_model(
        config,
        in_channels,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state"])

    model.to(DEVICE)
    model.eval()

    return model, config, data_list
model, config, data_list = load_model()

# --------------------------------------------------
# Load experiment results
# --------------------------------------------------
def get_all_predictions(model, data_list, device):


    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for graph in data_list:

            graph = graph.to(device)

            batch = torch.zeros(
                graph.x.shape[0],
                dtype=torch.long,
                device=device
            )

            out = model(graph.x, graph.edge_index, batch)
            prob = F.softmax(out, dim=1)

            all_probs.append(prob.cpu())
            all_labels.append(graph.y.cpu())

    return torch.cat(all_probs), torch.cat(all_labels)

#--------------------------------------------------
#SHAP EXPLAINER
#--------------------------------------------------



def model_wrapper(x_numpy, model, graph, device):
    import torch
    import torch.nn.functional as F
    import numpy as np

    x_numpy = np.array(x_numpy)

    num_nodes = graph.x.shape[0]
    num_features = graph.x.shape[1]

    outputs = []

    for i in range(x_numpy.shape[0]):

        sample = x_numpy[i]

        # 🔥 FIX: reshape back to graph format
        sample = sample.reshape(num_nodes, num_features)

        x = torch.tensor(sample, dtype=torch.float32).to(device)

        batch = torch.zeros(
            num_nodes,
            dtype=torch.long,
            device=device
        )

        out = model(x, graph.edge_index, batch)
        prob = F.softmax(out, dim=1)

        outputs.append(prob.detach().cpu().numpy()[0])

    return np.array(outputs)


def compute_shap_values(model, graph, device):
    import shap
    import numpy as np

    x = graph.x.cpu().numpy()
    num_nodes, num_features = x.shape

    def f(x_numpy):
        return model_wrapper(x_numpy, model, graph, device)

    x_flat = x.reshape(1, -1)

    explainer = shap.KernelExplainer(f, x_flat)

    shap_values = explainer.shap_values(x_flat, nsamples=20)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    shap_vals = shap_vals.flatten()

    expected_size = num_nodes * num_features
    shap_vals = shap_vals[:expected_size]

    shap_vals = shap_vals.reshape(num_nodes, num_features)

    node_importance = np.abs(shap_vals).sum(axis=1)

    return node_importance# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.markdown("## ⍟ Navigation ##")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Experiments",
        "Molecule Explorer"
    ],
    label_visibility="collapsed"

)    
st.divider()
st.caption("Dataset: MUTAG")
st.caption("Models: GCN / GIN / GAT")


# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    st.write("""
This project explores the use of ***Graph Neural Networks (GNNs)*** to predict whether a molecule is ***mutagenic or non-mutagenic*** using the MUTAG dataset, a benchmark dataset consisting of 188 chemical compounds.
             Three GNN architectures were implemented and compared:

Graph Convolutional Network (GCN)

Graph Isomorphism Network (GIN)

Graph Attention Network (GAT)

The models were trained using 10-fold cross-validation, allowing robust evaluation across multiple splits of the dataset. A Streamlit-based interactive dashboard was developed to visualize molecular graphs, run predictions, and analyze experimental results.
""")

    st.subheader("Best Model Configuration")

    config_df = pd.DataFrame(
        list(config.items()),
        columns=["Parameter", "Value"]
    )

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# Experiments
# --------------------------------------------------

elif page == "Experiments":
    st.write("""The table below summarizes the results of all experiments conducted with different GNN architectures and hyperparameter settings. Each row corresponds to a unique experiment configuration, along with its mean accuracy and standard deviation across the 10 folds of cross-validation. Use this table to compare the performance of different models and identify trends in how hyperparameters affect accuracy.""")
    st.divider()
    st.header("Experiment Results")

    df = pd.read_csv("experiments/experiment_log.csv")

    st.dataframe(df)

    st.subheader("Model Accuracy Comparison")

    st.bar_chart(df.groupby("model")["accuracy_mean"].mean())

# --------------------------------------------------
# Molecule Explorer
# --------------------------------------------------

elif page == "Molecule Explorer":
    st.write("""This interactive tool allows you to explore the MUTAG dataset and make predictions on new molecules using the trained GNN model. In the first tab, you can select any molecule from the MUTAG dataset to visualize its structure and see the model's prediction for mutagenicity. In the second tab, you can upload your own CSV file containing a 'smiles' column with SMILES strings representing molecules. The app will predict whether each molecule is mutagenic or non-mutagenic, display the results in a table, and allow you to inspect individual molecules along with their predictions.""")
    st.header("Molecule Explorer")

    tab1, tab2,tab3,tab4 = st.tabs([
        "MUTAG Dataset Explorer",
        "SMILES Predictor",
        "Calibration",
        "Adversarial Robustness"
    ])

    # ==================================================
    # MUTAG DATASET EXPLORER
    # ==================================================

    with tab1:

        st.subheader("Explore MUTAG Dataset")

        idx = st.slider(
            "Select Molecule",
            0,
            len(data_list)-1,
            0
        )

        
        graph = data_list[idx]
        original_x = graph.x.clone()

        scores = None
        important_nodes = None
        method_name = "None"

        show_importance = st.checkbox("Show Node Importance")
        show_subgraph = st.checkbox("Show Minimal Subgraph")
        show_shap = st.checkbox("Show SHAP Explanation")
        show_saliency = st.checkbox("Show Saliency Map")
        show_grad_input = st.checkbox("Show LRP-like Relevance (Grad × Input)")

        if show_shap:
            scores = compute_shap_values(model, graph, DEVICE)
            method_name = "SHAP"

        elif show_grad_input:
            scores = compute_grad_input_scores(model, graph, DEVICE)
            method_name = "Grad × Input (LRP-like)"

        elif show_saliency:
            scores = compute_saliency_scores(model, graph, DEVICE)
            method_name = "Saliency"

        elif show_importance:
            scores = compute_node_importance(model, graph, DEVICE)
            method_name = "Node Importance"

        else:
            scores = None
            method_name = "None"

        if scores is not None:
            scores = normalize_scores(scores)

        if show_subgraph and scores is not None:
            subgraph_data, important_nodes = get_minimal_subgraph(
                model, graph, scores, DEVICE
            )
        else:
            subgraph_data = graph
            important_nodes = None
            
                
        col1, col2 = st.columns([2,1])

        with col1:

            fig = draw_molecule_graph(subgraph_data, scores, important_nodes, original_x)

            st.pyplot(fig)

            st.write(f"Atoms: {graph.num_nodes}")
            st.write(f"Bonds: {graph.num_edges}")

            if scores is not None:
                st.markdown("### Most Important Atoms")

                top_nodes = sorted(
                    enumerate(scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for idx, score in top_nodes:
                    atom_idx = graph.x[idx].argmax().item()
                    atom_symbol = ATOM_MAP.get(atom_idx, "C")

                    st.write(f"Atom {idx} ({atom_symbol}) → {score:.3f}")

        with col2:

            graph = graph.to(DEVICE)

            with torch.no_grad():

                batch = torch.zeros(
                    graph.x.shape[0],
                    dtype=torch.long,
                    device=DEVICE
                )

                out = model(graph.x, graph.edge_index, batch)

                prob = torch.softmax(out, dim=1)

                pred = prob.argmax().item()

            label = "Mutagenic ⚠️" if pred == 1 else "Non-Mutagenic ✅"

            st.metric("Prediction", label)

            st.metric(
                "Confidence",
                f"{prob.max().item():.3f}"
            )

    # ==================================================
    # SMILES DATASET PREDICTOR
    # ==================================================

    with tab2:

        st.subheader("Upload SMILES Dataset")

        uploaded_file = st.file_uploader(
            "Upload CSV containing a 'smiles' column",
            type=["csv"]
        )

        if uploaded_file:

            df = pd.read_csv(uploaded_file)

            st.write("Dataset Preview")

            st.dataframe(df.head())

            results = []
            graphs = []

            for _, row in df.iterrows():

                smiles = row["smiles"]

                graph = smiles_to_graph(smiles)

                if graph is None:
                    continue

                graph = graph.to(DEVICE)

                batch = torch.zeros(
                    graph.x.shape[0],
                    dtype=torch.long,
                    device=DEVICE
                )

                with torch.no_grad():

                    out = model(graph.x, graph.edge_index, batch)

                    prob = torch.softmax(out, dim=1)

                    pred = prob.argmax().item()

                label = "Mutagenic ⚠️" if pred == 1 else "Non-Mutagenic ✅"

                results.append({
                    "smiles": smiles,
                    "prediction": label,
                    "confidence": float(prob.max())
                })

                graphs.append(graph.cpu())

            results_df = pd.DataFrame(results)

            st.subheader("Prediction Results")

            st.dataframe(results_df)

            if len(graphs) > 0:

                idx = st.slider(
                    "Inspect Molecule",
                    0,
                    len(graphs)-1,
                    0
                )

                col1, col2 = st.columns([2,1])

                with col1:

                    fig = draw_molecule_graph(graphs[idx])

                    st.pyplot(fig)

                with col2:

                    st.metric(
                        "Prediction",
                        results_df.iloc[idx]["prediction"]
                    )

                    st.metric(
                        "Confidence",
                        f"{results_df.iloc[idx]['confidence']:.3f}"
                    )

            st.download_button(
                "Download Predictions",
                results_df.to_csv(index=False),
                file_name="mutagenicity_predictions.csv"
            )


        # ==================================================
        # CALIBRATION
        # ==================================================
    
    with tab3:

        st.subheader("Model Calibration")

        if st.button("Run Calibration"):

            probs, labels = get_all_predictions(model, data_list, DEVICE)

            ece = compute_ece(probs, labels)
            st.metric("ECE", f"{ece:.4f}")

            fig = plot_calibration_curve(probs, labels)
            st.pyplot(fig)
    
        #--------------------------------------------------
        # ADVERSARIAL ROBUSTNESS
        #--------------------------------------------------
    with tab4:

        st.subheader("Adversarial Robustness Analysis")

        if st.button("Run Robustness Test"):

            with st.spinner("Running perturbations..."):

                results = evaluate_robustness(
                    model,
                    data_list,
                    DEVICE,
                    perturb_levels=[0.0, 0.05, 0.1]
                )

            st.success("Done!")

            for level, acc in results.items():
                st.write(f"Perturbation {int(level*100)}% → Accuracy: {acc:.3f}")
            levels = list(results.keys())
            accs = list(results.values())

            fig, ax = plt.subplots()

            ax.plot(levels, accs, marker='o')
            ax.set_xlabel("Perturbation Ratio")
            ax.set_ylabel("Accuracy")
            ax.set_title("Robustness Curve")

            st.pyplot(fig)
