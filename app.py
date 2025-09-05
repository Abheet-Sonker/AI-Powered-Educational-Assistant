import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# 🔐 Load environment variables from .env
load_dotenv()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="CampusX ML & DL Chatbot", layout="wide")
st.title("🤖 CampusX Machine Learning & Deep Learning Chatbot created by (Deepti & Abheet)")
st.write("Ask me anything from the CampusX ML and DL playlist!")

# --- Advanced Teaching Prompt ---
custom_prompt = PromptTemplate(
    template="""
You are a highly knowledgeable assistant helping beginners learn from the CampusX 100 Days of ML playlist.

When answering the question, always follow this structure:
1. *Definition*: Start by clearly defining the concept in simple words.
2. *Mathematical Explanation*: Include formulas or equations (if available) and explain them step by step.
3. *Intuition*: Describe the algorithm’s working in layman's terms, like how a teacher would explain using analogies or real-life comparisons.
4. *Example*: Give a practical example or case study to demonstrate the concept.
5. *Summary*: End with a quick summary of the key takeaway.

Also if any section is not applicable just skip it do not show it to user.

Only use the following context to answer:
{context}

If the answer is not present in the context, respond: "I don't know based on the provided videos."

Question:
{question}
""",
    input_variables=["context", "question"]
)



# --- Load Vector DB ---
@st.cache_resource
def load_vector_db():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# --- Load LLM (Groq) ---
@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),  # ✅ Use .env key like GROQ_API_KEY=sk-xxx
        temperature=0.1,
    )

# --- Setup QA Chain ---
db = load_vector_db()
llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

query = st.text_input("💬 Ask your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({"query": query})  # response is a dict with 'result'

            # Extract and clean up output
            result = response["result"]
            query_text = response["query"]

            # Format nicely: each heading in bold, split by \n\n
            formatted_result = ""
            for section in result.split("\n\n"):
                if ":" in section:
                    heading, content = section.split(":", 1)
                    formatted_result += f"{heading.strip()}:\n{content.strip()}\n\n"
                else:
                    formatted_result += section + "\n\n"

            # Show in UI
            st.markdown(f"### 🤖 Answer:")
            st.markdown(f"*Query:* {query_text}\n\n" + formatted_result, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
# --- Optional: Visualizer toggle ---
# Button click tracking using session_state
if "show_visualizer" not in st.session_state:
    st.session_state.show_visualizer = False

if st.button("🎨 Launch Algorithm Visualizer") or st.session_state.show_visualizer:
    st.session_state.show_visualizer = True
    st.markdown("---")
    st.header("🔍 Machine Learning Algorithm Visualizer")

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score ,precision_score, recall_score
    from sklearn.decomposition import PCA

    # Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.inspection import DecisionBoundaryDisplay

    def create_dataset(dataset_type, n_samples, n_features, n_classes, noise, random_state, separable):
        if dataset_type == "Binary Classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=2 if n_features > 1 else 1,
                n_redundant=0,
                n_classes=2,
                n_clusters_per_class=1,
                flip_y=noise,
                class_sep=2.0 if separable else 0.5,
                random_state=random_state
            )
        else:
            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_classes,
                cluster_std=noise*5,
                random_state=random_state
            )
            if not separable:
                X += np.random.normal(0, noise*2, X.shape)
        return X, y

    def get_model(model_name, params):
        if model_name == "Logistic Regression":
            return LogisticRegression(C=params['C'], max_iter=params['max_iter'])
        elif model_name == "SVM":
            return SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(max_depth=params['max_depth'])
        elif model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
        elif model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        return None

    with st.sidebar:
        st.header("⚙️ Visualizer Configuration")
        dataset_type = st.selectbox("Dataset type", ["Binary Classification", "Multiclass Classification"])
        model_name = st.selectbox("Classifier", [
            "Logistic Regression", "SVM", "Decision Tree", "Random Forest", "K-Nearest Neighbors"
        ])
        st.subheader("Dataset Parameters")
        n_samples = st.slider("Samples", 100, 2000, 500, step=50)
        n_features = st.slider("Features", 2, 5, 2)
        n_classes = st.slider("Classes", 2, 5, 2) if dataset_type == "Multiclass Classification" else 2
        noise = st.slider("Noise", 0.0, 1.0, 0.1, step=0.01)
        separable = st.checkbox("Linearly Separable", value=True)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", 0, 100, 42)

        st.subheader("Model Parameters")
        model_params = {}
        if model_name == "Logistic Regression":
            model_params['C'] = st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0)
            model_params['max_iter'] = st.slider("Max Iterations", 100, 1000, 100)
        elif model_name == "SVM":
            model_params['C'] = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
            model_params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            gamma_type = st.radio("Gamma Type", ["Predefined", "Custom"], index=0)
            if gamma_type == "Predefined":
                model_params['gamma'] = st.selectbox("Gamma", ["scale", "auto"])
            else:
                model_params['gamma'] = st.slider("Custom Gamma", 0.01, 10.0, 1.0)
        elif model_name == "Decision Tree":
            model_params['max_depth'] = st.slider("Max Depth", 1, 20, 3)
        elif model_name == "Random Forest":
            model_params['n_estimators'] = st.slider("Number of Trees", 10, 200, 100)
            model_params['max_depth'] = st.slider("Max Depth", 1, 20, 5)
        elif model_name == "K-Nearest Neighbors":
            model_params['n_neighbors'] = st.slider("Number of Neighbors", 1, 50, 5)

    # Data
    X, y = create_dataset(dataset_type, n_samples, n_features, n_classes, noise, random_state, separable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reduce to 2D
    if n_features > 2:
        pca = PCA(n_components=2)
        X_train_vis = pca.fit_transform(X_train)
        X_test_vis = pca.transform(X_test)
    else:
        X_train_vis = X_train
        X_test_vis = X_test

    model = get_model(model_name, model_params)
    model.fit(X_train_vis, y_train)

    y_pred = model.predict(X_test_vis)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)


    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    try:
        DecisionBoundaryDisplay.from_estimator(
            model, X_train_vis, response_method="predict",
            ax=ax[0], alpha=0.5, cmap='coolwarm'
        )
    except Exception as e:
        st.warning(f"Could not plot decision boundary: {e}")

    ax[0].scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, edgecolors='k')
    ax[0].set_title("Training Data + Decision Boundary")

    ax[1].scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_pred, marker='x', s=100, label='Predicted')
    ax[1].scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test, edgecolors='k', alpha=0.5, label='True')
    ax[1].set_title("Test Data Predictions")
    ax[1].legend()

    st.subheader("📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("F1 Score", f"{f1:.2%}")
    col3.metric("Precision", f"{precision:.2%}")
    col4.metric("Recall", f"{recall:.2%}")


    st.pyplot(fig)

    with st.expander("📁 Dataset Info"):
        st.write(f"**Dataset Type:** {dataset_type}")
        st.write(f"**Samples:** {n_samples} | **Features:** {n_features} | **Classes:** {n_classes}")
        st.write(f"**Noise Level:** {noise} | **Test Size:** {test_size} | **Separable:** {separable}")
        st.write("**First 5 samples (standardized):**")
        st.write(X_train[:5])
