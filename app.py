import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Malay Sentiment Dashboard", page_icon="📊", layout="wide")

# ---------------- RESULTS ----------------
ml = pd.DataFrame([
    ["Logistic Regression","BoW",70.75,70.70,70.75,70.58],
    ["Random Forest","BoW",67.66,67.54,67.66,67.50],
    ["Logistic Regression","TF-IDF",72.13,72.10,72.13,71.99],
    ["Random Forest","TF-IDF",67.86,67.71,67.86,67.68],
], columns=["Model","Representation","Accuracy","Precision","Recall","F1"])

dl = pd.DataFrame([
    ["CNN","Word2Vec",71.73,71.86,71.73,71.68],
    ["LSTM","Word2Vec",69.89,69.78,69.89,69.63],
    ["BiLSTM","Word2Vec",72.89,72.89,72.89,72.87],
    ["CNN","FastText",71.96,72.23,71.96,71.95],
    ["LSTM","FastText",69.57,69.46,69.57,69.45],
    ["BiLSTM","FastText",72.08,72.37,72.08,71.88],
], columns=["Model","Embedding","Accuracy","Precision","Recall","F1"])

tr = pd.DataFrame([
    ["MalayBERT",73.31,73.0,73.0,73.0,1693.07],
    ["mBERT",72.36,73.0,72.0,72.0,1211.25],
    ["XLM-R",77.11,77.0,77.0,77.12,1376.37],
], columns=["Model","Accuracy","Precision","Recall","F1","Training Time (s)"])

best = pd.DataFrame([
    ["Machine Learning","TF-IDF + Logistic Regression",72.13,71.99],
    ["Deep Learning","BiLSTM + Word2Vec",72.89,72.87],
    ["Transformer","XLM-R",77.11,77.12],
], columns=["Family","Best Model","Accuracy","F1"])

pages = ["Overview","ML Results","DL Results","Transformer Results",
         "Literature Comparison","Computational Efficiency"]
page = st.sidebar.radio("Dashboard", pages)
st.sidebar.info("Dataset: Combined Malay sentiment dataset from multiple Malay sentiment sources.\n\nDeployment focuses on performance comparison and computational efficiency; confusion matrices and epoch-level training curves are not displayed.")

def pct_table(df):
    return df.style.format({c:"{:.2f}%" for c in df.columns if c in ["Accuracy","Precision","Recall","F1"]})

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("📊 Malay Sentiment Analysis Dashboard")
    st.write("Comparative evaluation of Machine Learning, Deep Learning and Transformer-based models.")

    c1,c2,c3 = st.columns(3)
    c1.metric("Best ML","TF-IDF + Logistic Regression","72.13%")
    c2.metric("Best DL","BiLSTM + Word2Vec","72.89%")
    c3.metric("Best Transformer","XLM-R","77.11%")

    st.subheader("Best Accuracy by Model Family")
    fig = px.bar(best,x="Family",y="Accuracy",text="Accuracy")
    fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside")
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Overall Results")
    st.dataframe(best.style.format({"Accuracy":"{:.2f}%","F1":"{:.2f}%"}),use_container_width=True)

# ---------------- ML ----------------
elif page == "ML Results":
    st.title("🤖 Machine Learning Results")
    st.dataframe(pct_table(ml),use_container_width=True)

    fig = px.bar(ml,x="Model",y="Accuracy",color="Representation",
                 barmode="group",text="Accuracy",title="ML Accuracy")
    fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside")
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig,use_container_width=True)

    long = ml.melt(["Model","Representation"],["Accuracy","Precision","Recall","F1"],
                   var_name="Metric",value_name="Score")
    fig2 = px.bar(long,x="Model",y="Score",color="Metric",
                  facet_col="Representation",barmode="group",title="ML Metrics")
    fig2.update_yaxes(range=[0,100])
    st.plotly_chart(fig2,use_container_width=True)
    st.success("Best ML: TF-IDF + Logistic Regression — 72.13% accuracy, 71.99% F1.")

# ---------------- DL ----------------
elif page == "DL Results":
    st.title("🧠 Deep Learning Results")
    st.dataframe(pct_table(dl),use_container_width=True)

    fig = px.bar(dl,x="Model",y="Accuracy",color="Embedding",
                 barmode="group",text="Accuracy",title="DL Accuracy")
    fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside")
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig,use_container_width=True)

    long = dl.melt(["Model","Embedding"],["Accuracy","Precision","Recall","F1"],
                   var_name="Metric",value_name="Score")
    fig2 = px.bar(long,x="Model",y="Score",color="Metric",
                  facet_col="Embedding",barmode="group",title="DL Metrics")
    fig2.update_yaxes(range=[0,100])
    st.plotly_chart(fig2,use_container_width=True)
    st.success("Best DL: BiLSTM + Word2Vec — 72.89% accuracy, 72.87% F1.")

# ---------------- TRANSFORMERS ----------------
elif page == "Transformer Results":
    st.title("🤗 Transformer Results")
    st.dataframe(tr[["Model","Accuracy","Precision","Recall","F1"]].style.format({
        "Accuracy":"{:.2f}%","Precision":"{:.2f}%","Recall":"{:.2f}%",
        "F1":"{:.2f}%"
    }),use_container_width=True)

    fig = px.bar(tr,x="Model",y="Accuracy",text="Accuracy",title="Transformer Accuracy")
    fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside")
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig,use_container_width=True)

    long = tr.melt(["Model","Training Time (s)"],
                   ["Accuracy","Precision","Recall","F1"],
                   var_name="Metric",value_name="Score")
    fig2 = px.bar(long,x="Model",y="Score",color="Metric",barmode="group",
                  title="Transformer Metrics")
    fig2.update_yaxes(range=[0,100])
    st.plotly_chart(fig2,use_container_width=True)
    st.success("Best Transformer: XLM-R — 77.11% accuracy, 77.12% F1.")

# ---------------- LITERATURE ----------------
elif page == "Literature Comparison":
    st.title("📚 Comparison with Existing Studies")
    lit = pd.DataFrame([
        ["Ho et al. (2022)","ML","MALAYA 20K","~85–88%","Current best ML: 72.13%"],
        ["Ying et al. (2020)","CNN + Word2Vec","Informal Malay tweets","77.59%","Current CNN + Word2Vec: 71.73%"],
        ["Abdul Razab & Mohamed Hanum (2026)","LSTM/GRU","Malaysia Airlines reviews","~89–91%","Current best DL: 72.89%"],
        ["Zulkalnain et al. (2025)","BERT-CNN","Bahasa Malaysia corpus","96.3%","Current best Transformer: 77.11%"],
    ],columns=["Study","Method","Dataset","Reported Accuracy","Current Study"])
    st.dataframe(lit,use_container_width=True)

    st.subheader("Direct CNN + Word2Vec Comparison")
    direct = pd.DataFrame([["Ying et al. (2020)",77.59],["Current Study",71.73]],
                          columns=["Study","Accuracy"])
    fig = px.bar(direct,x="Study",y="Accuracy",text="Accuracy")
    fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside")
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig,use_container_width=True)
    st.warning("The current CNN + Word2Vec result is 5.86 percentage points lower. Dataset and experimental differences mean this is not a direct superiority claim.")

# ---------------- EFFICIENCY ----------------
else:
    st.title("⚡ Computational Efficiency")
    tr2 = tr.copy()
    tr2["Minutes"] = tr2["Training Time (s)"]/60

    c1,c2,c3 = st.columns(3)
    c1.metric("Fastest","mBERT","20.19 min")
    c2.metric("Best Accuracy","XLM-R","77.11%")
    c3.metric("Slowest","MalayBERT","28.22 min")

    fig = px.bar(tr2,x="Model",y="Minutes",text="Minutes",title="Transformer Training Time")
    fig.update_traces(texttemplate="%{text:.2f} min",textposition="outside")
    st.plotly_chart(fig,use_container_width=True)

    fig2 = px.scatter(tr2,x="Minutes",y="Accuracy",text="Model",size="F1",
                      title="Accuracy vs Training Time")
    fig2.update_yaxes(range=[65,80])
    st.plotly_chart(fig2,use_container_width=True)
