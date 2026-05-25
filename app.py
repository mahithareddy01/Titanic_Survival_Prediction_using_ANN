import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# -------------------------
# CSS
# -------------------------

st.markdown("""
<style>

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#1f4e79;
}

.sub{
text-align:center;
font-size:18px;
color:gray;
margin-bottom:30px;
}

.card{
padding:20px;
border-radius:15px;
background:#f5f7fa;
text-align:center;
box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

.result{
padding:20px;
border-radius:15px;
text-align:center;
font-size:25px;
font-weight:bold;
}

</style>
""",unsafe_allow_html=True)


st.markdown(
'<div class="title">🚢 Titanic Survival Prediction Dashboard</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="sub">Artificial Neural Network Simulation without TensorFlow</div>',
unsafe_allow_html=True
)

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("Passenger Information")

pclass=st.sidebar.selectbox(
"Passenger Class",
[1,2,3]
)

age=st.sidebar.slider(
"Age",
1,
80,
28
)

fare=st.sidebar.slider(
"Fare (£)",
0,
500,
50
)

# -------------------------
# ANN SIMULATION
# -------------------------

def sigmoid(x):
    return 1/(1+np.exp(-x))


def predict(pclass,age,fare):

    pclass=(pclass-1)/2
    age=age/80
    fare=fare/500

    x=np.array([pclass,age,fare])

    W1=np.array([
        [0.2,0.4,0.6],
        [0.3,0.5,0.7]
    ])

    b1=np.array([0.1,0.1])

    hidden=sigmoid(np.dot(W1,x)+b1)

    W2=np.array([0.5,0.8])

    output=sigmoid(np.dot(W2,hidden)+0.2)

    if pclass==0:
        output+=0.15

    if age<0.15:
        output+=0.10

    if fare>0.4:
        output+=0.10

    return np.clip(output,0,1)


prob=predict(
pclass,
age,
fare
)

survival=prob*100
death=(1-prob)*100

# -------------------------
# METRIC CARDS
# -------------------------

c1,c2,c3=st.columns(3)

with c1:
    st.metric(
        "Survival Probability",
        f"{survival:.1f}%"
    )

with c2:
    st.metric(
        "Passenger Age",
        age
    )

with c3:
    st.metric(
        "Fare",
        f"£{fare}"
    )


st.write("---")

# -------------------------
# RESULT
# -------------------------

if prob>=0.5:

    st.markdown(
    f"""
    <div class="result"
    style='background:#d4edda;color:green'>
    ✅ Passenger likely survived
    </div>
    """,
    unsafe_allow_html=True
    )

else:

    st.markdown(
    f"""
    <div class="result"
    style='background:#f8d7da;color:red'>
    ❌ Passenger likely did not survive
    </div>
    """,
    unsafe_allow_html=True
    )


# -------------------------
# CHARTS
# -------------------------

col1,col2=st.columns(2)

with col1:

    df=pd.DataFrame({
        "Outcome":["Survived","Perished"],
        "Probability":[survival,death]
    })

    fig=px.bar(
        df,
        x="Outcome",
        y="Probability",
        title="Survival Probability Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig2=go.Figure(
    data=[go.Pie(
    labels=["Survived","Perished"],
    values=[survival,death],
    hole=0.5
    )])

    fig2.update_layout(
    title="Prediction Breakdown"
    )

    st.plotly_chart(
    fig2,
    use_container_width=True
    )


# -------------------------
# GAUGE
# -------------------------

fig3=go.Figure(go.Indicator(
mode="gauge+number",
value=survival,
title={"text":"Survival Score"},
gauge={
'axis':{'range':[0,100]},
'steps':[
{'range':[0,50],'color':"lightcoral"},
{'range':[50,100],'color':"lightgreen"}
]
}
))

st.plotly_chart(
fig3,
use_container_width=True
)

# -------------------------
# INSIGHTS
# -------------------------

st.write("---")

st.subheader("📘 Passenger Insights")

if pclass==1:
    st.success(
    "First-class passengers historically had better survival chances."
    )

if age<12:
    st.info(
    "Children generally had increased survival priority."
    )

if fare>100:
    st.info(
    "Higher ticket fares often correlated with better cabin locations."
    )

st.subheader("📊 Titanic Facts")

st.write("""
• Total passengers and crew: **2,224**

• Total deaths: **1,517**

• Survival rate: **32%**

• Women and children received evacuation priority

• Passenger class strongly affected survival likelihood
""")

st.write("---")

st.caption(
"🚢 Titanic Survival Prediction System | ANN Architecture: 3 → 2 → 1"
)
