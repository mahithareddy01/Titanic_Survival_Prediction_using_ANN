import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor 🚢",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------
# Styling
# ------------------------------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1f77b4;
}
.sub{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.result-box{
padding:20px;
border-radius:15px;
text-align:center;
font-size:25px;
font-weight:bold;
}
</style>
""",unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>🚢 Titanic Survival Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub'>Artificial Neural Network Simulation (No TensorFlow)</div>",
    unsafe_allow_html=True
)

# ------------------------------------
# Input Section
# ------------------------------------

col1,col2=st.columns(2)

with col1:

    pclass=st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

    age=st.slider(
        "Age",
        1,
        80,
        30
    )

    fare=st.number_input(
        "Fare",
        0.0,
        500.0,
        50.0
    )

# ------------------------------------
# ANN Simulation
# ------------------------------------

def sigmoid(x):
    return 1/(1+np.exp(-x))

def predict_survival(pclass,age,fare):

    # normalize

    pclass=(pclass-1)/2
    age=age/80
    fare=fare/500

    x=np.array([pclass,age,fare])

    # Hidden layer weights
    W1=np.array([
        [0.2,0.4,0.6],
        [0.3,0.5,0.7]
    ])

    b1=np.array([0.1,0.1])

    hidden=sigmoid(np.dot(W1,x)+b1)

    # Output layer

    W2=np.array([0.5,0.8])
    b2=0.2

    output=sigmoid(np.dot(W2,hidden)+b2)

    # simple Titanic adjustments

    if pclass==0:
        output+=0.15

    if age<0.15:
        output+=0.10

    if fare>0.4:
        output+=0.10

    output=np.clip(output,0,1)

    return output


if st.button("Predict"):

    prob=predict_survival(
        pclass,
        age,
        fare
    )

    survive=prob>=0.5

    with col2:

        if survive:

            st.markdown(
            f"""
            <div class='result-box'
            style='background:#d4edda;color:green'>
            ✅ Survived
            <br>
            Probability:{prob:.2%}
            </div>
            """,
            unsafe_allow_html=True
            )

        else:

            st.markdown(
            f"""
            <div class='result-box'
            style='background:#f8d7da;color:red'>
            ❌ Not Survived
            <br>
            Probability:{prob:.2%}
            </div>
            """,
            unsafe_allow_html=True
            )

        fig=go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text':"Survival Probability"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"green"},
                'steps':[
                    {'range':[0,50],'color':"lightcoral"},
                    {'range':[50,100],'color':"lightgreen"}
                ]
            }
        ))

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


st.markdown("---")
st.write("ANN Architecture: 3 → 2 → 1")
st.write("Features: Passenger Class, Age, Fare")
