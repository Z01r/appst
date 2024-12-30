import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

st.set_page_config(layout="wide", page_title="Мобильное приложение", page_icon="📱")
@st.cache_resource
def load_model():
   
    data = pd.read_excel("Home_dush.xlsx")
    label_encoder = {"Тип": {}, "Состояние": {}, "Ремонт": {}}
    for column in label_encoder.keys():
        data[column], mapping = pd.factorize(data[column])
        label_encoder[column] = dict(enumerate(mapping))

    X = data.drop(["Цена"], axis=1)
    y = data["Цена"]

    model = CatBoostRegressor(verbose=0)
    model.fit(X, y)

    return model, label_encoder

model, label_encoder = load_model()

def prepare_input(data, label_encoder):
    for column, mapping in label_encoder.items():
        if column in data.columns:
            data[column] = data[column].map(mapping).fillna(-1).astype(int)
    return data
st.title("Калькулятор стоимости квартиры")


st.sidebar.header("Параметры квартиры")
input_data = {
    "Комнаты": st.sidebar.number_input("Количество комнат", value=1, step=1),
    "Этаж": st.sidebar.number_input("Этаж", value=1, step=1),
    "Площадь": st.sidebar.number_input("Площадь (кв.м)", value=50, step=1),
    "Тип": st.sidebar.selectbox("Тип", label_encoder["Тип"].values()),
    "Состояние": st.sidebar.selectbox("Состояние", label_encoder["Состояние"].values()),
    "Ремонт": st.sidebar.selectbox("Ремонт", label_encoder["Ремонт"].values()),
}


input_df = pd.DataFrame([input_data])
input_df = prepare_input(input_df, label_encoder)


if st.button("Рассчитать стоимость"):
    prediction = model.predict(input_df)
    st.subheader(f"Предполагаемая стоимость квартиры: {prediction[0].round(0):,.2f} сомони")
