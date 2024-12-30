import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Загрузка и подготовка модели
@st.cache_resource
def load_model():
    # Инициализация и обучение модели
    data = pd.read_excel("Home_dush.xlsx")

    # Кодирование категорий
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

# Преобразование пользовательского ввода
def prepare_input(data, label_encoder):
    for column, mapping in label_encoder.items():
        data[column] = mapping.get(data[column], -1)
    return data
try:
# Заголовок приложения
st.title("Калькулятор стоимости квартиры")

# Ввод данных пользователем
st.sidebar.header("Параметры квартиры")
input_data = {
    "Комнаты": st.sidebar.number_input("Количество комнат", value=1, step=1),
    "Этаж": st.sidebar.number_input("Этаж", value=1, step=1),
    "Площадь": st.sidebar.number_input("Площадь (кв.м)", value=50.0, step=1.0),
    "Тип": st.sidebar.selectbox("Тип", label_encoder["Тип"].values()),
    "Состояние": st.sidebar.selectbox("Состояние", label_encoder["Состояние"].values()),
    "Ремонт": st.sidebar.selectbox("Ремонт", label_encoder["Ремонт"].values()),
}

# Подготовка данных
input_df = pd.DataFrame([input_data])
input_df = prepare_input(input_df, label_encoder)

# Предсказание цены
if st.button("Рассчитать стоимость"):
    prediction = model.predict(input_df)
    st.subheader(f"Предполагаемая стоимость квартиры: {prediction[0]:,.2f} руб.")
except Exception as e:
    st.error(f"Произошла ошибка: {str(e)}")
