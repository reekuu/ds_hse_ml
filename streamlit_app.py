import pandas as pd
import streamlit as st
import joblib


# Создание шаблона Датафрейма
cat_features = [
    'gender', 'customer_type', 'type_of_travel', 'class',
    'inflight_wifi_service', 'departure/arrival_time_convenient',
    'ease_of_online_booking', 'gate_location', 'food_and_drink',
    'online_boarding', 'seat_comfort', 'inflight_entertainment',
    'on-board_service', 'leg_room_service', 'baggage_handling',
    'checkin_service', 'inflight_service', 'cleanliness',
    ]
non_cat_features = ['age', 'flight_distance', 'departure_delay_in_minutes', 'arrival_delay_in_minutes']
X = pd.Series(index=cat_features+non_cat_features, dtype='object')

# Загрузка обученной модели, и кодировщиков
model = joblib.load("./model/random_forest_model.joblib")
label_encoder =  joblib.load("./model/label_encoder.joblib")
hot_encoder = joblib.load("./model/hot_encoder.joblib")

# Создание интерфейса пользователя с помощью Streamlit
st.title('Предсказание удовлетворенности клиентов авиакомпании')

# Добавление полей ввода для характеристик клиента
X['gender'] = st.selectbox('Пол', ['Male', 'Female'])
X['age'] = st.number_input('Возраст', min_value=0, max_value=100, value=30)
X['customer_type'] = st.selectbox('Тип клиента', ['Loyal Customer', 'disloyal Customer'])
X['type_of_travel'] = st.selectbox('Цель поездки', ['Business travel', 'Personal Travel'])
X['class'] = st.selectbox('Класс обслуживания', ['Business', 'Eco', 'Eco Plus'])
X['flight_distance'] = st.number_input('Дальность перелета (мили)', min_value=0, max_value=5100, value=500)
X['departure_delay_in_minutes'] = st.number_input('Задержка вылета (минуты)', min_value=0, value=0)
X['arrival_delay_in_minutes'] = st.number_input('Задержка прилета (минуты)', min_value=0, value=0)

# Добавление полей ввода для оценок сервиса
X['inflight_wifi_service'] = st.selectbox('Wi-Fi на борту', ['5', '4', '3', '2', '1', '0'])
X['departure/arrival_time_convenient'] = st.selectbox('Удобство времени вылета/прилета', ['5', '4', '3', '2', '1', '0'])
X['ease_of_online_booking'] = st.selectbox('Простота онлайн-бронирования', ['5', '4', '3', '2', '1', '0'])
X['gate_location'] = st.selectbox('Расположение выхода на посадку', ['5', '4', '3', '2', '1', '0'])
X['food_and_drink'] = st.selectbox('Качество еды и напитков', ['5', '4', '3', '2', '1', '0'])
X['online_boarding'] = st.selectbox('Удобство выбора места', ['5', '4', '3', '2', '1', '0'])
X['seat_comfort'] = st.selectbox('Комфортность кресла', ['5', '4', '3', '2', '1', '0'])
X['inflight_entertainment'] = st.selectbox('Оценка развлекательной системы', ['5', '4', '3', '2', '1', '0'])
X['on-board_service'] = st.selectbox('Бортовое обслуживание', ['5', '4', '3', '2', '1', '0'])
X['leg_room_service'] = st.selectbox('Оценка пространства для ног', ['5', '4', '3', '2', '1', '0'])
X['baggage_handling'] = st.selectbox('Обработка багажа', ['5', '4', '3', '2', '1', '0'])
X['checkin_service'] = st.selectbox('Регистрации на рейс', ['5', '4', '3', '2', '1', '0'])
X['inflight_service'] = st.selectbox('Обслуживания на земле', ['5', '4', '3', '2', '1', '0'])
X['cleanliness'] = st.selectbox('Чистота', ['5', '4', '3', '2', '1', '0'])

# Кнопка для выполнения предсказания
if st.button('Предсказать'):
    # Выполнение предсказания на основе введенных данных
    X = pd.DataFrame(X).T
    X_encoded = pd.DataFrame(hot_encoder.transform(X[cat_features]).toarray(),
                             columns=hot_encoder.get_feature_names_out(cat_features),
                             index=X.index)
    X_encoded = pd.concat([X.drop(columns=cat_features), X_encoded], axis=1)
    prediction = label_encoder.inverse_transform(model.predict(X_encoded))[0]
    st.write(f'Уровень удовлетворенности: {prediction}')
