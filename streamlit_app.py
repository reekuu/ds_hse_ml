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
X['type_of_travel'] = st.selectbox('Тип поездки', ['Business travel', 'Personal Travel'])
X['class'] = st.selectbox('Класс обслуживания', ['Business', 'Eco', 'Eco Plus'])
X['flight_distance'] = st.number_input('Дальность перелета (мили)', min_value=0, max_value=5100, value=500)
X['departure_delay_in_minutes'] = st.number_input('Задержка отправления (минуты)', min_value=0, value=0)
X['arrival_delay_in_minutes'] = st.number_input('Задержка прибытия (минуты)', min_value=0, value=0)

# Добавление полей ввода для оценок сервиса
X['inflight_wifi_service'] = str(st.number_input('Оценка интернета на борту', min_value=0, max_value=5, value=5))
X['departure/arrival_time_convenient'] = str(st.number_input('Оценка удобства времени отправления/прибытия', min_value=0, max_value=5, value=5))
X['ease_of_online_booking'] = str(st.number_input('Оценка удобства онлайн-бронирования', min_value=0, max_value=5, value=5))
X['gate_location'] = str(st.number_input('Оценка расположения выхода на посадку', min_value=0, max_value=5, value=5))
X['food_and_drink'] = str(st.number_input('Оценка еды и напитков на борту', min_value=0, max_value=5, value=5))
X['online_boarding'] = str(st.number_input('Оценка выбора места в самолете', min_value=0, max_value=5, value=5))
X['seat_comfort'] = str(st.number_input('Оценка удобства сиденья', min_value=0, max_value=5, value=5))
X['inflight_entertainment'] = str(st.number_input('Оценка развлечений на борту', min_value=0, max_value=5, value=5))
X['on-board_service'] = str(st.number_input('Оценка обслуживания на борту', min_value=0, max_value=5, value=5))
X['leg_room_service'] = str(st.number_input('Оценка места в ногах на борту', min_value=0, max_value=5, value=5))
X['baggage_handling'] = str(st.number_input('Оценка обращения с багажом', min_value=0, max_value=5, value=5))
X['checkin_service'] = str(st.number_input('Оценка регистрации на рейс', min_value=0, max_value=5, value=5))
X['inflight_service'] = str(st.number_input('Оценка обслуживания во время полета', min_value=0, max_value=5, value=5))
X['cleanliness'] = str(st.number_input('Оценка чистоты на борту', min_value=0, max_value=5, value=5))

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
