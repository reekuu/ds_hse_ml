[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reekuu/ds_hse_ml/blob/main/notebooks/airline_customer_satisfaction_analysis.ipynb)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ds-hse-ml.streamlit.app)
# Исследование удовлетворенности клиентов авиакомпании

Проект включает в себя проведение разведочного анализа данных, построение моделей машинного обучения и разработку веб-сервиса на основе [Streamlit](https://streamlit.io), реализующего доступ к предсказательной модели.

## Описание

В данном исследовании проводится анализ данных клиентов авиакомпании с целью предсказания их удовлетворенности.

## Данные

Набор данных содержит информацию о клиентах авиакомпании, включая различные характеристики клиентов и их оценки удовлетворенности полетом.

Целевая переменная – `Satisfaction` (удовлетворенность клиента полетом) – бинарная переменная, принимающая значения satisfied, neutral or dissatisfied.

Признаки включают в себя пол клиента, возраст, лояльность авиакомпании, тип поездки, класс обслуживания, дальность перелета, задержки прибытия и отправления, оценки различных аспектов полета и обслуживания.

[Ссылка на исходный набор данных](https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv)

[Отчет о профилировании (Pandas Profiling)](https://htmlpreview.github.io/?https://github.com/reekuu/ds_hse_ml/blob/main/notebooks/profiling_report.html)

## Методика

Для анализа данных и построения модели использованы различные инструменты и методы машинного обучения, включая:

- Разведочный анализ данных
- Кодирование категориальных переменных
- Построение нескольких ML-моделей и их сравнение
- Оценка качества моделей (Confusion Matrix, Precision, Recall, ROC-AUC)
- Визуализация результатов
- Оценка важности признаков

В результате анализа были построены и оценены модели машинного обучения. Модели показали высокую точность предсказаний уровня удовлетворенности клиентов. Была проведена оценка важности признаков, которые наиболее сильно влияют на удовлетворенность клиентов.

## Зависимости

- catboost
- joblib
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn
- shap
- streamlit
- tqdm
- ydata_profiling

Необходимые версии библиотек указаны в файле requirements.txt. Для установки рекомендуется использовать pip:

```bash
$ pip install -r requirements.txt
```

## Запуск

Для запуска приложения выполните в терминале следующую команду:

```bash
$ streamlit run streamlit_app.py
```
