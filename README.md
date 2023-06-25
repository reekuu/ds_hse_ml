# Исследование удовлетворенности клиентов авиакомпании

Проект включает в себя проведение разведочного анализа данных, построение моделей машинного обучения и разработку веб-сервиса на основе [Streamlit](https://streamlit.io), реализующего исследования и модели в виде интерактивного дашборда.

## Описание

В данном исследовании проводится анализ данных клиентов авиакомпании с целью предсказания их удовлетворенности.

## Данные

Датасет содержит информацию о клиентах авиакомпании, включая различные характеристики клиентов и их оценки удовлетворенности полетом.

Целевая переменная - `Satisfaction` (удовлетворенность клиента полетом) - бинарная переменная, принимающая значения satisfied, neutral or dissatisfied.

Признаки включают в себя пол клиента, возраст, лояльность авиакомпании, тип поездки, класс обслуживания, дальность перелета, задержки прибытия и отправления, оценки различных аспектов полета и обслуживания.

[Ссылка на исходный датасет](https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/clients.csv)

[Отчет о профилировании (Pandas Profiling)](https://htmlpreview.github.io/?https://github.com/reekuu/ds_hse_ml/blob/main/notebooks/profiling_report.html)

## Методика

Для анализа данных и построения модели использованы различные инструменты и методы машинного обучения, включая:

- Разведочный анализ данных
- Кодирование категориальных переменных
- Построение моделей машинного обучения для предсказания удовлетворенности клиентов
- Оценка качества моделей (Confusion Matrix, Precision, Recall, ROC-AUC)
- Оценка важности признаков
- Визуализация результатов
- Результаты

В результате анализа были построены и оценены модели машинного обучения. Модели показали высокую точность предсказаний уровня удовлетворенности клиентов. Была проведена оценка важности признаков, которые наиболее сильно влияют на удовлетворенность клиентов.

## Использованные библиотеки

- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- CatBoost
- shap
- ydata_profiling

## Зависимости

Необходимые библиотеки и их версии указаны в файле requirements.txt.

## Установка

Для установки всех необходимых библиотек рекомендуется использовать pip:

```python
pip install -r requirements.txt
```

## Запуск

Для запуска исследования следует выполнить код в Jupyter Notebook.