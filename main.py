import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

# Carregando e preparando os dados
data = pd.read_csv('AEP_hourly.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})

# Streamlit interface para visualização dos dados históricos
st.title('Análise de Consumo de Energia')

# Seletor de intervalo de datas para visualização dos dados históricos
start_date, end_date = st.slider(
    "Selecione o intervalo de datas para visualização:",
    min_value=data['ds'].min().date(),
    max_value=data['ds'].max().date(),
    value=(data['ds'].min().date(), data['ds'].max().date()),
    key='slider1'
)

# Filtrando dados para visualização
filtered_data = data[(data['ds'].dt.date >= start_date) & (data['ds'].dt.date <= end_date)]

# Visualização dos dados históricos
fig = px.line(filtered_data, x='ds', y='y', title='Consumo de Energia ao Longo do Tempo')
st.plotly_chart(fig)

# Streamlit interface para previsão de séries temporais
st.title('Previsão de Consumo de Energia')

# Configuração para previsão
periods_input = st.number_input('Quantos períodos à frente gostaria de prever?', min_value=1, max_value=365)
freq = st.selectbox('Selecione a frequência de previsão:', options=['D', 'W', 'M'], format_func=lambda x: {'D': 'Diário', 'W': 'Semanal', 'M': 'Mensal'}[x])

# Modelo de previsão
model = Prophet()
model.fit(data[['ds', 'y']])  # Certifique-se de usar apenas as colunas renomeadas

future = model.make_future_dataframe(periods=periods_input, freq=freq)
forecast = model.predict(future)

# Mostrando previsões
fig2 = plot_plotly(model, forecast)
st.plotly_chart(fig2)
