import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import numpy as np

from streamlit_option_menu import option_menu
    # st.set_page_config(layout="wide")

selected = option_menu(
        menu_title=None, 
        options = ['Messy Dane' ,'Czyste Dane', 'Wizualizacja', 'Regresja'], 
        icons= ['alphabet', 'book', 'balloon-heart', 'briefcase'],
        menu_icon = 'cast',
        default_index=0, 
        orientation='horizontal'
    )

if selected == 'Czyste Dane':
    pd.set_option('display.max_columns', None)
    data = pd.read_csv("czyste_dane.csv", sep=",")
    st.title("Dane po czyszczeniu")

    available_columns = [col for col in data.columns if col != 'table']
    option = st.selectbox(
        'Którą zmienną chcesz zobaczyć?',
            available_columns)

    st.write('Wybrana zmienna:', option)
    st.write('Próbka danych wybranej kolumny')
    st.dataframe(data[[option]].head(20), use_container_width=True)

    if data[option].dtype == 'object':
        fig, ax = plt.subplots()
        sns.countplot(x=option, data=data, ax=ax)
        st.pyplot(fig)
    else:
        st.dataframe(data[option].describe(), use_container_width=True)
        # fig, ax = plt.subplots()
        # sns.countplot(x=option, data=data, ax=ax)
        # plt.xticks(rotation=45, ha='right')
        # st.pyplot(fig)

if selected == 'Messy Dane':
    pd.set_option('display.max_columns', None)
    data = pd.read_csv("messy_data.csv", sep=",")
    st.title("Dane przed czyszczeniem")
    status = st.radio("Co chcesz zobaczyć? ", ("Tabela", "Kolumny"))

    if status == "Tabela":
        st.dataframe(data.head(20), use_container_width=True)
    if status == "Kolumny":    
        available_columns = [col for col in data.columns if col != 'table']
        option = st.selectbox(
            'Którą zmienną chcesz zobaczyć?',
                available_columns)

        st.write('Wybrana zmienna:', option)
        st.write('Próbka danych wybranej kolumny')
        st.dataframe(data[[option]].head(20), use_container_width=True)
if selected == 'Wizualizacja':
    data = pd.read_csv("czyste_dane.csv", sep=",")
    st.title("Wizualizacja zaleności ceny od: ")
    status = st.radio("Co chcesz zobaczyć? ", ("Ilości karatów", "Koloru", "Przejrzystości", "Wymiaru x", "Wymiaru y", "Wymiaru z"))
    if status == "Koloru":
        fig_color = px.box(data, x='color', y='price', title='Zależność ceny od koloru')
        fig_color.show()
    if status == "Ilości karatów":
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='carat', y='price', data=data, ax=ax3)
        ax3.set_title('Zależność ceny od carat')
        ax3.set_xlabel('Carat')
        ax3.set_ylabel('Cena')
        st.pyplot(fig3)
    if status == "Przejrzystości":
        fig_clarity = px.box(data, x='clarity', y='price', title='Zależność ceny od przejrzystości')
        fig_clarity.show()
    if status == "Wymiaru x":
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='xdimension', y='price', data=data, ax=ax3)
        ax3.set_title('Zależność ceny od wymiaru x')
        ax3.set_xlabel('wymiar x')
        ax3.set_ylabel('Cena')
        st.pyplot(fig3)
    if status == "Wymiaru y":
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='ydimension', y='price', data=data, ax=ax3)
        ax3.set_title('Zależność ceny od wymiaru y')
        ax3.set_xlabel('wymiar y')
        ax3.set_ylabel('Cena')
        st.pyplot(fig3)
    if status == "Wymiaru z":
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='zdimension', y='price', data=data, ax=ax3)
        ax3.set_title('Zależność ceny od wymiaru z')
        ax3.set_xlabel('wymiar z')
        ax3.set_ylabel('Cena')
        st.pyplot(fig3)
if selected == 'Regresja':
    data = pd.read_csv("czyste_dane.csv", sep=",")
    st.title("Model regresji, cena od: ")
    status = st.radio("Co chcesz zobaczyć? ", ( "Koloru", "Karatów", "Wymiarów"))
    if status == "Koloru":
        formula = "price ~ C(color)"

    # Budowanie modelu
        model = smf.ols(formula=formula, data=data).fit()

        # Wyświetlenie podsumowania modelu
        st.write(model.summary()) 
        checkbox = st.checkbox("Pokaz błąd R2")
        if checkbox:
            model_mean = smf.ols("price ~ 1", data=data).fit()
            resid_mean, resid_model = np.mean(model_mean.resid**2), np.mean(model.resid**2)
            resid_mean, resid_model = model_mean.mse_resid, model.mse_resid
            r2 = (resid_mean - resid_model)/resid_mean # QED
            st.write("R2 wynosi: ", r2)
    if status == "Wymiarów":
        formula = "price ~ xdimension + ydimension + zdimension"

    # Budowanie modelu
        model = smf.ols(formula=formula, data=data).fit()

        # Wyświetlenie podsumowania modelu
        st.write(model.summary())
        checkbox = st.checkbox("Pokaz błąd R2")
        if checkbox:
            model_mean = smf.ols("price ~ 1", data=data).fit()
            resid_mean, resid_model = np.mean(model_mean.resid**2), np.mean(model.resid**2)
            resid_mean, resid_model = model_mean.mse_resid, model.mse_resid
            r2 = (resid_mean - resid_model)/resid_mean # QED
            st.write("R2 wynosi: ", r2)
    if status == "Karatów":
        formula = "price ~ carat"

    # Budowanie modelu
        model = smf.ols(formula=formula, data=data).fit()

        # Wyświetlenie podsumowania modelu
        st.write(model.summary())  
        data["fitted"] = model.fittedvalues

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data["carat"], y=data["price"], name="Karaty vs Cena", mode="markers"))

        fig.add_trace(go.Scatter(
            x=data["carat"], y=data["fitted"], name="Model regresji"))

        fig.update_layout(title="Regression: Karaty vs Cena",
                        xaxis_title="Karaty",
                        yaxis_title="Cena")

        fig.show()
        checkbox = st.checkbox("Pokaz błąd R2")
        if checkbox:
            model_mean = smf.ols("price ~ 1", data=data).fit()
            resid_mean, resid_model = np.mean(model_mean.resid**2), np.mean(model.resid**2)
            resid_mean, resid_model = model_mean.mse_resid, model.mse_resid
            r2 = (resid_mean - resid_model)/resid_mean # QED
            st.write("R2 wynosi: ", r2)