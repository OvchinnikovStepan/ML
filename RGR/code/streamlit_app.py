import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle
import seaborn as sns

data = pd.read_csv("..\data\kc_house_data.csv")
data['bathrooms']=data['bathrooms'].astype(int)
data['price']=data['price'].astype(int)
data.pop('date')
data.pop('id')
data.pop('lat')
data.pop('long')
data.pop('zipcode')
data['cherdak']=(data['floors']%1*2).astype(int)
data['floors']=data['floors'].astype(int)
data=data.drop_duplicates()
data.head()

neoroReg=keras.models.load_model('..\models\_neuroRegressor.keras')
with open("..\models\LinearReg.pkl","rb") as f:
    linReg=pickle.load(f)
with open("..\models\BaggingRegressor.pkl","rb") as f:
    baggingReg=pickle.load(f)
with open("..\models\GradientBoostingRegressor.pkl","rb") as f:
    gradientBoostingReg=pickle.load(f)
with open("..\models\StackingRegressor.pkl","rb") as f:
    stackingReg=pickle.load(f)

def make_prediction_single(df):
    st.header("Результаты предсказания:")
    linReg_result=linReg.predict(df)
    st.write("Результат множественной линейной регрессии:",float(linReg_result))
    baggingReg_result=baggingReg.predict(df)
    st.write("Результат BaggingRegressor:",float(baggingReg_result))
    gradientBoostingReg_result=gradientBoostingReg.predict(df)
    st.write("Результат GradientBoostingRegressor:",float(gradientBoostingReg_result))
    stackingReg_result=stackingReg.predict(df)
    st.write("Результат StackingRegressor:",float(stackingReg_result))
    neoroReg_result=neoroReg.predict(df)
    st.write("Результат нейронной сети:",float(neoroReg_result))

def main():
    select_page = st.sidebar.selectbox("Page list", ("Title","DataSet description", "Visualization","Predict"), key = "Select")
    if (select_page == "Title"):
        title_page()

    elif (select_page == "DataSet description"):
        description_page()

    elif (select_page == "Predict"):
        prediction_page()

    elif (select_page == "Visualization"):
        visualization()

def title_page():
    st.title('РГР по машинному обучению студента группы ФИТ-222 Овчинникова С. А.')

def description_page():
    st.title('Описание датасета')
    st.write("Датасет содержит информацию о стоимости домов в округе Кинг, все модели обучены на нём")
    st.write(data)
    st.header("Описание признаков")
    st.write("- price: стоимость дома ")
    st.write("- bedrooms: количество спален")
    st.write("- bathrooms: количество ванных комнат")
    st.write("- sqft_living: количество квадратных футов жилого пространства")
    st.write("- sqft_lot15: количество квадратных футов пространства")
    st.write("- floors: количество этажей")
    st.write("- waterfront: находится ли дом на побережье")
    st.write("- view: оценка вида дома, 0 если не оценивался")
    st.write("- condition: состаяние дома от 1 до 5")
    st.write("- grade: оценка дома")
    st.write("- sqft_above:общая площадь в футах")
    st.write("- sqft_basement: площадь подвала в футах")
    st.write("- yr_built: год постройки дома")
    st.write("- yr_renovated: год реновации дома, 0 если не было")
    st.write("- sqft_living15: количество квадратных футов жилого пространства в пересчёте")
    st.write("- sqft_lot15: количество квадратных футов пространства в пересчёте")
    st.write("- cherdak: наличие в доме чердака")

def prediction_page():
    st.header("Interactive Price Web Application")


    sqft_living= st.slider("Choose sqft_living", 200, 20000)
    sqft_lot= st.slider("Choose sqft_lot", 200, 2000000)
    sqft_basement= st.slider("Choose sqft_basement", 0, 8000)
    sqft_above= st.slider("Choose sqft_above", 200, 20000)
    sqft_living15= st.slider("Choose sqft_living15",200, 20000)
    sqft_lot15 = st.slider('Choose sqft_lot15',200, 2000000)


    bedrooms = st.number_input('Choose the number of bedrooms', min_value= 0,
        max_value=100, value =1, step=1)
    bathrooms = st.number_input('Choose the number of bathrooms', min_value= 0,
        max_value=100, value =1, step=1)
    floors = st.number_input('Choose the number of floors', min_value= 1,
        max_value=100, value =1, step=1)
    
    condition = st.slider('Choose condition',1, 5)
    grade = st.slider('Choose grade',1, 13)
    view = st.slider('Choose view',0,4)
    yr_built = st.slider('Choose year_built',1800, 2015)

    Question_of_renovation = st.selectbox("Renovated?", ("Yes", "No"), key = "answer")
    if (Question_of_renovation == "Yes"):
        yr_renovated = st.slider('Choose renovation year',1800, 2015)

    elif (Question_of_renovation == "No"):
        yr_renovated=0

    checkbox_one = st.checkbox("On the coast?")
    checkbox_two = st.checkbox("Has attic?")

    if (checkbox_one):
        waterfront = 1
    else:
        waterfront = 0

    if (checkbox_two):
        cherdak = 1
    else:
        cherdak = 0

    arrayToPredict=[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,
                             sqft_above,sqft_basement,yr_built,yr_renovated,sqft_living15,sqft_lot15,cherdak]
    arrayToPredictNames=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade',
                             'sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15','cherdak']
    st.header("Введённые вами данные:")
 #   for i in range(len(arrayToPredict)):
 #       st.write(arrayToPredictNames[i],"-",arrayToPredict[i])

    new_df=pd.DataFrame(data=np.array([arrayToPredict]),columns=arrayToPredictNames )
    st.write(new_df)
    getPredButton=st.button("Получить предсказание")
    if getPredButton:
        make_prediction_single(pd.DataFrame(data=np.array([arrayToPredict]),columns=arrayToPredictNames ))

    st.header("Вы можете загрузить свой датасет для обработки")
    uploaded_file = st.file_uploader("Выберите файл в формате .csv", type='csv')
    if uploaded_file:
        dataframe = pd.read_csv(uploaded_file)
        if 'price' in dataframe.columns:
            dataframe.pop('price')
        getPredButton1=st.button("Получить предсказание при помощи линейной регрессии")
        getPredButton2=st.button("Получить предсказание при помощи BaggingRegressor")
        getPredButton3=st.button("Получить предсказание при помощи GradientBoostingRegressor")
        getPredButton4=st.button("Получить предсказание при помощи StackingRegressor")
        getPredButton5=st.button("Получить предсказание при помощи нейронной сети")
        if getPredButton1:
            linReg_result=linReg.predict(dataframe)
            st.write("Результат множественной линейной регрессии:",pd.DataFrame(linReg_result,columns=['predicted_price']))
        if getPredButton2:
            baggingReg_result=baggingReg.predict(dataframe)
            st.write("Результат BaggingRegressor:",pd.DataFrame(baggingReg_result,columns=['predicted_price']))
        if getPredButton3:
            gradientBoostingReg_result=gradientBoostingReg.predict(dataframe)
            st.write("Результат GradientBoostingRegressor:",pd.DataFrame(gradientBoostingReg_result,columns=['predicted_price']))
        if getPredButton4:
            stackingReg_result=stackingReg.predict(dataframe)
            st.write("Результат StackingRegressor:",pd.DataFrame(stackingReg_result,columns=['predicted_price']))
        if getPredButton5:
            neoroReg_result=neoroReg.predict(dataframe)
            st.write("Результат нейронной сети:",pd.DataFrame(neoroReg_result,columns=['predicted_price']))

   

def visualization():
    st.header("Корреляция стоимости с количеством спален, ванн и этажей дома")
    fig=plt.figure()
    fig.add_subplot(sns.heatmap(data[["price",'bedrooms','bathrooms','floors']].corr(),annot=True))
    st.pyplot(fig)

    st.header("Соотношение домов с разным числом этажей")
    fig=plt.figure()
    size=data.groupby("floors").size()
    plt.pie(size.values,labels=size.index,autopct='%1.0f%%')
    st.pyplot(plt)

    st.header("График с усами о распределение площадей домов")
    fig=plt.figure()
    plt.boxplot(data[['sqft_living','sqft_above','sqft_basement']],labels=['sqft_living','sqft_above','sqft_basement'])
    st.pyplot(plt)

    st.header("Гисторграмма, показывающая, когда были построенны дома из датасета")
    fig=plt.figure()
    plt.hist(data['yr_built'])
    st.pyplot(plt)
main()
