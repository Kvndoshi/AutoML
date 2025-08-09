import pandas as pd
import streamlit as st
import ydata_profiling
from pycaret.regression import *
from pycaret.classification import *
from streamlit_pandas_profiling import st_profile_report

with st.sidebar:
    st.title('MyApp')
    choice = st.radio('navigation', ['1', '2', '3', '4'])

if 'tmp_flg' not in st.session_state:
    st.session_state['tmp_flg'] = True
if choice == "1":
    # st.session_state
    if st.session_state['tmp_flg']:
        st.title('upload data for modelling')
        file = st.file_uploader("drop your file here")
        st.session_state['file'] = file

        if file:
            st.session_state['tmp_flg'] = False
            df = pd.read_csv(file, index_col=None)
            st.session_state['df'] = df
            st.dataframe(df)
            # tmp_flg=False
            if st.button('generate report'):
                report = df.profile_report()
                st_profile_report(report)

    else:
        st.dataframe(st.session_state['df'])
        # tmp_flg=False
        if st.button('generate report'):
            report = st.session_state['df'].profile_report()
            st_profile_report(report)


elif choice == '2':
    try:
        # st.session_state
        # st.write(st.session_state.df)
        # st.dataframe(st.session_state['df'])
        # edited_df = st.experimental_data_editor(st.session_state['df'].columns,num_rows='dynamic')
        # st.write(st.session_state['df'].columns)
        # st.text_area(st.session_state['df'].columns)
        options = st.multiselect('Select columns you want to drop', st.session_state['df'].columns)

        if st.button('drop'):
            for i in options:
                st.session_state['df'] = st.session_state['df'].drop([i], axis=1)

    except:
        st.title("First Upload the data")


elif choice == '3':

    st.session_state

    try:
        text = st.text_area('code operations on data', value=st.session_state['text_area'])
    except:
        text = st.text_area('code operations on data')
    st.session_state.text_area = text
    st.code(text)

    if st.button('ok'):
        try:
            data = st.session_state.df
            data = eval(text)
            # data=data.drop(['age'], axis=1)
            st.session_state.df = data

        except:
            st.write('error,try again')
elif choice == '4':

    pass



