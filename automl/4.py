import pandas as pd
import streamlit as st
import ydata_profiling
# import streamlit_jupyter as sj
# from pycaret.regression import *
# from pycaret.classification import *

from streamlit_pandas_profiling import st_profile_report
# st.session_state.df=pd.read_csv(r'C:\Users\kevin\PycharmProjects\pythonProject\automl\income.csv')
with st.sidebar:
    st.title('MyApp')
    choice=st.radio('navigation',['1','2','3','4'])



if 'tmp_flg' not in st.session_state:
    st.session_state['tmp_flg'] =True
if choice=="1" :
    # st.session_state
    if st.session_state['tmp_flg']:
        st.title('upload data for modelling')
        file=st.file_uploader("drop your file here")
        st.session_state['file']=file

        if file :
            st.session_state['tmp_flg'] =False
            df=pd.read_csv(file,index_col=None)
            st.session_state['df']=df
            st.dataframe(df)
            # tmp_flg=False
        if st.button('ok'):
            st.session_state['tmp_flg'] = False
        # if st.button('generate report'):
        #     report = df.profile_report()
        #     st_profile_report(report)

    else:

        st.dataframe(st.session_state['df'])
        # tmp_flg=False
        if st.button('generate report'):
            report = st.session_state['df'].profile_report()
            st.session_state['profile_report']=report
            st_profile_report(report)
        try:
            st_profile_report(st.session_state['profile_report'])
        except:
            pass

elif choice=='2':
    try:
        # st.session_state
        # st.write(st.session_state.df)
        # st.dataframe(st.session_state['df'])
        # edited_df = st.experimental_data_editor(st.session_state['df'].columns,num_rows='dynamic')
        # st.write(st.session_state['df'].columns)
        # st.text_area(st.session_state['df'].columns)
        options=st.multiselect('Select columns you want to drop',st.session_state['df'].columns)

        if st.button('drop'):
            for i in options:
                st.session_state['df'] = st.session_state['df'].drop([i], axis=1)

    except:
        st.title("First Upload the data")


    try:
        text=st.text_area('code operations on data',value=st.session_state['text_area'])
    except:
        text = st.text_area('code operations on data')
    st.session_state.text_area=text
    st.code(text)

    if st.button('ok'):
        try:
            data=st.session_state.df
            data=eval(text)
            # data=data.drop(['age'], axis=1)
            st.session_state.df=data

        except:
            st.write('error,try again')


elif choice=='3':

    st.session_state


elif choice=='4':
    data=st.session_state.df
    model = st.radio('Select Model', ['Regression', 'Classification', 'Custom'],horizontal=True)
    if model=="Regression":
        from pycaret.regression import *

        data = st.session_state.df
        if 'reg_flg' not in st.session_state:
            st.session_state['reg_flg'] = False
        if 'rdwd_flg' not in st.session_state:
            st.session_state['rdwd_flg'] = False
        target = st.selectbox('Select Target', data.columns)
        gpu_flg = False
        gpu = st.checkbox('enable gpu')
        if gpu:
            gpu_flg = True
        mdl = st.radio('Select Model', ['Compare and train best', 'Custom'], horizontal=True)

        if mdl == 'Compare and train best':
            data = st.session_state.df
            if st.button('Train'):
                st.session_state.reg_flg = True

            if st.session_state.reg_flg:
                exp_name = setup(data=data, target=target, use_gpu=gpu_flg)
                setup_df = pull()

                st.session_state.rsetup1 = setup_df
                # st.info("setup info")
                st.dataframe(setup_df)
                best_model = compare_models()
                best_df = pull()
                st.session_state.rbest_df1 = best_df
                st.dataframe(best_df)

                model = create_model(best_model)
                model_df = pull()
                st.session_state.rmodel_df1 = model_df
                st.dataframe(model_df)

                save_model(model, "saved_model")
                st.session_state['rdwd_flg'] = True

            if st.session_state.reg_flg == False:
                try:
                    st.session_state.rsetup1
                except:
                    pass
                try:
                    st.session_state.rbest_df1
                except:
                    pass
                try:
                    st.session_state.rmodel_df1
                except:
                    pass
            if st.session_state['rdwd_flg']:
                with open('saved_model.pkl', 'rb') as f:
                    st.download_button("Download model", f, "trained_model.pkl")
                    st.session_state.reg_flg = False

        if mdl == 'Custom':
            data = st.session_state.df
            model_name = st.text_input('enter model name')
            # st.session_state.model=model_name
            if st.button('Train'):
                st.session_state.reg_flg = True

            if st.session_state.reg_flg:
                exp_name = setup(data=data, target=target, use_gpu=gpu_flg)
                setup_df = pull()
                st.session_state.rsetup2 = setup_df
                # st.info("setup info")
                # st.dataframe(setup_df)
                setup_df

                model = create_model(model_name)
                model_df = pull()
                st.session_state.rmodel_df2 = model_df
                # st.dataframe(model_df)
                model_df
                save_model(model, "saved_model")
                st.session_state['rdwd_flg'] = True

            if st.session_state.reg_flg == False:
                try:
                    st.session_state.rsetup2
                except:
                    pass
                try:
                    st.session_state.rmodel_df2
                except:
                    pass
            if st.session_state['rdwd_flg']:
                with open('saved_model.pkl', 'rb') as f:
                    st.download_button("Download model", f, "trained_model.pkl")
                    st.session_state.reg_flg = False





        # target = st.selectbox('Select Target', data.columns)
        # stp = st.radio('Select Setup', ['Default', 'Custom'],horizontal=True)
        # if st.button('submit'):
        #     if stp=="Default":
        #         setup_txt = st.empty()
        #         setup_txt.write('setuping....')
        #         exp_name = setup(data=data, target=target)
        #         setup_info=pull()
        #         setup_txt.write('this is setup info')
        #         setup_info
        # mdl=st.radio('Select Model', ['Compare', 'Custom'],horizontal=True)
        # # if mdl=="compare":



        # if stp=="custom":
        #     custom_setup=st.text_area('write setup')
        #     if st.button('ok'):
        #         exp_name=eval(custom_setup)
    if model=="Classification":
        from pycaret.classification import *
        data = st.session_state.df
        if 'cls_flg' not in st.session_state:
            st.session_state['cls_flg'] = False
        if 'dwd_flg' not in st.session_state:
            st.session_state['dwd_flg'] = False
        target = st.selectbox('Select Target', data.columns)
        gpu_flg=False
        gpu = st.checkbox('enable gpu')
        if gpu:
            gpu_flg=True
        mdl = st.radio('Select Model', ['Compare and train best', 'Custom'], horizontal=True)



        if mdl=='Compare and train best':
            data = st.session_state.df
            if st.button('Train'):
                st.session_state.cls_flg =True

            if st.session_state.cls_flg:
                exp_name = setup(data=data, target=target , use_gpu=gpu_flg)
                setup_df = pull()

                st.session_state.setup1=setup_df
                # st.info("setup info")
                st.dataframe(setup_df)
                best_model=compare_models()
                best_df=pull()
                st.session_state.best_df1 = best_df
                st.dataframe(best_df)

                model = create_model(best_model)
                model_df = pull()
                st.session_state.model_df1 = model_df
                st.dataframe(model_df)

                save_model(model, "saved_model")
                st.session_state['dwd_flg']=True

            if st.session_state.cls_flg == False:
                try:
                    st.session_state.setup1
                except:
                    pass
                try:
                    st.session_state.best_df1
                except:
                    pass
                try:
                    st.session_state.model_df1
                except:
                    pass
            if st.session_state['dwd_flg']:
                with open('saved_model.pkl', 'rb') as f:
                    st.download_button("Download model", f, "trained_model.pkl")
                    st.session_state.cls_flg = False

        if mdl=='Custom':
            data = st.session_state.df
            model_name=st.text_input('enter model name')
            # st.session_state.model=model_name
            if st.button('Train'):
                st.session_state.cls_flg = True

            if st.session_state.cls_flg:
                exp_name = setup(data=data, target=target, use_gpu=gpu_flg)
                setup_df = pull()
                st.session_state.setup2=setup_df
                # st.info("setup info")
                # st.dataframe(setup_df)
                setup_df


                model = create_model(model_name)
                model_df = pull()
                st.session_state.model_df2=model_df
                # st.dataframe(model_df)
                model_df
                save_model(model,"saved_model")
                st.session_state['dwd_flg'] = True



            if st.session_state.cls_flg == False:
                try:
                    st.session_state.setup2
                except:
                    pass
                try:
                    st.session_state.model_df2
                except:
                    pass
            if st.session_state['dwd_flg']:
                with open('saved_model.pkl', 'rb') as f:
                    st.download_button("Download model", f, "trained_model.pkl")
                    st.session_state.cls_flg = False





        #
        # if st.button('Setup'):
        #
        #     setup_txt = st.empty()
        #     st.session_state.setup_txt=setup_txt
        #     setup_txt.write('setuping....')
        #     exp_name = setup(data=data, target=target)
        #     setup_info = pull()
        #     st.session_state.setup_info=setup_info
        # try:
        #     st.session_state.setup_txt.write('this is setup info')
        #     st.session_state.setup_info
        # except:
        #     pass
        #
        # cmp=st.checkbox('want to compare models?')
        # if cmp:
        #     best_models=compare_models()
        #     st.session_state.best_model = best_models
        #     compare=pull()
        #     st.session_state.compare=compare
        # try:
        #     st.dataframe(st.session_state.compare)
        # except:
        #     pass
        #
        # name=st.text_input('Write model name')
        # mdl = st.checkbox('take best?')
        # if mdl:
        #     model=create_model(st.session_state.best_model)
        #     model_df=pull()
        #     model_df
        #
        # # if st.checkbox('show available models'):
        # #     st.write()
        # if st.button('Create Model'):
        #     model=create_model(name)
        #     st.session_state.model=model
        #     model_pull=pull()
        #     save_model(model,"save")
        #     st.session_state.model_pull=model_pull
        # try:
        #     st.session_state.model_pull
        # except:
        #     pass











