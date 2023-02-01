import streamlit as st
import graphviz as graph
import matplotlib.pyplot as plt
import altair as alt
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import hiplot as hip
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.features import ParallelCoordinates
import plotly.graph_objects as go
from sklearn import datasets
##################################################################################################################################################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
##################################################################################################################################################################
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, ExpSineSquared,DotProduct
##################################################################################################################################################################
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn import metrics
##################################################################################################################################################################

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

##################################################################################################################################################################

st.markdown(""" <style> .font_title {
font-size:50px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header {
font-size:50px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:35px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subsubheader {
font-size:28px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:26px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subtext {
font-size:18px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px; font-family: 'times';
}
</style>
"""

####################################################################################################################################################################

st.markdown('<p class="font_title">Classification Playground</p>', unsafe_allow_html=True)

####################################################################################################################################################################

st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png")
cols = st.columns([2, 4 , 2])
with cols[0].expander("Calming Video"):
    st.video("https://www.youtube.com/watch?v=SA8lu-m2IZY",start_time=1)
with cols[1]:
    st.markdown('<p class="font_text"> The idea explored here is to investigate different classification methods and their accuracies regarding several dataset including MNIST_Digits, Iris, and Penguin (for now). With that in mind, multiple classification functions and some of their hyper-parameters are studied to measure their impact on the accuracy of each classifier. </p>', unsafe_allow_html=True)
with cols[2].expander("Calming Video"):
    st.video("https://www.youtube.com/watch?v=_kT38XB1YHo&t=5s",start_time=1)
####################################################################################################################################################################

st.write(font_css, unsafe_allow_html=True)
tab = st.tabs(["Binary/Multi-Label Classification", "Case Study: Binary Classification of Biased Dataset"])

with tab[0]:
    cols=st.columns(7,gap='medium')
    Dataset_Name = cols[0].selectbox( 'Choose dataset for binary classification',('MNIST_Digits', 'Iris','Penguin','MNIST_Fashion'),index=1)
    if Dataset_Name == 'MNIST_Digits':
        # A= pd.concat(map(pd.read_csv, ['MNIST_1.csv', 'MNIST_2.csv','MNIST_3.csv','MNIST_4.csv','MNIST_5.csv','MNIST_6.csv','MNIST_7.csv']), ignore_index=True)
        A = pd.read_csv('MNIST_1.csv')
        X=A.iloc[:,1:].to_numpy()
        y=A.iloc[:,0].to_numpy()
        labels=np.unique(y)
        
        num_to_plot = 20 # plotting the first 16 images in the dataset
        #fig=plt.figure(figsize=(60,60))
        Visualizaiton = cols[1].checkbox('Visualize the investigated data?', value=False)

        if Visualizaiton==True:
            fig = px.imshow(X[:num_to_plot, :].reshape(num_to_plot,28,28), animation_frame=0,height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        
    elif Dataset_Name == 'Iris':
        X=datasets.load_iris().data
        y=datasets.load_iris().target
        labels=np.unique(y)
        
        Visualizaiton = cols[1].checkbox('Visualize the investigated data?', value=False)

        if Visualizaiton==True:
            df = px.data.iris()
            fig1 = px.scatter_matrix(df,dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species",width=800, height=800)
            st.plotly_chart(fig1, use_container_width=True)
    
    elif Dataset_Name == 'MNIST_Fashion':
        A = pd.read_csv('Fashion_MNIST.csv')
        X=A.iloc[:,1:].to_numpy()
        y=A.iloc[:,0].to_numpy()
        labels=np.unique(y)
        
        num_to_plot = 20 # plotting the first 16 images in the dataset
        #fig=plt.figure(figsize=(60,60))
        Visualizaiton = cols[1].checkbox('Visualize the investigated data?', value=False)

        if Visualizaiton==True:
            fig = px.imshow(X[:num_to_plot, :].reshape(num_to_plot,28,28), animation_frame=0,height=400)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        df = sns.load_dataset("penguins")
        df=df.dropna(axis=0)
        Visualizaiton = cols[1].checkbox('Visualize the investigated data?', value=False)

        if Visualizaiton==True:
            fig2 = px.scatter_matrix(df,dimensions=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"], color="species",width=800, height=800)
            st.plotly_chart(fig2, use_container_width=True)
        X=df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]].to_numpy()
        y=df
        labels=y.species.unique()
        y=y['species'].replace(labels,np.arange(len(labels))).to_numpy()
        labels=np.unique(y)
    
    # Binary Classification
    
    Classification_Method = cols[3].selectbox( 'Classification type?',('Binary', 'Multi-Label'),index=0)
    
    if Classification_Method == 'Binary':
        Classification_Option = cols[4].selectbox( 'Binary classification option?',('One vs. One', 'One vs. Rest'),index=1)
        if Dataset_Name == 'MNIST_Digits':
            if Classification_Option =='One vs. One':
                First_Label = cols[5].slider('First class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))-1)
                Second_Label = cols[6].slider( 'Second class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))+1)
            else:
                First_Label = cols[5].slider('First class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))-1)
        elif Dataset_Name == 'Iris':
            if Classification_Option =='One vs. One':
                First_Label = cols[5].slider('First class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))-1)
                Second_Label = cols[6].slider( 'Second class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))+1)
            else:
                First_Label = cols[5].slider('First class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))-1)
        else:
            if Classification_Option =='One vs. One':
                First_Label = cols[5].slider('First class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))-1)
                Second_Label = cols[6].slider( 'Second class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))+1)
            else:
                First_Label = cols[5].slider('First class label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels))-1)

        if Classification_Option =='One vs. One':
            X1 = np.copy(X)
            y1 = np.copy(y)
            X1_label_1 = X1[y1==First_Label,:]
            X1_label_2 = X1[y1==Second_Label,:]
            y1_label_1 = y1[y1==First_Label]
            y1_label_2 = y1[y1==Second_Label]
            X_New = np.append(X1_label_1,X1_label_2,axis=0)
            y_New = np.append(y1_label_1,y1_label_2,axis=0)
        else:
            X_New = np.copy(X)
            y1 = np.copy(y)
            a=np.where(labels!=First_Label)
            y1[y1!=First_Label] = a[0][0]
            y_New=np.copy(y1)
    else:
        if Dataset_Name == 'MNIST_Digits':
            Labels_Options_1 = cols[4].multiselect('Choose labels from 0 to 4:',['0', '1', '2', '3'],default = '0')
            Labels_Options_2 = cols[5].multiselect('Choose labels from 4 to 6:',['4', '5', '6'],default = '4')
            Labels_Options_3 = cols[6].multiselect('Choose labels from 7 to 9:',['7', '8', '9'],default = '7')
            Labels_Option=Labels_Options_1+Labels_Options_2+Labels_Options_3
        elif Dataset_Name == 'MNIST_Fashion':
            Labels_Options_1 = cols[4].multiselect('Choose labels from 0 to 4:',['0', '1', '2', '3'],default = '0')
            Labels_Options_2 = cols[5].multiselect('Choose labels from 4 to 6:',['4', '5', '6'],default = '4')
            Labels_Options_3 = cols[6].multiselect('Choose labels from 7 to 9:',['7', '8', '9'],default = '7')
            Labels_Option=Labels_Options_1+Labels_Options_2+Labels_Options_3
        elif Dataset_Name == 'Penguin':
            Labels_Options_1 = cols[4].multiselect('Choose labels from 0 to 2:',['0', '1', '2'],default = '0')
            Labels_Options_2 = cols[5].multiselect('Choose labels from 0 to 2:',['0', '1', '2'],default = '1')
            Labels_Options_3 = cols[6].multiselect('Choose labels from 0 to 2:',['0', '1', '2'],default = '2')
            Labels_Option=Labels_Options_1+Labels_Options_2+Labels_Options_3
        else:
            Labels_Options_1 = cols[4].multiselect('Choose labels from 0 to 2:',['0', '1', '2'],default = '0')
            Labels_Options_2 = cols[5].multiselect('Choose labels from 0 to 2:',['0', '1', '2'],default = '1')
            Labels_Options_3 = cols[6].multiselect('Choose labels from 0 to 2:',['0', '1', '2'],default = '2')
            Labels_Option=Labels_Options_1+Labels_Options_2+Labels_Options_3
        Labels = list(map(int, Labels_Option))
        y1=[]
        for i in Labels:
            y1=np.append(y1,y[y==i],axis=0)
            if i==Labels[0]:
                X1= X[y==i,:]
            else:
                X1= np.append(X1,X[y==i,:],axis=0)
        
        X_New=np.copy(X1)
        y_New=np.copy(y1)
    
    Classifier_List = ['Nearest Neighbors', 'Support Vector Machine',
                   'Decision Tree','Random Forest','Neural Network','Ada Boost',
                   'Naive Bayes', 'Quadratic Discriminant Analysis']#, 'Gaussian Process']
    Classification_Object=cols[2].selectbox( 'Classifier Object?', Classifier_List,index=0)
    
    # Creating Classifier Object
    
    if Classification_Object == 'Nearest Neighbors':
        cols1_KNN = st.columns(5,gap='medium')
        
        Neighbor_KNN = cols1_KNN[0].slider('Number of neighbors KNN:', 1, 20, 5,format='%i')
        Weights_KNN = cols1_KNN[1].select_slider('Select weight function used in prediction KNN:',options=['uniform', 'distance'],value='uniform')
        Algorithm_KNN = cols1_KNN[2].select_slider('Select algorithm used to compute the nearest neighbors KNN:',options=['auto', 'ball_tree', 'kd_tree', 'brute'],value='auto')
        Power_Distance_KNN = cols1_KNN[3].slider('Minkowski-distance power KNN:', 1, 20, 2,format='%i')
        Leaf_Size_KNN=30
        if Algorithm_KNN == 'kd_tree' or Algorithm_KNN == 'ball_tree':
            Leaf_Size = cols1_KNN[4].number_input('Choose the leaf size:',value=30,format='%i')
        
        Estimator = KNeighborsClassifier(n_neighbors=Neighbor_KNN, weights=Weights_KNN, algorithm=Algorithm_KNN, p=Power_Distance_KNN, metric='minkowski',leaf_size=Leaf_Size_KNN)
    
    elif Classification_Object == 'Support Vector Machine':
        cols1_SVC  = st.columns(7,gap='medium')
        
        Regulizer_SVC = cols1_SVC [0].number_input('Choose the value of regularization parameter SVC:',value=1.00)
        Kernel_SVC = cols1_SVC [1].select_slider('Choose a kernel function SVC:',options=['linear', 'poly','rbf','sigmoid'],value='linear')
        Degree_SVC = 3
        Gamma_SVC = 'scale'
        if Kernel_SVC == 'poly':
            Degree_SVC = cols1_SVC [6].slider('Degree of the polynomial kernel function SVC:', 1, 20, 3,format='%i')
        
        if Kernel_SVC == 'poly' or Kernel_SVC == 'rbf' or Kernel_SVC == 'sigmoid':
            Gamma_SVC = cols1_SVC [5].select_slider('Choose kernel coefficient SVC:',options=['scale','auto'],value='scale')
        
        Random_State_SVC = cols1_SVC [2].slider('Seed number for random shuffeling SVC:', 1,200, value=45,format='%i')
        Tolerance_SVC = cols1_SVC [3].number_input('Stopping tolerance value for SVC between 0.00001 and 0.01:', min_value=0.00001,max_value=0.01, value=0.001,step=0.00001,format='%f')
        Max_Iteration_SVC = cols1_SVC [4].number_input('Limit for number of iteration SVC:', min_value=-1,max_value=100000, value=-1,step=100,format='%i')
        
        Estimator = SVC(C=Regulizer_SVC, kernel=Kernel_SVC, degree=Degree_SVC, gamma=Gamma_SVC, tol=Tolerance_SVC, max_iter=-1, random_state=Random_State_SVC)
               
    elif Classification_Object == 'Gaussian Process':
        cols1_GPC = st.columns(5,gap='medium')
        
        Kernel_Label_GPC = cols1_GPC[0].selectbox('Select kernel function for GPC:',['RBF', 'RationalQuadratic', 'Matern', 'ExpSineSquared','DotProduct'],index = 0)        
        Length_Scale_GPC = cols1_GPC[1].number_input('Input a value for length scale GPC:',value=1.0,format='%f')
        if Kernel_Label_GPC == 'RBF':
            Kernel_GPC = RBF(Length_Scale_GPC, (1e-5, 1e5))
        elif Kernel_Label_GPC == 'RationalQuadratic':
            Kernel_GPC = RationalQuadratic(length_scale=Length_Scale_GPC, alpha=1.0)
        elif Kernel_Label_GPC == 'Matern':
            Kernel_GPC = Matern(length_scale=Length_Scale_GPC, length_scale_bounds=(1e-05, 100000.0), nu=4.5)
        elif Kernel_Label_GPC == 'ExpSineSquared':
            Kernel_GPC = ExpSineSquared(length_scale=Length_Scale_GPC, periodicity=3.0, length_scale_bounds=(1e-05, 100000.0), periodicity_bounds=(1e-05, 100000.0))
        else:
            Kernel_GPC = DotProduct()
        Random_State_GPC = cols1_GPC[2].slider('Seed number for random shuffeling GPC:', 1,200, value=45,format='%i')
        Restart_Optimizer_GPC = cols1_GPC[3].slider('The number of restarts of the kernel’s optimizer GPC', 0, 1000, 200,format='%i')
        Max_Iteration_GPC = cols1_GPC[4].slider('Limit for number of iteration GPC:', 1,10000, value=300,format='%i')
    
        Estimator = GaussianProcessClassifier(kernel=Kernel_GPC, n_restarts_optimizer=Restart_Optimizer_GPC, max_iter_predict=Max_Iteration_GPC, random_state=Random_State_GPC)
    
    elif Classification_Object == 'Decision Tree':
        cols1_DTC = st.columns(7,gap='medium')
        
        Criterion_DTC = cols1_DTC[0].selectbox('Function to measure the quality of split DTC:',['gini', 'entropy', 'log_loss'],index = 0)
        Splitter_DTC = cols1_DTC[1].selectbox('Split strategy at each node DTC:',['best', 'random'],index = 0)
        Max_Feature_DTC = cols1_DTC[2].selectbox('Method for number of features used in split DTC:',['auto', 'sqrt', 'log2'],index = 0)
        Random_State_DTC = cols1_DTC[3].slider('Seed number for random shuffeling DTC:', 1,200, value=45,format='%i')
        Max_Depth_DTC = cols1_DTC[4].slider('Maximum Depth of Tree DTC:', 1,100, value=10,format='%i')
        Min_Samples_Split_DTC = cols1_DTC[5].slider('Minimum number of samples for split at internal node DTC:', 1,10, value=2,format='%i')
        Min_Samples_Leaf_DTC = cols1_DTC[6].slider('Minimum number of samples for each leaf node DTC:', 1,10, value=1,format='%i')
        
        Estimator = DecisionTreeClassifier(criterion=Criterion_DTC, splitter=Splitter_DTC, max_depth=Max_Depth_DTC, max_features=Max_Feature_DTC, min_samples_leaf=Min_Samples_Leaf_DTC, min_samples_split=Min_Samples_Split_DTC, random_state=Random_State_DTC)
    
    elif Classification_Object == 'Random Forest':
        cols1_RFC = st.columns(8,gap='medium')
        
        N_Estimators_RFC = cols1_RFC[0].slider('Number of trees in forest RFC:', 1,400, value=100,format='%i')
        Criterion_RFC = cols1_RFC[1].selectbox('Function to measure the quality of split RFC:',['gini', 'entropy', 'log_loss'],index = 0)
        Max_Feature_RFC = cols1_RFC[2].selectbox('Method for number of features used in split RFC:',['auto', 'sqrt', 'log2'],index = 0)
        Random_State_RFC = cols1_RFC[3].slider('Seed number for random shuffeling RFC:', 1,200, value=45,format='%i')
        Bootstraping_RFC = cols1_RFC[4].checkbox('Training data bootstrapping RFC?')
        Max_Depth_RFC = cols1_RFC[5].slider('Maximum Depth of Tree RFC:', 1,100, value=10,format='%i')
        Min_Samples_Split_RFC = cols1_RFC[6].slider('Minimum number of samples for split at internal node RFC:', 1,10, value=2,format='%i')
        Min_Samples_Leaf_RFC = cols1_RFC[7].slider('Minimum number of samples for each leaf node RFC:', 1,10, value=1,format='%i')
        
        Estimator = RandomForestClassifier(n_estimators=N_Estimators_RFC, criterion=Criterion_RFC, max_features=Max_Feature_RFC, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=Bootstraping_RFC, random_state=Random_State_RFC)
                                   
    elif Classification_Object == 'Neural Network':
        cols1 = st.columns(9,gap='medium')
        
        Num_Hidden_Layer = cols1[0].slider('Number of Hidden Layers for NN Classifier: ',1,20, value=2,format='%i')
        Activation = cols1[1].selectbox('Select activation function for NN Classifier:',['identity', 'relu', 'logistic', 'tanh'],index = 0)
        Solver = cols1[2].selectbox('Select solver type for NN Classifier:',['adam', 'sgd', 'lbfgs'],index = 0)
        Alpha = cols1[3].number_input('Alpha (non-negative) for NN Classifier: ',value=0.01,format='%f')
        Learning_Rate = cols1[4].selectbox('Select learning rate type for NN Classifier:',['constant', 'invscaling', 'adaptive'],index = 0)
        Learning_Rate_Init = cols1[5].number_input('Initial learning rate for NN Classifier: ',value=0.001,format='%f')
        Max_Iteration = cols1[6].slider('Number of iteration for NN Classifier:', 0, 20000, 200,format='%i')
        Random_State = cols1[7].slider('Random state for NN Classifier:', 0, 200, 40,format='%i')
        Tolerence = cols1[8].number_input('Tolerence value for NN Classifier: ',value=0.0001,format='%f')
        
        cols2 = st.columns(Num_Hidden_Layer)
        Num_Neuron=np.zeros(Num_Hidden_Layer)
        for j in range (Num_Hidden_Layer):
            with cols2[j]:
                if Dataset_Name == 'MNIST_Digits':
                    Num_Neuron[j] = st.slider('Number of Neurons in '+str(j+1)+' Hidden Layer:',1, 1000, value=200,format='%i')
                elif Dataset_Name == 'Iris':
                    Num_Neuron[j] = st.slider('Number of Neurons in '+str(j+1)+' Hidden Layer:',1, 20, value=6,format='%i')
                else:
                    Num_Neuron[j] = st.slider('Number of Neurons in '+str(j+1)+' Hidden Layer:',1, 20, value=10,format='%i')
        Num_Neuron=Num_Neuron.astype(int)
        
        Estimator = MLPClassifier(hidden_layer_sizes=Num_Neuron, activation=Activation, solver=Solver, alpha=Alpha,
                          batch_size='auto', learning_rate=Learning_Rate, learning_rate_init=Learning_Rate_Init,
                          max_iter=Max_Iteration, random_state=Random_State, tol=Tolerence)

    elif Classification_Object == 'Ada Boost':
        Estimator = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME', random_state=30)

    elif Classification_Object == 'Naive Bayes':
        Estimator = GaussianNB()

    else:
        Estimator = QuadraticDiscriminantAnalysis()    
        
    # Scaling Section
    
    cols = st.columns([2,1,3])
    Scaler = cols[0].checkbox('Scaling Data?')
    if Scaler:
        Scaler_Type = cols[2].select_slider('Select scaler object:',['Min-Max Scaler', 'Standard Scaler', 'Max-Abs Scaler'],value = 'Standard Scaler')
        if Scaler_Type == 'Min-Max Scaler':
            Scaler_Object = MinMaxScaler()
        elif Scaler_Type == 'Standard Scaler':
            Scaler_Object = StandardScaler()
        else:
            Scaler_Object = MaxAbsScaler()
        
        X_Classification = Scaler_Object.fit_transform(X_New)
    
    else:
        X_Classification = np.copy(X_New)
    
    y_Classification = np.copy(y_New)
    
    # tabs = st.tabs(["Cross Validation Score", "Confusion Matrix Visualizaiton", "Precision-Recall Curve" , "Receiver Operating Characteristic"])
    
    # if Classification_Method == 'Binary':
    if Classification_Method == 'Binary':
        cols2 = st.columns(2,gap='medium')
    else:
        cols2 = st.columns(2,gap='medium')
        
    with cols2[1]:
        Test_Size = st.number_input('Test Split:', min_value=0.0, max_value=1.0,step=0.01, value=0.2,format='%f')
        Random_State = st.slider('Random state for Splitter:', 0, 200, 173)
        X_train, X_test, y_train, y_test = train_test_split(X_Classification, y_Classification,random_state=Random_State,test_size=Test_Size)
        Estimator.fit(X_train, y_train)
        predictions = Estimator.predict(X_Classification)
        cm = confusion_matrix(y_Classification, predictions, labels=Estimator.classes_)
        Labels_Confuse = np.unique(y_Classification)
        Tickvals=[]
        Ticktext=[]
        for i in Labels_Confuse:
            Tickvals=np.append(Tickvals,int(i))
            Ticktext=np.append(Ticktext,str(int(i)))
        fig=px.imshow(cm, x=Estimator.classes_, y=Estimator.classes_, labels=dict(x="Predicted Labels", y="True Labels", color="Count"), text_auto=True,height=np.maximum(int(Tickvals.shape[0])*100,400),width=np.maximum(int(Tickvals.shape[0])*100,400))
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = Tickvals,
                ticktext = Ticktext
            )
        )
        fig.update_layout(
            yaxis = dict(
                tickmode = 'array',
                tickvals = Tickvals,
                ticktext = Ticktext
            )
        )
        st.plotly_chart(fig, use_container_width=True)
            
    with cols2[0]:
        CV_Number = st.slider('Number of cross validation folds:', 2, 30, 5,format='%i')
        CV_Score=cross_val_score(Estimator, X_Classification, y_Classification, cv=CV_Number)
        CV_X = np.arange(1,CV_Number+1)
        fig = px.bar(x=CV_X, y=CV_Score,labels={'x': "", 'y': "Score" })#, color=CV_X,color_discrete_sequence=px.colors.qualitative.G10)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
            
#     st.markdown('<p class="font_header">* Something to think about</p>', unsafe_allow_html=True)
#     st.markdown('<p class="font_text">Question 1: What is your interpertation of Iris or Penguin dataset using visualization?</p>', unsafe_allow_html=True)
#     st.markdown('<p class="font_text">Question 2: What does confusion matrix tell you?</p>', unsafe_allow_html=True)
#     st.markdown('<p class="font_text">Question 3: What is the importance of cross-validation for the investigated datasets? What does it represents?</p>', unsafe_allow_html=True)
#     st.markdown('<p class="font_text">Question 4: Precision-Recall accuracy is a measure of a classifier performance. How can you calculate this using confusion matrix for a classification method (Try to use Penguin or MNIST_Digits dataset)? </p>', unsafe_allow_html=True)
    
##################################################################################################################################################################

with tab[1]:
    st.markdown('<p class="font_text">Next, for MNIST_Digits dataset, we want to explore how unbalance dataset affect the performance of classification method for one label versus the rest. </p>', unsafe_allow_html=True)
    cols = st.columns([4,2,2,2,2,2,2])
        
    A = pd.read_csv('MNIST_1.csv')
    X=A.iloc[:,1:].to_numpy()
    y=A.iloc[:,0].to_numpy()
    labels=np.unique(y)
    
    source = pd.DataFrame({"Label": ['0','1','2','3','4','5','6','7','8','9'], "Count": [y[y==0].shape[0],y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0],y[y==4].shape[0],
                                 y[y==5].shape[0],y[y==6].shape[0],y[y==7].shape[0],y[y==8].shape[0],y[y==9].shape[0]]})

    d=alt.Chart(source).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Label", type="nominal",scale=alt.Scale(scheme='rainbow')),
        tooltip=("Label","Count")
    ).interactive()

    cols[0].altair_chart(d, use_container_width=True)
    
    Classifier_List = ['Nearest Neighbors', 'Support Vector Machine',
                   'Decision Tree','Random Forest','Neural Network']#, 'Gaussian Process']
    Classification_Object = cols[1].selectbox( 'Classification Method?', Classifier_List,index=0)
    
    First_Label_Biased = cols[2].slider('Choose Label:', int(np.min(labels)), int(np.max(labels)), int(np.median(labels)),format='%i')
    
    Train_Size = cols[3].number_input('Train/Test Split:', min_value=0.0, max_value=1.0,step=0.01, value=0.8,format='%f')
    
    Overall_Count = y.shape[0]
    Desired_Count = y[y==int(First_Label_Biased)].shape[0]
    
    Biased_Ratio = cols[4].number_input('Count Ratio for Rest of the Labels vs Selected Label:', min_value=0.1, max_value=8.0,step=0.01, value=1.0,format='%f')
    
    st.markdown('<p class="font_text">Explanation for "Count Ratio for Rest of the Labels vs Selected Label": This ratio define the count for the other label based on the count of the investigated label, e.g. for count ratio 1.0, if your selected label (default label 4) has 900 rows in the dataset, the counts for the other label would be 1*900 which is 900 rows.  </p>', unsafe_allow_html=True)
    
    # Creating Classifier Object
    
    if Classification_Object == 'Nearest Neighbors':
        cols1_KNN = st.columns(5,gap='medium')
        
        Neighbor_KNN = cols1_KNN[0].slider('Neighbors Number KNN:', 1, 20, 5,format='%i')
        Weights_KNN = cols1_KNN[1].select_slider('Weight function used in prediction KNN:',options=['uniform', 'distance'],value='uniform')
        Algorithm_KNN = cols1_KNN[2].select_slider('Algorithm used to compute the nearest neighbors KNN:',options=['auto', 'ball_tree', 'kd_tree', 'brute'],value='auto')
        Power_Distance_KNN = cols1_KNN[3].slider('Minkowski-distance power for KNN:', 1, 20, 2,format='%i')
        Leaf_Size_KNN=30
        if Algorithm_KNN == 'kd_tree' or Algorithm_KNN == 'ball_tree':
            Leaf_Size = cols1_KNN[4].number_input('Leaf size:',value=5,format='%i')
        
        Estimator = KNeighborsClassifier(n_neighbors=Neighbor_KNN, weights=Weights_KNN, algorithm=Algorithm_KNN, p=Power_Distance_KNN, metric='minkowski',leaf_size=Leaf_Size_KNN)
    
    elif Classification_Object == 'Support Vector Machine':
        cols1_SVC  = st.columns(7,gap='medium')
        
        Regulizer_SVC = cols1_SVC [0].number_input('Value of regularization parameter SVC:',value=1.00)
        Kernel_SVC = cols1_SVC [1].select_slider('Kernel function SVC:',options=['linear', 'poly','rbf','sigmoid'],value='rbf')
        Degree_SVC = 3
        Gamma_SVC = 'scale'
        if Kernel_SVC == 'poly':
            Degree_SVC = cols1_SVC [6].slider('Polynomial degree kernel function SVC:', 1, 20, 3,format='%i')
        
        if Kernel_SVC == 'poly' or Kernel_SVC == 'rbf' or Kernel_SVC == 'sigmoid':
            Gamma_SVC = cols1_SVC [5].select_slider('Kernel coefficient SVC:',options=['scale','auto'],value='scale')
        
        Random_State_SVC = cols1_SVC [2].slider('Seed number random shuffeling SVC:', 1,200, value=45,format='%i')
        Tolerance_SVC = cols1_SVC [3].number_input('Tolerance value for SVC between 0.00001 and 0.01:', min_value=0.00001,max_value=0.01, value=0.001,step=0.00001,format='%f')
        Max_Iteration_SVC = cols1_SVC [4].number_input('Iteration limit SVC:', min_value=-1,max_value=100000, value=-1,step=100,format='%i')
        
        Estimator = SVC(C=Regulizer_SVC, kernel=Kernel_SVC, degree=Degree_SVC, gamma=Gamma_SVC, tol=Tolerance_SVC, max_iter=-1, random_state=Random_State_SVC)
    
    elif Classification_Object == 'Decision Tree':
        cols1_DTC = st.columns(7,gap='medium')
        
        Criterion_DTC = cols1_DTC[0].selectbox('Measure the quality of split DTC:',['gini', 'entropy', 'log_loss'],index = 0)
        Splitter_DTC = cols1_DTC[1].selectbox('Split-strategy at each node DTC:',['best', 'random'],index = 0)
        Max_Feature_DTC = cols1_DTC[2].selectbox('Split-method for number of features in DTC:',['auto', 'sqrt', 'log2'],index = 0)
        Random_State_DTC = cols1_DTC[3].slider('Seed number random shuffeling DTC:', 1,200, value=45,format='%i')
        Max_Depth_DTC = cols1_DTC[4].slider('Maximum Depth Tree DTC:', 1,100, value=45,format='%i')
        Min_Samples_Split_DTC = cols1_DTC[5].slider('Minimum number samples for split at internal node DTC:', 1,10, value=2,format='%i')
        Min_Samples_Leaf_DTC = cols1_DTC[6].slider('Minimum number samples for each leaf node DTC:', 1,10, value=1,format='%i')
        
        Estimator = DecisionTreeClassifier(criterion=Criterion_DTC, splitter=Splitter_DTC, max_depth=Max_Depth_DTC, max_features=Max_Feature_DTC, min_samples_leaf=Min_Samples_Leaf_DTC, min_samples_split=Min_Samples_Split_DTC, random_state=Random_State_DTC)
    
    elif Classification_Object == 'Random Forest':
        cols1_RFC = st.columns(8,gap='medium')
        
        N_Estimators_RFC = cols1_RFC[0].slider('Tree number in forest RFC:', 1,400, value=100,format='%i')
        Criterion_RFC = cols1_RFC[1].selectbox('Measure the quality of split RFC:',['gini', 'entropy', 'log_loss'],index = 0)
        Max_Feature_RFC = cols1_RFC[2].selectbox('Method for features number used in split RFC:',['auto', 'sqrt', 'log2'],index = 0)
        Random_State_RFC = cols1_RFC[3].slider('Seed number random shuffeling RFC:', 1,200, value=45,format='%i')
        Bootstraping_RFC = cols1_RFC[4].checkbox('Training data bootstrapping RFC')
        Max_Depth_RFC = cols1_RFC[5].slider('Tree maximum depth RFC:', 1,100, value=45,format='%i')
        Min_Samples_Split_RFC = cols1_RFC[6].slider('Minimum number of samples for internal node split  RFC:', 1,10, value=2,format='%i')
        Min_Samples_Leaf_RFC = cols1_RFC[7].slider('Minimum number of samples for leaf node RFC:', 1,10, value=1,format='%i')
        
        Estimator = RandomForestClassifier(n_estimators=N_Estimators_RFC, criterion=Criterion_RFC, max_features=Max_Feature_RFC, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=Bootstraping_RFC, random_state=Random_State_RFC)
                                   
    else:
        cols1 = st.columns(9,gap='medium')
        
        Num_Hidden_Layer = cols1[0].slider('Number of Hidden Layers NN Classifier: ',1,20, value=2,format='%i')
        Activation = cols1[1].selectbox('Select activation function NN Classifier:',['identity', 'relu', 'logistic', 'tanh'],index = 0)
        Solver = cols1[2].selectbox('Select solver type NN Classifier:',['adam', 'sgd', 'lbfgs'],index = 0)
        Alpha = cols1[3].number_input('Alpha (non-negative) NN Classifier: ',value=0.01,format='%f')
        Learning_Rate = cols1[4].selectbox('Select learning rate type NN Classifier:',['constant', 'invscaling', 'adaptive'],index = 0)
        Learning_Rate_Init = cols1[5].number_input('Initial learning rate NN Classifier: ',value=0.001,format='%f')
        Max_Iteration = cols1[6].slider('Number of iteration NN Classifier:', 0, 20000, 200,format='%i')
        Random_State = cols1[7].slider('Random state NN Classifier:', 0, 200, 40,format='%i')
        Tolerence = cols1[8].number_input('Tolerence value NN Classifier: ',value=0.0001,format='%f')
        
        cols2 = st.columns(Num_Hidden_Layer)
        Num_Neuron=np.zeros(Num_Hidden_Layer)
        for j in range (Num_Hidden_Layer):
            with cols2[j]:
                Num_Neuron[j] = st.slider('Neurons Number '+str(j+1)+' Hidden Layer:',1, 1000, value=200,format='%i')
                
        Num_Neuron=Num_Neuron.astype(int)
        
        Estimator = MLPClassifier(hidden_layer_sizes=Num_Neuron, activation=Activation, solver=Solver, alpha=Alpha,
                          batch_size='auto', learning_rate=Learning_Rate, learning_rate_init=Learning_Rate_Init,
                          max_iter=Max_Iteration, random_state=Random_State, tol=Tolerence)
                          
    X_Biased = X[y==int(First_Label_Biased),:]
    y_Biased = y[y==int(First_Label_Biased)]
    X_Rest = X[y!=int(First_Label_Biased),:]
    y_Rest = y[y!=int(First_Label_Biased)]
    
    Count = int(np.ceil(Biased_Ratio*Desired_Count))
    
    idx = np.random.choice(int(X_Rest.shape[0]), Count, replace=False)    
    
    X_Rest_Biased = X_Rest[idx,:]
    y_Rest_Biased = y_Rest[idx]
    a=np.where(labels!=First_Label_Biased)
    y_Rest_Biased[:] = a[0][0]
    
    X_Biased_Classification = np.append(X_Biased,X_Rest_Biased,axis=0)
    y_Biased_Classification = np.append(y_Biased,y_Rest_Biased,axis=0)
    
    st.write('Based on these hyperparameters, the classification dataset used for training and testing has the size of ',Count+int(X_Biased.shape[0]), ', where it has ', int(X_Biased.shape[0]) ,' for the investigated label (',First_Label_Biased,'), and rest represent other rows in MNIST_Digits dataset.')
    
    X_Classification = X_Biased_Classification
    y_Classification = y_Biased_Classification
    Scaler = cols[5].checkbox('Scale Data?')
    if Scaler:
        Scaler_Type = cols[6].select_slider('Select scaler object:',['Min-Max Scaler', 'Standard Scaler', 'Max-Abs Scaler'],value = 'Standard Scaler')
        if Scaler_Type == 'Min-Max Scaler':
            Scaler_Object = MinMaxScaler()
        elif Scaler_Type == 'Standard Scaler':
            Scaler_Object = StandardScaler()
        else:
            Scaler_Object = MaxAbsScaler()
        
        X_Classification = Scaler_Object.fit_transform(X_Biased_Classification)
    
    X_train, X_test, y_train, y_test = train_test_split(X_Classification, y_Classification,random_state=42,train_size=Train_Size)
    Estimator.fit(X_train, y_train)
    predictions = Estimator.predict(X_Classification)
    cm = confusion_matrix(y_Classification, predictions, labels=Estimator.classes_)
    Labels_Confuse = np.unique(y_Classification)
    Ready = st.checkbox("See confusion matrix for biased dataset?")
    
    if (Ready):
        fig=px.imshow(cm, x=Estimator.classes_, y=Estimator.classes_, labels=dict(x="Predicted Labels", y="True Labels", color="Count"), text_auto=True)
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = [Labels_Confuse[0],Labels_Confuse[1]],
                ticktext = [str(First_Label_Biased),'Rest']
            )
        )
        fig.update_layout(
            yaxis = dict(
                tickmode = 'array',
                tickvals = [Labels_Confuse[0],Labels_Confuse[1]],
                ticktext = [str(First_Label_Biased),'Rest']
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
#     st.markdown('<p class="font_header">* Something to think about</p>', unsafe_allow_html=True)
#     st.markdown('<p class="font_text">Question 1: What are the values in the figure on the right, when you move around the figure? </p>', unsafe_allow_html=True)
#     st.markdown('<p class="font_text">Question 2: How do you think hypothetical bias would affect the classification? </p>', unsafe_allow_html=True)
#     st.markdown('<p class="font_text">Question 3: Understanding the goal of confusion matrix, what are the impacts of train-test split size and count ratio on the classification? Do you think scaling could help improving the accuracy? What you can say about other classification technique (KNN is default estimator, try to use other classification methods)? </p>', unsafe_allow_html=True)

    # st.write(idx)
    
    

##################################################################################################################################################################
    
# with tab[2]:
    # st.markdown('<p class="font_header">Under Construction!!!!!! </p>', unsafe_allow_html=True)

##################################################################################################################################################################

st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013). </p>', unsafe_allow_html=True)

##################################################################################################################################################################
