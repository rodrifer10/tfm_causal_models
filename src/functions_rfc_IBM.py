import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from sklearn.impute import KNNImputer
from termcolor import colored, cprint
import scipy.stats as ss
import warnings
import importlib
from toolz import curry
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, fbeta_score, make_scorer,\
                            accuracy_score,average_precision_score, precision_recall_curve, roc_curve,\
                            auc, recall_score, precision_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import scikitplot as skplt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import graphviz
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

# Indice:
## 1. Funciones de la cátedra utilizadas (4)
## 2. Funciones propias (15)

###-------------------------------- Funciones de la cátedra utilizadas ---------------------------------------
def duplicate_columns(frame):
    '''
    Lo que hace la función es, en forma de bucle, ir seleccionando columna por columna del DF que se le indique
    y comparar sus values con los de todas las demás columnas del DF. Si son exactamente iguales, añade dicha
    columna a una lista, para finalmente devolver la lista con los nombres de las columnas duplicadas.
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups

### -----------------------

def dame_variables_categoricas(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_variables_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada con menos de 100 valores diferentes
            -- 1: la ejecución es incorrecta
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    lista_variables_categoricas = []
    other = []
    for i in dataset.columns:
        if (dataset[i].dtype!=float) & (dataset[i].dtype!=int):
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)

    return lista_variables_categoricas, other

### ----

def outliers_detection(df, list_vars, target, multiplier=3, list_out=False, list_threshold=0):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    outliers_index = []
    
    for i in list_vars:
        
        series_mean = df[i].mean()
        series_std = df[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = df[i].size
        
        perc_goods = df[i][(df[i] >= left) & (df[i] <= right)].size/size_s
        perc_excess = df[i][(df[i] < left) | (df[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(df[target][(df[i] < left) | (df[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                        pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop(target,axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = df[i][(df[i] < left) | (df[i] > right)].size
            pd_concat_percent['percentaje_sum_outlier_values'] = perc_excess
            pd_final = (pd.concat([pd_final, pd_concat_percent]
                                ,axis=0)
                        .reset_index(drop=True)
                        .sort_values(by='percentaje_sum_outlier_values'
                                    ,ascending=False))
            
        if perc_excess > list_threshold:
            outliers_index.append(df[i][(df[i] < left) | (df[i] > right)].index.values)
            
    if pd_final.empty:
        print('No existen variables con outliers')
    
    pd_final_styled = pd_final.style.background_gradient(cmap='YlOrRd', subset=['percentaje_sum_outlier_values'], vmin=0, vmax=0.1)
    
    outliers_index = list(set([item for sublist in outliers_index for item in sublist]))

    if list_out:
        return outliers_index
    else:
        return pd_final_styled

#####

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


###----------------------------------------- Funciones Propias -----------------------------------------------

def tipos_vars(df=None, show=True):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función tipos_vars:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe como argumento un dataframe, analiza cada una de sus variables y muestra
        en pantalla el listado, categorizando a cada una como "categoric","bool" o "numeric". Para
        variables categóricas y booleanas se muestra el listado de categorías. Si son numéricas solo
        se informa el Rango y la Media de la variable.
        Además, luego de imprimir la información comentada, la función devuelve 3 listas, cada una
        con los nombres de las variables pertenecientes a cada grupo ("bools", "categoric" y "numeric").
        El orden es: 1. bools, 2. categoric, 3. numeric.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- show: Argumento opcional, valor por defecto True. Si show es True, entonces se mostrará la
        información básica de con cada categoría. Si es False, la función solo devuelve las listas con
        los nombres de las variables según su categoría.
    - Return:
        -- list_bools: listado con el nombre de las variables booleanas encontradas
        -- list_cat: listado con el nombre de las variables categóricas encontradas
        -- list_num: listado con el nombre de las variables numéricas encontradas
    '''
    # Realizo una verificación por si no se introdujo ningún DF
    if df is None:
        print(u'No se ha especificado un DF para la función')
        return None
    
    # Genero listas vacías a rellenar con los nombres de las variables por categoría
    list_bools = []
    list_cat = []
    list_num = []
    
    # Analizo variables, completo las listas e imprimo la información de cada variable en caso de que el Show no se desactive
    for i in df.columns:
        if len(df[i].unique()) <= 2:
            list_bools.append(i)
            if show:
                print(f"{colored('(boolean)','blue')} - \033[4m{i}\033[0m :  {df[i].unique()}")
        elif len(df[i].unique()) < 40 or df[i].dtype in ('object', 'category', 'string'):
            list_cat.append(i)
            if len(df[i].unique()) < 40 and show:
                print(f"{colored('(categoric)','red')}(\033[1mType\033[0m: {df[i].dtype}) - \033[4m{i}\033[0m : {sorted(df[i].astype(str).unique())}")
            elif len(df[i].unique()) >= 40 and show:
                print(f"{colored('(categoric)','red')}\(\033[1mType\033[0m: {df[i].dtype}) - \033[4m{i}\033[0m : \033[1mUnique values\033[0m = {len(df[i].unique())}. Sample: {df[i].unique()[:5]} ... {df[i].unique()[-5:]}")
        else:
            list_num.append(i)
            if show:
                print(f"{colored('(numeric)','green')} - \033[4m{i}\033[0m : \033[1mRange\033[0m = [{df[i].min():.2f} to {df[i].max():.2f}], \033[1mMean\033[0m = {df[i].mean():.2f}")
    
    # Finalmente devuelvo las listas con los nombres de las variables por cada categoría
    return list_bools,list_cat,list_num

#####

def dame_info(var_list, df, df_info, only_desc=False, names_col='Variable', descrip_col='Description'):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_info:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función bastante específica para el trabajo de este DF de fraude, aunque bastante
    necesaria. Útil para llamarla mientras se analiza el comportamiento de X variables para obtener contexto
    sobre éstas. Recibe una lista de variables de las que se desea obtener una breve descripción, el DF en el
    que se encuentran dichas variables como parámetros obligatorios, y el DF con la descripción de las
    variables. Luego busca y trae las descripciones de cada variable solicitada desde el diccionario de datos,
    además de mostrar la cantidad de nulls y los valores que puede tomar cada variable.
    - Inputs:
        -- var_list: Listado de variables cuya descripción se necesita imprimir
        -- df: DataFrame de Pandas en donde se encuentran las variables a analizar
        -- only_desc: Permite elegir ver únicamente la descripción en texto de la variable sin datos extra
        -- df_info: DataFrame de Pandas en el que se encuentre el diccionario de datos
        -- names_col: Nombre de la columna de df_info en donde se encuentran los nombres de las variables a
        buscar
        -- descrip_col= Nombre de la columna del df_indo en donde se encuentra la descripción de cada variable
    - Return: No hay return, solo imprime cada descripción de cada variable solicitada.
    '''    
    
    for i,n in zip(var_list,range(1,(len(var_list)+1))):
        if i not in df.columns.values:
            print(f'\033[1m {n}. {i}\033[0m : Variable no encontrada en el DF')
        else:
            if only_desc:
                print(f'\033[1m {n}.',i,'\033[0m',':',df_info.set_index(names_col).loc[i,descrip_col],'')
            else:
                print(f'\033[1m {n}.',i,'\033[0m',':',df_info.set_index(names_col).loc[i,descrip_col],'')
                print(f'- Nulls: {df[i].isna().sum()}')
                if df[i].dtype in ['string','object']:
                    print('- Values:',df[i].unique(),'\n')
                elif (df[i].dtype in ['int','int64','bool','float','float64']) & (len(df[i].unique())<=10):
                    print('- Values:',df[i].sort_values().unique(),'\n')
                elif (df[i].dtype in ['float','float64','int','int64']) &  (len(df[i].unique())>10):
                    print(f"- Range = [{df[i].min():.2f} to {df[i].max():.2f}], Mean = {df[i].mean():.2f}\n")
                    
#####

def corr_cat(df,target=None,target_transform=False,cat_cols=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función corr_cat:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe como un dataframe, detecta las variables categóricas y calcula una especie de
        matriz de correlaciones mediante el uso del estadístico Cramers V. En la función se incluye la
        posibilidad de que se transforme a la variable target a string si no lo fuese y que se incluya en la
        lista de variables a analizar. Esto último  puede servir sobre todo para casos en los que la variable
        target es un booleano o está codificada.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- target: String con nombre de la variable objetivo
        -- target_transform: Transforma la variable objetivo a string para el procesamiento y luego la vuelve
        a su tipo original.
    - Return:
        -- corr_cat: matriz con los Cramers V cruzados.
    '''

    if cat_cols is None:
        df_cat_string = list(df.select_dtypes('category').columns.values)
    else:
        df_cat_string = cat_cols
    
    if target_transform:
        t_type = df[target].dtype
        df[target] = df[target].astype('string')
        df_cat_string.append(target)

    corr_cat = []
    vector = []

    for i in df_cat_string:
        vector = []
        for j in df_cat_string:
            confusion_matrix = pd.crosstab(df[i], df[j])
            vector.append(cramers_v(confusion_matrix.values))
        corr_cat.append(vector)

    corr_cat = pd.DataFrame(corr_cat, columns=df_cat_string, index=df_cat_string)
    
    if target_transform:
        df_cat_string.pop()
        df[target] = df[target].astype(t_type)

    return corr_cat

#####
    
def double_plot(df, col_name, is_cont, target, palette=['deepskyblue','crimson'], y_scale='log'):
    """
    ----------------------------------------------------------------------------------------------------------
    Función double_plot:
    ----------------------------------------------------------------------------------------------------------
     Me inspiré en una función de la cátedra para crear mi propia función de gráficos para cada variable
     según su tipo.
     - Funcionamiento:
        La función recibe como un dataframe y la variable a graficar. En base a si es continua o
        si es categórica, se mostrarán dos gráficos de un tipo o de otro
            - Para variables continuas se muestra un histograma y un boxplot en base al Target.
            - Para variables categóricas se muestran dos barplots, uno con la variable sola y la otra en base
            al target. Además, este segundo aplica una transformación logarítmica a la escala del eje y. Esto
            está pensado especialmente para este dataset, debido a que el desbalanceo es tan grande que casi
            no se llegan a percibir los valores 1 en la variable objetivo. Por eso para diferenciar se grafica
            de esta manera.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- col_name: Columna del DF a graficar
        -- is_cont: True o False. Determina si la variable a graficar es continua o no
        -- target: Variable objetivo del DF
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    if is_cont:
        sns.histplot(df[col_name], kde=False, ax=ax1, color='limegreen', bins=50)
    else:
        barplot_df = pd.DataFrame(df[col_name].value_counts()).reset_index()
        sns.barplot(barplot_df, x=col_name, y='count', hue=col_name, palette='YlGnBu', ax=ax1, legend=False)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    plt.xticks(rotation = 90)

    if is_cont:
        sns.boxplot(data=df, x=col_name, y=df[target].astype('string'), hue=df[target].astype('string'), palette=palette, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by '+target)
    else:
        barplot2_df = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
        sns.barplot(data=barplot2_df, x=col_name, y='proportion', hue=barplot2_df[target].astype('string'), palette=palette, ax=ax2)
        plt.yscale(y_scale)
        ax2.set_ylabel('Proportion')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        #Prueba descartada:
        #barplot2_df = df.pivot_table(columns=[target,col_name], aggfunc='count').iloc[0,:].reset_index()
        #sns.barplot(data=barplot2_df, x=col_name, y=np.log(barplot2_df.iloc[:,2]), hue=barplot2_df[target].astype('string'), palette=['deepskyblue','crimson'], ax=ax2)
        #ax2.set_ylabel('Log(Count)')
        
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()
    plt.show()



######
    
def plot_corr(corr, title='Matriz de correlaciones', figsize=(14,8), target_text=False, annot_floor=0.4, annot_all=False, cmap='icefire', vmin=-1, vmax=1, fmt='.1f'):
    """
    ----------------------------------------------------------------------------------------------------------
    Función plot_corr:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe una matriz de correlaciones y genera un gráfico tipo heatmap en base a ella.
        Se pueden determinar el título, tamaño del gráfico, y que cuadrantes tengan o no el valor de la
        correlación.
    - Inputs:
        - corr: Dataframe de la matriz de correlaciones
        - title: Título elegido. Por defecto será 'Matriz de correlaciones'
        - figsize: Tamaño del gráfico
        - target_text: Se debe activar si se desean mostrar todos los valores en los cuadrantes de la última
        fila de la matriz, en mi caso usada para colocar a la variable target.
        - annot_floor: Valor desde el cual se pretenden mostrar los valores de las correlaciones en los
        cuadrantes de la matriz.
        - annot_all: Si es verdadero, se muestran los valores en todos los cuadrantes de la matriz.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=fmt, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)
    for t in ax.texts:
        if ((float(t.get_text())>=annot_floor) | (float(t.get_text())<=-annot_floor)) & (float(t.get_text())<1) :
            t.set_text(t.get_text()) # solo quiero que muestre las anotaciones para correlaciones mayores a 0.4, para identificarlos. El resto no me es muy relevante
        elif (t in ax.texts[-len(corr.iloc[0]):len(list(ax.texts))]) & (target_text) :
            t.set_text(t.get_text()) # además me interesa mostrar todas las correlaciones con mi variable objetivo, en la última fila
        elif annot_all==False:
            t.set_text("")
    plt.title(title, fontdict={'size':'20'})
    plt.show()
    
########

def highlight_max(s, props=''):
    """
    Función básica para dar formato al valor máximo de cada fila de un DF.
        - s: valores a evaluar
        - props: detalle con las propiedades de estilos que se le quiere dar a la celda
    """
    return np.where(abs(s) == np.nanmax(abs(s.values)), props, '')

#######

def y_pred_base_model(y_train, X_test):
    """
    ----------------------------------------------------------------------------------------------------------
    Función y_pred_base_model:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe dos DataFrames (o Serie en caso de y_train), analiza cual es la clase mayoritaria
        del set de datos, y devuelve como output la predicción del modelo base para el set de test.
    - Inputs:
        - y_train: Target del set de entrenamiento. DataFrame o Series.
        - X_test: DataFrame X del set de test (o validación de ser el caso).
    - Return:
        La función devuelve un array de numpy con la predicción del modelo base para los datos otorgados.
    """
    value_max = y_train.value_counts(normalize=True).argmax()
    size = X_test.index.value_counts().size
    y_pred_base = np.random.choice([value_max, abs(1-value_max)], size=size, p=[abs(1-value_max),value_max])
                  # preparé la función para dar valores random en otros casos, pero en este caso los valores serán una probabilidad fija.
    return y_pred_base

########

def metrics_summ(y_true, y_pred, model_name='Model'):
    """
    ----------------------------------------------------------------------------------------------------------
    Función metrics_summ:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe dos arrays/Series, al igual que cualquier función de métricas de SKLearn. Con ellas
        calcula una serie de métricas y las muestra en pantalla. Estas métricas son:
            - Accuracy
            - Balanced Accuracy
            - F2 Score
            - F1 Score
            - Precision
            - Recall
            - Confusion Matrix
    - Inputs:
        - y_true: array/Serie con los valores reales de la variable target. Es decir, el y_test o y_val.
        - y_pred: array/Serie con los valores predecidos por el modelo.
    """

    accuracy = accuracy_score(y_true,y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true,y_pred)
    f2 = fbeta_score(y_true,y_pred, beta=2)
    f1 = f1_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    conf_matrix = confusion_matrix(y_true,y_pred)

    df_output = pd.DataFrame({'Model': model_name #lgbm.__class__.__name__
                              ,'Accuracy': [accuracy]
                              ,'Balanced Accuracy': [balanced_accuracy]
                              ,'F2 Score': [f2]
                              ,'F1 Score': [f1]
                              ,'Precision': [precision]
                              ,'Recall': [recall]
                              }
                            )

    print(f'''
Accuracy: {accuracy:.5f}
\033[1mBalanced Accuracy: {balanced_accuracy:.5f}\033[0m
F2 score: {f2:.5f}
F1 score: {f1:.5f}
Precision: {precision:.5f}
Recall: {recall:.5f}

Confusion Matrix:
{conf_matrix}''')

    return df_output

########

def roc_curve_plot(y_true=None, y_pred=None, title='ROC Curve', model_name='Model', figsize=(7,5), ax=None, curve_color='dodgerblue'):
    """
    ----------------------------------------------------------------------------------------------------------
    Función roc_curve_plot:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función basada en un ejemplo de la cátedra, en la que se grafica la curva ROC y el punto óptimo entre
        el true positive rate y el false negative rate. Además se informa el valor de dicho punto, el threshold
        que le corresponde, el coeficiente de GINI y el área bajo la curva.
    - Imputs:
        - y_true: array/Serie con los valores reales de la variable objetivo (y_test o y_val)
        - y_pred: array/Serie con los valores predecidos por el modelo.
        - title: Título que se le quiera dar al gráfico.
        - model_name: Nombre del modelo implementado para mostrar en el gráfico como leyenda.
        - figsize: tupla con el tamaño deseado para el gráfico.
    """
    if ((y_true is None) or (y_pred is None)):
        print(u'\nFaltan parámetros por pasar a la función')
        return 1

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    
    ax.plot(fpr, tpr, marker='.', color=curve_color, lw=2, label=f'{model_name} (area = %0.3f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='crimson', lw=3, linestyle='--', label='No Skill')
    ax.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best', zorder=2)
    ax.set_xlim([-0.025, 1.025])
    ax.set_ylim([-0.025, 1.025])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontdict={'fontsize':18})
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5)

    gini = (2.0 * roc_auc) - 1.0

    df_output = pd.DataFrame({'Model': model_name
                              ,'Best_threshold': [thresholds[ix]]
                              ,'G-Mean': [gmeans[ix]]
                              ,'Gini': [gini]
                              ,'ROC_AUC': [roc_auc]
                              }
                            )

    return df_output

########

def pr_curve_plot(y_true, y_pred_proba, title='Precision-Recall Curve', f_score_beta=1, model_name='Model', figsize=(7,5), ax=None, curve_color='dodgerblue'):
    """
    ----------------------------------------------------------------------------------------------------------
    Función pr_curve_plot:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función basada en un ejemplo de la cátedra, en la que se grafica la curva Precision-Recall y el punto
        con la combinación óptima de ambos para el F score deseado. Además se informan tanto el threshold
        correspondiente a esa mejor combinación, el F score logrado y el área bajo la curva del modelo.
    - Imputs:
        - y_true: array/Serie con los valores reales de la variable objetivo (y_test o y_val)
        - y_pred_proba: array/Serie con las probabilidades predecidas por el modelo.
        - title: Título que se le quiera dar al gráfico.
        - f_score_beta: Beta para el F score. Normalmente 0.5, 1 o 2.
        - model_name: Nombre del modelo implementado para mostrar en el gráfico como leyenda.
        - figsize: tupla con el tamaño deseado para el gráfico.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    f_score = ((1+(f_score_beta**2)) * precision * recall) / ((f_score_beta**2) * precision + recall)
    ix = np.argmax(f_score)
    auc_rp = auc(recall, precision)
    
    #plt.ylim([0,1])
    no_skill= len(y_true[y_true==1])/len(y_true)
    ax.plot([0,1],[no_skill, no_skill], linestyle='--', label='No Skill', color='crimson', lw=3)
    ax.plot(recall, precision, marker='.', label=model_name, color=curve_color)
    ax.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label=f'Best', zorder=2)
    ax.set_title(str(title), fontdict={'fontsize':18})
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(alpha=0.5)

    df_output = pd.DataFrame({'Model': model_name
                              ,'Best_threshold': [thresholds[ix]]
                              ,f'F{f_score_beta} Score': [f_score[ix]]
                              ,'PR_AUC': [auc_rp]
                              }
                            )

    return df_output

########

def plot_recall_precission(recall_precision, y_true, y_pred_proba, title='Recall & Precision VS Threshold'):
    """
    ----------------------------------------------------------------------------------------------------------
    Función plot_recall_precission:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función basada en un ejemplo de la cátedra, en la que se grafican diferentes métricas del modelo en
        base a los distintos threshold posibles para determinar el valor de la clase objetivo. Las métricas
        que se grafican son:
            - Precision
            - Recall
            - F2 Score
            - F1 Score
        Además, también se muestra en la leyenda el threshold óptimo para maximizar el F2 Score.
    - Imputs:
        - recall_precision: lista de listas en las que cada elemento representa un threshold con sus
        respectivas méticas dentro. Es decir que cada lista dentro de la lista padre contendrá 5 elementos:
        el threhold y las 4 métricas nombradas.
    """
    plt.figure(figsize=(15, 5))
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[1] for element in recall_precision],
                     color="red", label='Recall', scale=1)
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[2] for element in recall_precision],
                     color="blue", label='Precission')
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[3] for element in recall_precision],
                     color="gold", label='F2 Score', lw=2)
    ax = sns.pointplot(x = [round(element[0],2) for element in recall_precision], y=[element[4] for element in recall_precision],
                     color="limegreen", label='F1 Score', lw=1)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f2_score = ((1+(2**2)) * precision * recall) / ((2**2) * precision + recall)
    ix = np.argmax(f2_score)
    
    ax.scatter((round(thresholds[ix],2)*100), f2_score[ix], s=100, marker='o', color='black', label=f'Best F2 (th={thresholds[ix]:.3f}, f2={f2_score[ix]:.3f})', zorder=2)
    ax.set_title(title, fontdict={'fontsize':20})
    ax.set_xlabel('threshold')
    ax.set_ylabel('probability')
    ax.legend()
    
    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if(i%5 == 0) or (i%5 ==1) or (i%5 == 2) or (i%5 == 3):
            labels[i] = '' # skip even labels
            ax.set_xticklabels(labels, rotation=45, fontdict={'size': 10})
    plt.show()

########

def plot_cmatrix(y_true, y_pred, title='Confusion Matrix', figsize=(20,6)):
    """
    ----------------------------------------------------------------------------------------------------------
    Función plot_cmatrix:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función que grafica la matriz de confusión en base a datos reales y predicciones de la variable target
        tanto en valores absolutos como en su forma normalizada.
    - Imputs:
        - y_true: array/Serie con los valores reales de la variable objetivo (y_test o y_val)
        - y_pred_proba: array/Serie con las probabilidades predecidas por el modelo.
        - title: Título que se le quiera dar al gráfico.
        - figsize: tupla con el tamaño deseado para la suma de ambos gráficos.
    """    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    ConfusionMatrixDisplay(confusion_matrix).from_predictions(y_true,y_pred, cmap='Blues', values_format=',.0f', ax=ax1)
    ConfusionMatrixDisplay(confusion_matrix).from_predictions(y_true,y_pred, cmap='Blues', normalize='true', values_format='.2%', ax=ax2)
    ax1.set_title(f'{title}', fontdict={'fontsize':18})
    ax2.set_title(f'{title} - Normalized', fontdict={'fontsize':18})
    ax1.set_xlabel('Predicted Label',fontdict={'fontsize':15})
    ax2.set_xlabel('Predicted Label',fontdict={'fontsize':15})
    ax1.set_ylabel('True Label',fontdict={'fontsize':15})
    ax2.set_ylabel('True Label',fontdict={'fontsize':15})
    rc('font', size=14)
    rc('xtick', labelsize=12)
    rc('ytick', labelsize=12)
    plt.show()

#######

def k_means_search(df, clusters_max, figsize=(6, 6)):
    """
    ----------------------------------------------------------------------------------------------------------
    Función k_means_search:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función que ejecuta el modelo no supervisado k-means sobre el DataFrame introducido tantas veces como
        la cantidad máxima de clusters que se requiera analizar y devuelve un gráfico que muestra la suma de
        los cuadrados de la distancia para cada cantidad de clusters. En función a dicho gráfico se puede
        determinar la cantidad óptima de clusters que necesitamos para nuestro análisis.
    - Imputs:
        - df: DataFrame de Pandas sobre el que se ejecuta el K-Means
        - clusters_max: número máximo de clusters que se quiere analizar.
        - figsize: tupla con el tamaño deseado para la suma de ambos gráficos.
    """
    sse = []
    list_k = list(range(1, clusters_max+1))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(df)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=figsize)
    plt.plot(list_k, sse, '-o')
    plt.xlabel(f'Number of clusters {k}')
    plt.ylabel('Sum of squared distance')
    plt.show()

    
#########

def nulls_detection(df, style=True):
    """
    ---------------------------------------------------------------------------------------
    nulls_detection function
    ---------------------------------------------------------------------------------------
    This function takes a dataframe as input and returns a styled pandas dataframe with the
    count and percentage of missing values in each column.

    Parameters
        - df: pandas dataframe to be analyzed

    Returns
        - nulls: styled pandas dataframe with the count and percentage of missing values
        in each column
    """
    nulls = (pd.DataFrame(df.isnull()
                        .sum()
                        .where(lambda x: x>=1)
                        .dropna()
                        ,columns=['Missing Values'])
            .sort_values(by='Missing Values', ascending=False))
    nulls['Percentage'] = nulls['Missing Values']/df.shape[0]
    nulls['Type'] = df[nulls.index].dtypes

    if style:
        nulls = (nulls
                .style.format({'Missing Values':"{:.0f}",'Percentage': "{:.2%}"})
                .background_gradient(cmap='YlOrRd'))

    return nulls


#########

def df_filter_clean(df):
    """
    Filtrado y ajuste de tipos
        - Filtrado según variable tratamiento ('Tech Support')
        - Ajuste de tipo de variable 'Total Charges'
    """
    df = df[df['Tech Support'].isin(['Yes', 'No'])]
    df.loc[:,'Total Charges'] = df['Total Charges'].replace(' ', np.nan).astype(float)
    return df


#########

def Preprocessing(df, cat_transf='mix', scale=True, cat_vars=None, num_vars=None, bool_vars=None,
                  vars_ord=[], vars_oh=[], not_scale=[], nulls_strategy='median', show_info=False):
    """
    ----------------------------------------------------------------------------------------------------------
    Preprocessing function:
    ----------------------------------------------------------------------------------------------------------
    - Description: Function that receives a dataframe and performs the preprocessing based on the parameters
                that the user chooses.
    - Inputs:
        - \033[1mdf:\033[0m dataframe to be preprocessed.
        - \033[1mcat_transf:\033[0m type of transformation for categorical variables. It can be 'ordinal',
        'onehot', 'mean' or 'mix'.
        - \033[1mscale:\033[0m boolean that indicates if the numeric variables should be scaled.
        - \033[1mcat_vars:\033[0m list of categorical variables to be transformed.
        - \033[1mnum_vars:\033[0m list of numeric variables to be transformed.
        - \033[1mbool_vars:\033[0m list of boolean variables to be transformed.
        - \033[1mvars_ord:\033[0m list of categorical variables to be transformed using ordinal encoding.
        - \033[1mvars_oh:\033[0m list of categorical variables to be transformed using onehot encoding.
        - \033[1mnot_scale:\033[0m list of numeric variables that should not be scaled.
        - \033[1mnulls_strategy:\033[0m strategy to fill null values in numeric variables. It can be 'mean',
        'median' or 'most_frequent'.
    
    - Output: preprocessor object that can be used in a pipeline.
    """

    # Defining transformers

    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=nulls_strategy))
        ,('scaler', StandardScaler() if scale else 'passthrough')])
    
    # Categoircal transformers
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ,('ordinal', OrdinalEncoder())])
    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ,('onehot', OneHotEncoder(sparse_output=False))])
    mean_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ,('mean_encoder', TargetEncoder())])
    

    # Defining types of features
    df_bool, _, _ = tipos_vars(df,False)

    if num_vars == None:
        num_vars = (df
                    .select_dtypes(include=['int64', 'float64'])
                    .drop(not_scale, axis=1)
                    .drop(df_bool, axis=1)
                    .columns)
    else:
        num_vars = np.array(num_vars)[~np.isin(num_vars, not_scale)]
    
    if bool_vars == None:
        bool_vars = df[df_bool].columns
    
    if cat_vars == None:
        cat_vars = df.select_dtypes(include=['object','string','category']).columns.values
    else:
        cat_vars = np.array(cat_vars)

    # Defining type of transformation for categorical variables
    if cat_transf=='ordinal':
        vars_ord = cat_vars.copy()
        cat_vars = []
    elif cat_transf=='onehot':
        vars_oh = cat_vars.copy()
        cat_vars = []
    elif cat_transf=='mix' and (vars_ord!=[] or vars_oh!=[]):
        cat_vars = cat_vars[~np.isin(cat_vars, [vars_ord + vars_oh])]
    elif cat_transf not in ['ordinal','onehot','mix','mean']:
        print('Error: cat_transf debe ser uno de los siguientes valores: "ordinal", "onehot", "mean" o "mix"')
        return None

    # Defining preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_vars),
            ('ordinal', ordinal_transformer, vars_ord),
            ('onehot', onehot_transformer, vars_oh),
            ('mean', mean_transformer, cat_vars),
            ('bool', ordinal_transformer, bool_vars)
            ]
        ,remainder='passthrough'
        ,verbose_feature_names_out=False).set_output(transform="pandas")
    
    if show_info:
        print(f'''Prprocessing pipeline defined with the following parameters:
        Mean encoding: {cat_vars}
        Numerical transformation (scaling={scale} input strategy={nulls_strategy}): {num_vars}
        Booleans encoding (ordinal 0-1): {bool_vars}
        Ordinal encoding: {vars_ord}
        One-Hot encoding: {vars_oh}''')

    return preprocessor


#########

def feature_selection(df, add=[]):
    """
    ----------------------------------------------------------------------------------------------------------
    Función feature_selection:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Recibe un DataFrame y un opcional de columnas extras a eliminar. Devuelve un DataFrame
    con las columnas del feature selection eliminadas, además de la columna extra en caso de haberse introducido.
    - Inputs:
        - df: DataFrame de Pandas al que se le reducirá el número de variables
        - add: argumento opcional en el que se pueden incluir más variables que se quieran eliminar
    - Return: DataFrame de pandas con las columnas reducidas según el feature selection aplicado.
    """
    drop = ['CustomerID'
            ,'Count'
            ,'Country'
            ,'State'
            ,'Zip Code'
            ,'Lat Long'
            ,'Latitude'
            ,'Longitude'
            ,'Churn Score'
            ,'Churn Label'
            ,'Churn Reason'
            ,'City'
            ,'Total Charges'
            ,'Multiple Lines'
            ,'Streaming Movies'
            ,'Streaming TV'
            ,'Gender']
    if add != []:
        drop+=add
    df_new = df.drop(drop, axis=1)
    return df_new


#########

def make_graph(adjacency_matrix, labels=None, threshold=0.01):
    """
    ----------------------------------------------------------------------------------------------------------
    Función make_graph:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe una matriz de adyacencia y devuelve un gráfico de grafo con las
    conexiones entre las variables. Se puede introducir una lista de nombres para las variables, en caso de
    no introducirse se les asignará un nombre genérico.
    - Inputs:
        - adjacency_matrix: Matriz de adyacencia de las variables a analizar
        - labels: lista de nombres de las variables
    - Return: Gráfico de grafo con las conexiones entre las variables
    """
    idx = np.abs(adjacency_matrix) > threshold
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d


#########

def str_to_dot(string):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función str_to_dot:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Convierte un string de la librería graphviz a un formato de grafo DOT válido.
    - Inputs:
        - string: string de la librería graphviz
    - Return: string con formato de grafo DOT válido
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph


#########

def plot_save(sm, path):
    """
    ----------------------------------------------------------------------------------------------------------
    Función plot_save:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe un modelo de estructura y un path donde se quiere guardar el gráfico
    generado por el modelo. Guarda el gráfico en el path especificado.
    - Inputs:
        - sm: modelo de estructura
        - path: path donde se quiere guardar el gráfico
    """
    viz = plot_structure(
        sm,
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK,
    )
    viz.save_graph(path)


#########

@curry
def effect(data, y, t):
        return (np.sum((data[t] - data[t].mean())*data[y]) /
                np.sum((data[t] - data[t].mean())**2))

#########


def cumulative_gain_curve(df, prediction, y, t, ascending=False, normalize=False, steps=100):
    """
    ----------------------------------------------------------------------------------------------------------
    Función cumulative_gain_curve:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe un DataFrame, una predicción, la variable target, la variable
    treatment, el orden y un booleano para determinar si se quiere normalizar o no. Devuelve un array con la
    sensibilidad acumulada en función de las predicciones.

    - Inputs:
        - df: DataFrame de Pandas
        - prediction: predicción (CATE en modelos causales)
        - y: variable target/outcome
        - t: variable treatment
        - ascending: booleano que indica si se quiere ordenar de forma ascendente o descendente
        - normalize: booleano que indica si se quiere normalizar o no
        - steps: cantidad de pasos a realizar

    - Return: array con el efecto acumulado en función de la variable objetivo.
    """
    
    effect_fn = effect(t=t, y=y)
    normalizer = effect_fn(df) if normalize else 0
    
    size = len(df)
    ordered_df = (df
                  .sort_values(prediction, ascending=ascending)
                  .reset_index(drop=True))
    
    steps = np.linspace(size/steps, size, steps).round(0)
    effects = [(effect_fn(ordered_df.query(f"index<={row}"))
                -normalizer)*(row/size) 
               for row in steps]

    return np.array([0] + effects)


#########

def cumulative_effect_curve(dataset, prediction, y, t, ascending=False, steps=100):
    """
    ----------------------------------------------------------------------------------------------------------
    Función cumulative_effect_curve:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe un DataFrame, una predicción, la variable target, la variable
    treatment, el orden y la cantidad de pasos. Devuelve un array con el efecto acumulado en función
    de las predicciones.

    - Inputs:
        - dataset: DataFrame de Pandas
        - prediction: predicción (CATE en modelos causales)
        - y: variable target/outcome
        - t: variable treatment
        - ascending: booleano que indica si se quiere ordenar de forma ascendente o descendente
        - steps: cantidad de pasos a realizar

    - Return: array con el efecto acumulada en función de la variable objetivo.
    """

    size = len(dataset)
    ordered_df = (dataset
                  .sort_values(prediction, ascending=ascending)
                  .reset_index(drop=True))
    
    steps = np.linspace(size/steps, size, steps).round(0)
    
    return np.array([effect(ordered_df.query(f"index<={row}"), t=t, y=y)
                     for row in steps])


#########

def effect_by_quantile(df, pred, y, t, q=10):
    """
    ----------------------------------------------------------------------------------------------------------
    Función effect_by_quantile:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe un DataFrame, una predicción, la variable target, la variable
    treatment y la cantidad de cuantiles en los que se quiere dividir la variable predicción. Devuelve un
    DataFrame con el efecto estimado en cada cuantil.

    - Inputs:
        - df: DataFrame de Pandas
        - pred: predicción (CATE en modelos causales)
        - y: variable target/outcome
        - t: variable treatment
        - q: cantidad de cuantiles en los que se quiere dividir la variable predicción
    
    - Return: DataFrame con el efecto estimado en cada cuantil.
    """
    
    # makes quantile partitions
    groups = np.round(pd.IntervalIndex(pd.qcut(df[pred], q=q)).mid, 2) 
    
    return (df
            .assign(**{f"{pred}_quantile": groups})
            .groupby(f"{pred}_quantile")
            # estimate the effect on each quantile
            .apply(effect(y=y, t=t))) 


