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

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop(target,axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

#####

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
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
        if len(df[i].unique()) <= 2 and df[i].dtype=='int64':
            list_bools.append(i)
            if show:
                print(f"{i} {colored('(boolean)','blue')} :  {df[i].unique()}")
        elif len(df[i].unique()) < 50:
            list_cat.append(i)
            if show:
                print(f"{i} {colored('(categoric)','red')} (\033[1mType\033[0m: {df[i].dtype}): {df[i].unique()}")
        else:
            list_num.append(i)
            if show:
                print(f"{i} {colored('(numeric)','green')} : \033[1mRange\033[0m = [{df[i].min():.2f} to {df[i].max():.2f}], \033[1mMean\033[0m = {df[i].mean():.2f}")
    
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

def corr_cat(df,target=None,target_transform=False):
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
    df_cat_string = list(df.select_dtypes('category').columns.values)
    
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
        sns.histplot(df[col_name], kde=False, ax=ax1, color='limegreen')
    else:
        barplot_df = pd.DataFrame(df[col_name].value_counts()).reset_index()
        sns.barplot(barplot_df, x=col_name, y='count', palette='YlGnBu', ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    plt.xticks(rotation = 90)

    if is_cont:
        sns.boxplot(data=df, x=col_name, y=df[target].astype('string'), palette=palette, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by '+target)
    else:
        barplot2_df = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
        sns.barplot(data=barplot2_df, x=col_name, y='proportion', hue=barplot2_df[target].astype('string'), palette=palette, ax=ax2)
        plt.yscale(y_scale)
        ax2.set_ylabel('Proportion')
        
        #Prueba descartada:
        #barplot2_df = df.pivot_table(columns=[target,col_name], aggfunc='count').iloc[0,:].reset_index()
        #sns.barplot(data=barplot2_df, x=col_name, y=np.log(barplot2_df.iloc[:,2]), hue=barplot2_df[target].astype('string'), palette=['deepskyblue','crimson'], ax=ax2)
        #ax2.set_ylabel('Log(Count)')
        
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()


######
    
def plot_corr(corr, title='Matriz de correlaciones', figsize=(14,8), target_text=False, annot_floor=0.4, annot_all=False):
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
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='icefire', ax=ax, vmin=-1, vmax=1)
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

def metrics_summ(y_true, y_pred):
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
    print(f'''
Accuracy: {accuracy_score(y_true,y_pred):.5f}
Balanced Accuracy: {balanced_accuracy_score(y_true,y_pred):.5f}
\033[1mF2 score: {fbeta_score(y_true,y_pred, beta=2):.5f}\033[0m
\033[1mF1 score: {f1_score(y_true,y_pred):.5f}\033[0m
Precision: {precision_score(y_true,y_pred):.5f}
Recall: {recall_score(y_true,y_pred):.5f}

Confusion Matrix:
{confusion_matrix(y_true,y_pred)}''')

########

def roc_curve_plot(y_true=None, y_pred=None, title='ROC Curve', model_name='Model', figsize=(7,5)):
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
    print('Best Threshold = %f, G-Mean = %.3f' % (thresholds[ix], gmeans[ix]))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, marker='.', color='dodgerblue', lw=2, label=f'{model_name} (area = %0.3f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='crimson', lw=3, linestyle='--', label='No Skill')
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best', zorder=2)
    ax.set_xlim([-0.025, 1.025])
    ax.set_ylim([-0.025, 1.025])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontdict={'fontsize':18})
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5)

    gini = (2.0 * roc_auc) - 1.0

    print('\n*************************************************************')
    print(u'\nEl coeficiente de GINI es: %0.2f' % gini)
    print(u'\nEl área por debajo de la curva ROC es: %0.4f' %roc_auc)
    print('\n*************************************************************')

########

def pr_curve_plot(y_true, y_pred_proba, title='Precision-Recall Curve', f_score_beta=1, model_name='Model', figsize=(7,5)):
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
    print(f'Best Threshold = {thresholds[ix]:.5f}, F{f_score_beta} Score = {f_score[ix]:.3f}, AUC = {auc_rp:.4f}')
    
    #plt.ylim([0,1])
    fig, ax = plt.subplots(figsize=figsize)
    no_skill= len(y_true[y_true==1])/len(y_true)
    ax.plot([0,1],[no_skill, no_skill], linestyle='--', label='No Skill', color='crimson', lw=3)
    ax.plot(recall, precision, marker='.', label=model_name, color='dodgerblue')
    ax.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label=f'Best', zorder=2)
    ax.set_title(str(title), fontdict={'fontsize':18})
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(alpha=0.5)

########

def plot_recall_precission(recall_precision, y_true, y_pred_proba):
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
    ax.set_title('Recall & Precision VS Threshold', fontdict={'fontsize':20})
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

def ex_preprocessing(df, cat_transf='mix', scale=True):
    """
    ----------------------------------------------------------------------------------------------------------
    Función preprocessing:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe un dataframe y realiza el preprocesamiento en base a los parámetros
    que el usuario elija.
    - Inputs:
        - df: DataFrame de pandas sobre el que se generará el objeto preprocessor
        - cat_transf: metogolodía de transformación de variables. Puede tomar los siguientes valores:
            * 'mix': mix de encodings personalizado para caso fraude, con onehot, ordinal y mean encoding.
            * 'orginal': se aplica Ordinal Encoding
            * 'onehot': se aplica One Hot Encoding
            * 'mean': se aplica Mean Encoder (TargetScorer en SKLearn)
    - Output: devuelve un objeto preprocessor listo para ser instanciado.
    """
    
    if scale:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])
    else:
        numeric_transformer = Pipeline(steps=[
            ('pass', 'passthrough')])
    
    if (cat_transf=='ordinal') | (cat_transf=='mix'):
        categorical_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder())])
    elif cat_transf=='onehot':
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse=False, sparse_output=False))])
    elif cat_transf=='mean':
        categorical_transformer = Pipeline(steps=[
            ('mean_encoder', TargetEncoder())])
    
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse=False, sparse_output=False))])
    mean_transformer = Pipeline(steps=[
        ('mean_encoder', TargetEncoder())])

    df_bool, df_cat, df_num = tipos_vars(df,False)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(df_bool, axis=1).columns
    boolean_features = df[df_bool].columns
    
    if cat_transf=='mix':
        categorical_features = df.select_dtypes(include=['object']).drop(['device_os'], axis=1).columns
    else:
        categorical_features = df.select_dtypes(include=['object']).columns
    
    device_feature = np.array(['device_os'])
    payment_feature = np.array(['payment_type'])
    
    if cat_transf=='mix':
        preprocessor = ColumnTransformer(
            transformers=[
                ('num',numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('device', onehot_transformer, device_feature),
                ('bool','passthrough', boolean_features)]
            ,remainder='passthrough'
            ,verbose_feature_names_out=False).set_output(transform="pandas")
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num',numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('bool','passthrough', boolean_features)]
            ,remainder='passthrough'
            ,verbose_feature_names_out=False).set_output(transform="pandas")
    
    return preprocessor


######################

def preprocessing(df, cat_transf='mix', scale=True):
    """
    ----------------------------------------------------------------------------------------------------------
    Función preprocessing:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe un dataframe y realiza el preprocesamiento en base a los parámetros
    que el usuario elija.
    - Inputs:
        - df: DataFrame de pandas sobre el que se generará el objeto preprocessor
        - cat_transf: metogolodía de transformación de variables. Puede tomar los siguientes valores:
            * 'mix': mix de encodings personalizado para caso fraude, con onehot, ordinal y mean encoding.
            * 'orginal': se aplica Ordinal Encoding
            * 'onehot': se aplica One Hot Encoding
            * 'mean': se aplica Mean Encoder (TargetScorer en SKLearn)
    - Output: devuelve un objeto preprocessor listo para ser instanciado.
    """
    if scale:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])
    else:
        numeric_transformer = Pipeline(steps=[
            ('pass', 'passthrough')])
    
    if (cat_transf=='ordinal') | (cat_transf=='mix'):
        categorical_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder())])
    elif cat_transf=='onehot':
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse=False, sparse_output=False))])
    elif cat_transf=='mean':
        categorical_transformer = Pipeline(steps=[
            ('mean_encoder', TargetEncoder())])
    
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse=False, sparse_output=False))])
    mean_transformer = Pipeline(steps=[
        ('mean_encoder', TargetEncoder())])

    df_bool, df_cat, df_num = tipos_vars(df,False)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(df_bool, axis=1).columns
    boolean_features = df[df_bool].columns
    
    if cat_transf=='mix':
        categorical_features = df.select_dtypes(include=['object']).drop(['device_os','payment_type'], axis=1).columns
    else:
        categorical_features = df.select_dtypes(include=['object']).columns
    
    device_feature = np.array(['device_os'])
    payment_feature = np.array(['payment_type'])
    
    if cat_transf=='mix':
        preprocessor = ColumnTransformer(
            transformers=[
                ('num',numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('device', onehot_transformer, device_feature),
                ('payment', mean_transformer, payment_feature),
                ('bool','passthrough', boolean_features)]
            ,remainder='passthrough'
            ,verbose_feature_names_out=False).set_output(transform="pandas")
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num',numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('bool','passthrough', boolean_features)]
            ,remainder='passthrough'
            ,verbose_feature_names_out=False).set_output(transform="pandas")
    
    return preprocessor

########

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
    drop = ['source','velocity_24h','velocity_6h','phone_mobile_valid', 'foreign_request']
    if add != []:
        drop+=add
    df_new = df.drop(drop, axis=1) #'device_fraud_count',
    return df_new




