import pandas as pd
import sqlite3 as db
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder, FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# python
import pickle
from functools import partial
# basics
import scipy.stats as stats
# graphing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import plot_confusion_matrix
# feature selection
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
# model selection
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, make_scorer,
                             confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay)
# models
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from collections import Counter
from datetime import datetime

import unidecode
import string
import re
from names_dataset import NameDataset


def readdata():
    ISM_SourceData = pd.read_excel('Data/Tickets 2021-2022 (V02).xlsx', sheet_name="Squirrel SQL Export")
    return ISM_SourceData

def selectdata(ISM_SourceData, from_date, to_date):
    ISM_SourceData_test = ISM_SourceData[(ISM_SourceData['ACTUALSTART']>=from_date) & (ISM_SourceData['ACTUALSTART']<=from_date)]
    return ISM_SourceData_test

def cleandatastructure(ISM_SourceData_test, country_list):
    ## Put all text fields together
    ISM_SourceData_test['FULLDESC'] = ISM_SourceData_test['TITLE']+ ' ' +ISM_SourceData_test['LONGDESCRIPTION']
    ISM_SourceData_test['FULLDESC']
    ISM_test = ISM_SourceData_test[['TICKETID','FULLDESC','AffectedPersonCountry','OWNERGROUP']]

    ## Do some Cleaning
    ISM_test['FULLDESC'].fillna('NA',inplace = True)
    ISM_test['AffectedPersonCountry'].fillna('NA',inplace = True)
    ISM_test['FULLDESC'] = ISM_test['FULLDESC'].str.upper()

    #Delete Null values
    ISM_test = ISM_test[ISM_test['AffectedPersonCountry']!="NA"]
    ISM_test = ISM_test[ISM_test['FULLDESC']!="NA"]

    ## Consolidate Countriees
    ISM_test['AffectedPersonCountry'] = ISM_test['AffectedPersonCountry'].map({'DOM': 'Dominican Republic','Bahamas': 'Bahamas', 'Bangladesh': 'Banglasdesh',\
                                        'Belize': 'Belize','Central': 'Central', \
                                        'China': 'China','Clombia': 'Colombia','Colombia': 'Colombia','Costa Rica': 'Costa Rica','Croatia': 'Croatia','CZ': 'Czech Republic',\
                                        'Czech Republic': 'Czech Republic','DE': 'Germany','Dominican Republic': 'Dominican Republic', \
                                        'EG': 'Egypt','EGY': 'Egypt','Egypt': 'Egypt','El Salvador': 'El Salvador','ES': 'Spain','España': 'Spain',\
                                        'Filipinas': 'Philippines','FR': 'France','France': 'France', \
                                        'GB': 'United Kingdom','German': 'Germany','Germany': 'Germany','Great Britain': 'United Kingdom','Guatemala': 'Guatemala',\
                                        'Filipinas': 'Philippines',\
                                        'Haiti': 'Haiti','HU': 'Hungary','IL': 'Israel','Israel': 'Israel','Jamaica': 'Jamaica','JM': 'Jamaica','Malaysia': 'Malaysia',\
                                        'Latvia': 'Latvia','Mexic': 'Mexico',\
                                        'México': 'Mexico','Mexio': 'Mexico','Mexixo': 'Mexico','Mexico': 'Mexico','Filipinas': 'Philippines','MX': 'Mexico',\
                                        'Filipinas': 'Philippines','Netherlands': 'Netherlands',\
                                        'NIC': 'Nicaragua','Nicaragua': 'Nicaragua','Panam': 'Panama','Panama': 'Panama','Peru': 'Peru','Philippines': 'Philippines', \
                                        'PHL': 'Philippines','PH': 'Philippines','Philippines': 'Philippines','Poland': 'Poland','PL': 'Poland','Puerto Rico': 'Puerto Rico',\
                                        'Republica Dominicana': 'Dominican Republic','RO': 'Romania',\
                                        'Romania': 'Romania','Filipinas': 'Philippines','Republica Checca': 'Czech Republic','Spain': 'Spain','Switzerland': 'Switzerland',\
                                        'Trinidad y Tobago': 'Trinidad y Tobago', \
                                        'UAE': 'United Arab Emirates','UK': 'United Kingdom','United Kingdom': 'United Kingdom','United Arab Emirates': 'United Arab Emirates',\
                                        'Switzerland': 'Switzerland',\
                                        'United States of America': 'United States of America','United States of Americaśśś': 'United States of America',\
                                        'Unites States of America': 'United States of America',\
                                        'US': 'United States of America','USA': 'United States of America'})
    ISM_test['AffectedPersonCountry'].unique()

    ## Delete all not mapped values 
    ISM_test = ISM_test[ISM_test['AffectedPersonCountry'].isnull() == False]
    ISM_test = ISM_test[ISM_test['AffectedPersonCountry'].isin(country_list)]
    #ISM_test = ISM_test[ISM_test['AffectedPersonCountry'] in list)
    return ISM_test

def PrepareWordLists(datasets, df_words):
    # Recibe la lista de los datasets seccionados, para evitar tiempos muy largos de procesamiento 
    nd = NameDataset()
    no_name = nd.search('##CUCARAMACARATITIRIFUE##').get('first_name')
    for dataset in datasets:
        errcount = 0
        count = 0
        count_check = 1
        #print(dataset.info())

        for ind in dataset.index:
            #Seperar las palabras de cada regitro
            #Crear un registro para cada palabra
            count = count + 1
            if count == count_check:
                print(datetime.now())
                print(count)
                count_check = count_check + 500

            word_list = getlistofwords(dataset['FULLDESC'][ind])
            word_list = list(dict.fromkeys(word_list))
            
            for word in word_list:
              #Evalua cada palabra y si no es de las exlcluidas por default la agrega a la lista de los palabars del grupo

               if len(word) > 2:
                    if word.upper() not in exclusionlist() \
                    and word.find('CID') == -1 and word.find('@CEMEX.COM') == -1 and word.find('@EXT.CEMEX.COM') == -1 and word.find('IMAGE') == -1 and word.find('>>;') == -1 \
                    and word.find('GSC') == -1 and word.find('______') == -1 and word.find('---') == -1 and word.find('  ') == -1 and word.find('GIF') == -1 \
                    and word.find('SAFELINKS') == -1 and word.find('ATTACHE') == -1 and word.find('ADDRESS') == -1 \
                    and word.find('PNG') == -1 and word.find('E0') == -1 and word.find('E-') == -1 and word.find('****') == -1 and word.find('#') == -1 and word.find('LISTENER') \
                    and word.find('YOU') == -1 and word.find('@') == -1 and not word.startswith('ATT00') and not word.startswith('/')\
                    and not word.startswith('+') and not word[0].isdigit() and not word[1].isdigit() and not word.startswith('<') \
                    and not word.startswith('/>') and not word.startswith('@'):
                    
                        try:
                            is_name = nd.search(word).get('first_name').get('country').get('Mexico')
                            if is_name == no_name:
                                # Guarda la palabra para el grupo y país
                                #df_words = df_words.append([dataset['AffectedPersonCountry']
                                #[index],dataset['OWNERGROUP'][index], word.upper()],['COUNTRY','OWNERGROUP','WORDS'], [str(index)])    
                                df_words = df_words.append({'COUNTRY':dataset['AffectedPersonCountry'][ind],'OWNERGROUP':dataset['OWNERGROUP'][ind],'WORDS':word.upper()}, ignore_index=True)
                        except:
                            errcount = errcount + 1
                            df_words = df_words.append({'COUNTRY':dataset['AffectedPersonCountry'][ind],'OWNERGROUP':dataset['OWNERGROUP'][ind],'WORDS':word.upper()}, ignore_index=True)
                            
    print(datetime.now())
    print('Words: ',len(df_words['WORDS'].unique())," Groups: ",len(df_words['OWNERGROUP'].unique()),' Countries:',len(df_words['COUNTRY'].unique()))
    return df_words


def buildwordsindex(ISM_wgwords_all, delete_limit):
    #Eliminar palabras que nod dan valor por pocas repeticiones.
    minimum_repeat = delete_limit
    CountedWords = pd.DataFrame(ISM_wgwords_all.groupby(['WORDS'])[['WORDS']].count())
    CountedWords = CountedWords.drop(CountedWords[CountedWords['WORDS'] < minimum_repeat].index)
    CountedWords['WORDS'] = CountedWords.index
    CountedWords.rename(columns ={'WORDS':'VALUES'}, inplace = True)

    ## Hacemos Inner Merge con ISM_wgwords para dejar solo las palabras útiles
    ISM_wgwords_all = pd.merge(ISM_wgwords_all, CountedWords, on="WORDS")

    ## Creamos columna Unique para identificar combinaciones únicas
    ISM_wgwords_all['UNIQUE'] = ISM_wgwords_all['COUNTRY'] + '<<|>>' + ISM_wgwords_all['OWNERGROUP']+ '<<|>>' + ISM_wgwords_all['WORDS']
    arr_wgwords_uniques = ISM_wgwords_all['UNIQUE'].unique()

    #Convertimos en dataframe y separamos las columnas
    df_uniquewords = pd.DataFrame(arr_wgwords_uniques, columns=['UNIQUE'])
    df_uniquewords[['COUNTRY','X1', 'OWNERGROUP', 'X2', 'WORDS']] = df_uniquewords['UNIQUE'].str.split('<<|>>', 5, expand=True)
    df_uniquewords = df_uniquewords[['COUNTRY', 'OWNERGROUP', 'WORDS']]

    #Guardamos los resultados en in dataframe = ISM_wgwords_uniques
    ISM_wgwords_uniques = df_uniquewords
    
    #Movemos los valores al índice para acelerar las búsquedas en la preparadación del trainin data sert
    ISM_wgwords_uniques['STR_INDEX']=ISM_wgwords_uniques['COUNTRY']+"-"+ISM_wgwords_uniques['OWNERGROUP']
    ISM_wgwords_uniques.index = ISM_wgwords_uniques['STR_INDEX']
    return ISM_wgwords_uniques


def trainingstructuredata(ISM_wgwords_uniques):
    GroupColumnsNames =ISM_wgwords_uniques['OWNERGROUP'].unique()
    TrainingData = pd.DataFrame(columns=GroupColumnsNames)
    TrainingData['COUNTRY'] = 'NA'
    TrainingData['OWNERGROUP'] = 'NA'
    TrainingData['TICKETID'] = 'NA'

    TrainingData.COUNTRY.datatype = object
    TrainingData.OWNERGROUP.datatype = object
    TrainingData.TICKETID.datatype = object
    return TrainingData



def training_dataset(ISM_test, TrainingData, ISM_wgwords_uniques, square_values):
## ISM_Test datos de ISM Country / Ownergroups / Full Text
## TrainingData Estructura de Salida
## ISM_wgwords_uniques: Calisficador de Palabras: Country / Ownergroup / Words
## square_values: Indica si se elevan al cuadrado los valores de matches de palabras (Bolean)
    GroupColumnsNames =ISM_wgwords_uniques['OWNERGROUP'].unique()
    errcount = 0
    count = 0
    count2 = 0
    count_check = 1

    for numindex, ind in enumerate(ISM_test.index):
        count = count + 1
        if count == count_check:
            print(datetime.now())
            print(count)

            if count == count_check:
                count_check = count_check + count_check

        word_list = getlistofwords(ISM_test['FULLDESC'][ind])        
        
        TrainingData = TrainingData.append({'TICKETID':ISM_test['TICKETID'][ind], 'COUNTRY':ISM_test['AffectedPersonCountry'][ind],'OWNERGROUP':ISM_test['OWNERGROUP'][ind]}, ignore_index=True) 
        #TrainingData = TrainingData.append([ISM_test['AffectedPersonCountry'][ind],ISM_test['OWNERGROUP'][ind]], ['COUNTRY','OWNERGROUP'],[numindex])
        
        ## Seteo valores de columnas de grupos del registros nuevo para que no sean null y puedan sumarse
        for column in TrainingData.columns:
            if column not in [0,'COUNTRY','OWNERGROUP','TICKETID']:
                TrainingData[column][numindex] = 0

        for columns in GroupColumnsNames:
            #grouplistwords = ISM_wgwords_uniques[(ISM_wgwords_uniques['COUNTRY']==ISM_test['AffectedPersonCountry'][ind]) &(ISM_wgwords_uniques['OWNERGROUP']==columns) ][['WORDS']]
            searchindex = ISM_test['AffectedPersonCountry'][ind]+"-"+columns
            
            try:
                grouplistwords = ISM_wgwords_uniques.loc[searchindex]
                grouplistwords = grouplistwords['WORDS'].to_list()

                list_matches = set(word_list) & set(grouplistwords)
                if len(list_matches) > 0:
                        #TrainingData[listwords['OWNERGROUP'][listidx]][numindex] = TrainingData[listwords['OWNERGROUP'][listidx]][numindex] + len(list_matches)
                    if square_values:
                        TrainingData[columns][numindex] = TrainingData[columns][numindex] + (len(list_matches))**2
                    else:
                        TrainingData[columns][numindex] = TrainingData[columns][numindex] + len(list_matches)
            except:
                errcount = errcount + 1
            else:
                errcount = errcount + 1
            
            
    print('Finished: ', datetime.now(), ' Errors:',errcount)
    return TrainingData

def exclusionlist():
        
    exclusionlist = ['OCTOBER','OCTUBRE','NOVEMBER','NOVIEMBRE','MAY','TUESDAY','MONDAY','WEDNESDAY','LUNES','VIERNES','FRIDAY','JUEVES','0','MARTES',\
        'THURSDAY','DOMINGO','MIERCOLES','JULIO','NOV','OCT','MON','SEPTIEMBRE','SABADO','SEPTEMBER','JAN','DOM','SATURDAY','AGO','MAR','SAB','ABRIL',\
        'FEBRERO','JUNE','SUNDAY','MAYO','AUGUST','AGOSTO','DICIEMBRE','JULY','DECEMBER','THU','APRIL','JUE','FRI','MARZO','SEP','AUG','LUN','JANUARY','JUNIO',\
        'ENERO','TUE','MIE','SAT','JUL','JUN','APR','FEBRUARY','DIC','WED','SUN','MARCH','VIE','FEB',\
        'SCREENSHOT','THE','AND','ATTACHMENT','PLEASE','HELLO','FOR','CEMEX','SUBJECT','FAVOR','RE','CAUTION',\
        'REGARDS','SALUDOS','listener','FROM','SUBJECT','CC','TO','FW','0','BUT',\
        'DIA','APOYO','SENT','WITH''THAT','COMO','CAN','HELP','BUT','LES','SUS', 'ADJUNT','SENDER','RECEIVED',   
        'SENT:','GSC', 'THIS', 'DEL','POR','PARA','QUE','CON','WAS','ARE','NOT','THANKS','THANK','LAS','ESTE','E-MAIL','SAVED','LOS','LAS','ESTE',
        'BUEN','BUENOS','QUEDO','BEST','ASUNTO','DEAR','GRACIAS','GRACIAS','UNA','SALUDOS','RESOLVED','ATENTA',\
        'ADJUNTO','ADJUNTA','ADJUNTOS','ADJUNTAMOS','ADJUNTANDO','*ADJUNTO','ATTACHED','ATTACHMENTS','ATTACH','ATTACHING','ATTACHED','ATTACHEE',\
        'ADDRESS','DIRECCION',\
        'A','ANTE','ANTES','BAJO','CON','CONTRA','DE','DESDE','PARA','POR','SEGUN','SIN','SOBRE','TRAS','PERO','SAN','NOS','NAME','PUEDE'\
        'UN','UNOS','UNA','UNAS','UNO',\
        'HOLA','BUEN','BUENAS','DIA','BUENOS','DIAS','GRACIAS','MUCHAS','TARDES','SIGUIENTES'\
         'ESTA','ESTO','ESTAS','ESTOS','ESA','ESOS','ESAS','ESOS',\
         'AYUDA','TIENE',\
         'HTTP','HTTPS','WWW','COM','MAILTO','COM/','JPG','CEL','NAME','TEL','TELEFONO']

    return exclusionlist

def getlistofwords(str_desc):

    cleanstring = str_desc.replace("\\N"," ")
    cleanstring = cleanstring.replace(r"\N"," ")
    cleanstring = re.sub('[-_"&.,\[\]?!¡#<>;:\'=$()*\\\]', ' ',cleanstring)
    #word_list = [re.sub('[?&,:;"><\'.()!¡\[\]-_]', ' ', unidecode.unidecode(wd.strip())) for wd in word_list]
    word_list = cleanstring.split(" ")
    word_list = [unidecode.unidecode(wd.strip()) for wd in word_list]
    
    return word_list
