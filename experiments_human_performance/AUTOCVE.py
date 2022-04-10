from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

from dataprep_scripts import competition_dataprep

import sklearn

if sklearn.__version__>='0.22':
    from sklearn.impute import SimpleImputer as Imputer
else:
    from sklearn.preprocessing import Imputer

from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

from shutil import move, copy
import signal
import sys
import os
import time

TIME_PER_TASK=5400
GRACE_PERIOD=100


FEATURE_ENGINEERING_MOD = False

SEED_LIST=[5249444, 7592240, 4510071, 2688207, 5344429, 6581462, 6381116, 4901935, 1977886, 2591851, 9404208, 4098546]


def parse_competition_datasets(competition_name):

    load_datset, _ = competition_dataprep[competition_name]


    df_train, df_test = load_datset(f'../competition_datasets/{competition_name}', feature_engineering = FEATURE_ENGINEERING_MOD)

    df_train = df_train[~df_train['target'].isnull()]

    x_cols = [c for c in df_train.columns if c != 'target']
    X_train = df_train[x_cols]
    y_train = df_train['target']
    X_test = df_test[x_cols]

    return X_train, X_test, y_train


def fit_predict(estimator, X, y, X_test, predict_proba = False):

    if estimator is None:
        return None

    if np.any(np.isnan(X.values)) or np.any(np.isnan(X_test.values)):
        imputer=Imputer(strategy="median")
        imputer.fit(X)
        X=imputer.transform(X)
        X_test=imputer.transform(X_test)
    else:   
        X=X.values  #TPOT operators need numpy format for been applied
        X_test=X_test.values
        y=y.values

    estimator.fit(X,y)

    # If the metric requires probability, try to change the ensemble strategy to soft voting 
    if predict_proba:
        try:
            estimator.voting = 'soft'
            return estimator.predict_proba(X_test)

        except:
            estimator.voting = 'hard'
            ohe = OneHotEncoder().fit(y.reshape(-1, 1))
            return ohe.transform(estimator.predict(X_test).reshape(-1,1)).toarray()

    
    return estimator.predict(X_test)

#MOD GRAMMAR: grammar without PolynomialFeatures
def execute_exp(d_id, id_trial, seed, subsample=1, METRIC='balanced_accuracy', predict_proba = False, mod_grammar=False, RUN_TEST=False):

    def write_end_file():
        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("\n"+200*"-"+"\n")
        with open('pipe_found.txt', 'a+') as log_file:
            log_file.write("\n"+200*"-"+"\n")

    try:
        if os.path.exists('exp_checklist.txt'):
            with open('exp_checklist.txt') as log_file:
                print(f"{d_id}_{id_trial}" )
                if f"{d_id}_{id_trial}" in log_file.read():
                    print(f"Skiping problem {d_id}, trial {id_trial}")
                    
                    return 1

        with open('log_exp.txt', 'a+') as log_file:
            log_file.write(str(d_id)+"_"+str(id_trial)+": "+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+"\n")


        if RUN_TEST:
            if mod_grammar:
                p=AUTOCVEClassifier(generations=10, population_size_components=10, population_size_ensemble=20, grammar='/media/Data/Github/THESIS-AUTOML/Analise de Performance/kaggle tests/grammarTPOTMOD', max_pipeline_time_secs=5, \
                                    max_evolution_time_secs=20, cv_evaluation_mode=False, n_jobs=-1, scoring=METRIC, random_state=seed, verbose=1)
            else:
                p=AUTOCVEClassifier(generations=10, population_size_components=10, population_size_ensemble=20, max_pipeline_time_secs=5, \
                                    max_evolution_time_secs=20, cv_evaluation_mode=False, n_jobs=-1, scoring=METRIC, random_state=seed, verbose=1)
        else:
            if mod_grammar:
                p=AUTOCVEClassifier(generations=100, population_size_components=50, mutation_rate_components=0.9, crossover_rate_components=0.9, \
                                    population_size_ensemble=50, mutation_rate_ensemble=0.1, crossover_rate_ensemble=0.9, \
                                    grammar='/media/Data/Github/THESIS-AUTOML/Analise de Performance/kaggle tests/grammarTPOTMOD', \
                                    max_pipeline_time_secs=60, max_evolution_time_secs=TIME_PER_TASK, n_jobs=-1, \
                                    random_state=seed, scoring=METRIC, cv_evaluation_mode=False, verbose=1)
            else: 
                p=AUTOCVEClassifier(generations=100, population_size_components=50, mutation_rate_components=0.9, crossover_rate_components=0.9, \
                                    population_size_ensemble=50, mutation_rate_ensemble=0.1, crossover_rate_ensemble=0.9, \
                                    max_pipeline_time_secs=60, max_evolution_time_secs=TIME_PER_TASK, n_jobs=-1, \
                                    random_state=seed, scoring=METRIC, cv_evaluation_mode=False, verbose=1)



        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("Parameters: "+str(p.get_parameters())+"\n")

        X_train, X_test, y_train = parse_competition_datasets(d_id)

        def handler(signum, frame):
            print("Maximum time reached.")
            raise SystemExit('Time limit exceeded, sending system exit...')
        
        signal.signal(signal.SIGALRM, handler)

        signal.alarm(TIME_PER_TASK + GRACE_PERIOD)

        start = time.time()

        try:
            status = p.optimize(X_train,y_train,subsample_data=subsample)

            if status == 0:
                with open('log_exp.txt', 'a+') as log_file:

                    log_file.write(f"Paused experiment inside AUTOCVE: \n")                

                write_end_file()

                signal.alarm(0)

                return 0

        except KeyboardInterrupt as e:
            with open('log_exp.txt', 'a+') as log_file:

                log_file.write(f"Paused experiment: {e} \n")                

            write_end_file()

            signal.alarm(0)

            return 0

        except SystemExit as e:
            with open('log_exp.txt', 'a+') as log_file:

                log_file.write(f"System exit with message: {e} \n")                
                log_file.write(f"Optimization time: {(time.time()-start)} \n")

                print(e)

                write_end_file()

                signal.alarm(0)
                
                return 1

        except Exception as e:
            if 'Population not initialized' in str(e):                                
                with open('log_exp.txt', 'a+') as log_file:

                    log_file.write(f"Paused experiment inside AUTOCVE: \n")                

                write_end_file()

                signal.alarm(0)

                return 0

            with open('log_exp.txt', 'a+') as log_file:
                
                log_file.write("Optimize method finish with error on problem "+str(d_id)+" trial "+str(id_trial)+"\n")
                log_file.write(str(e)+"\n")
                log_file.write(f"Optimization time: {(time.time()-start)} \n")

                print(e)

                write_end_file()

                signal.alarm(0)

                return 1

        duration = time.time() - start

        signal.alarm(0)

        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("Optimization time: "+ str(duration)+"\n")


        try:
            move("evolution.log", str(d_id)+"_"+str(id_trial)+"_evolution.log")
            move("matrix_sim.log", str(d_id)+"_"+str(id_trial)+"_matrix_sim.log")
            move("evolution_ensemble.log", str(d_id)+"_"+str(id_trial)+"_evolution_ensemble.log")
            move("competition.log", str(d_id)+"_"+str(id_trial)+"_competition.log")
            move("competition_ensemble.log", str(d_id)+"_"+str(id_trial)+"_competition_ensemble.log")
            move("matrix_sim_next_gen.log", str(d_id)+"_"+str(id_trial)+"_matrix_sim_next_gen.log")
        except:
            with open('log_exp.txt', 'a+') as log_file:
                log_file.write("Error when try to move log files on problem "+str(d_id)+" trial "+str(id_trial)+"\n")

                write_end_file()

                return 1

        with open('pipe_found.txt', 'a+') as log_file:
            log_file.write("Problem: "+str(d_id)+", Trial: "+str(id_trial)+"\n\n")

        try:
            submit=fit_predict(p.get_best_voting_ensemble(), X_train, y_train, X_test, predict_proba)

            if submit is not None:
                with open('pipe_found.txt', 'a+') as log_file:
                    log_file.write("AUTOCVE: "+str(p.get_best_voting_ensemble().estimators)+"\n")


                _, get_submit_df = competition_dataprep[d_id]

                get_submit_df(f'../competition_datasets/{d_id}', submit).to_csv(f'{d_id}_{id_trial}_predict.csv', index=False)

                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()

                api.competition_submit(f'{d_id}_{id_trial}_predict.csv','Auto-CVE', d_id)
                
                if not os.path.exists(f'{d_id}-publicleaderboard.csv'):
                    api.competition_leaderboard_download(d_id,'.')

                    import zipfile
                    with zipfile.ZipFile(f'{d_id}.zip', 'r') as zip_ref:
                        zip_ref.extractall('.')

                time.sleep(5)

                submission_info = api.competition_submissions(d_id)[0]

                print(api.competition_submissions(d_id))
                print(submission_info.publicScore)

                with open('results.txt', 'a+') as results_out:
                    results_out.write(str(d_id)+";AUTOCVE;"+str(submission_info.date)+";"+str(submission_info.publicScore)+";"+str(submission_info.privateScore)+";"+str(duration)+"\n")

        except Exception as e:
            with open('log_exp.txt', 'a+') as log_file:
                log_file.write("Experience error when fit pipeline in problem "+str(d_id)+" trial "+str(id_trial)+"\n")
                log_file.write(str(e)+'\n')

                try:
                    log_file.write(f"Pipeline found {p.get_best_voting_ensemble().estimators} \n")
                except Exception as e2:
                    log_file.write(f"Cannot access pipeline with error {e2} \n")

                log_file.write(str(e)+"\n")

                write_end_file()

                return 1

        with open('exp_checklist.txt', 'a+') as log_file:
            log_file.write(f"{d_id}_{id_trial}\n")


    except Exception as e:
        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("Error when try to run the experiment on problem "+str(d_id)+" trial "+str(id_trial)+"\n")
            log_file.write(str(e)+"\n")

        signal.alarm(0)
        
    write_end_file()

    return 1


if __name__ == '__main__':
    RUN_TEST=False

    if len(sys.argv)==2 and sys.argv[1]=='TESTE':
        RUN_TEST=True
        N_TRIALS=1

    elif len(sys.argv)!=4:
        raise Exception("Need to insert the name of experiment, description and number of trials")

    if not RUN_TEST:
        if sys.argv[3].isdigit() and int(sys.argv[3])>0 and int(sys.argv[3])<=20:
            N_TRIALS=int(sys.argv[3])
        else:
            raise Exception("The number of trials is expected to be an integer with value between 1 and 20.")

    folder="./"+str(sys.argv[1])

    if not os.path.exists(folder):
        os.makedirs(folder)

    os.chdir(folder)

    if not os.path.exists('log_exp.txt'):
        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("Experience "+str(sys.argv[1])+"\n")
            
            if not RUN_TEST:
                log_file.write("Description: "+str(sys.argv[2])+"\n\n")
    else:
        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("\nResuming experiment...\n\n")


    list_exp=[('amazon-employee-access-challenge', 'roc_auc', True, 1.0, False), #ROC AUC
              ('challenges-in-representation-learning-the-black-box-learning-challenge', 'accuracy', False, 1.0, False), #Multi-class classification accuracy
              ('otto-group-product-classification-challenge', 'neg_log_loss', True, 1.0, False),# multi-class logarithmic loss
              ('predict-who-is-more-influential-in-a-social-network', 'roc_auc', True, 1.0, False), #ROC AUC
              ('santander-customer-satisfaction', 'roc_auc', True, 1.0, True), #ROC AUC
              ('santander-customer-transaction-prediction', 'roc_auc', True, 0.5, True), #ROC AUC + Test subsample
              ]
              
    status = 0

    for id_exp, metric, predict_proba, subsample, mod_grammar in list_exp:
        print("Instance "+str(id_exp))

        for id_trial in range(N_TRIALS):
            print("Trial "+str(id_trial))
        
            status = execute_exp(id_exp, id_trial, SEED_LIST[id_trial], RUN_TEST=RUN_TEST, METRIC = metric, predict_proba = predict_proba, subsample = subsample, mod_grammar=mod_grammar)

            if status == 0:
                break

        if status == 0:
            break

    if status == 0:
        if os.path.exists('results.txt'):
            copy('results.txt','results_finished.txt')
    else:
        if os.path.exists('results.txt'):
            move('results.txt','results_finished.txt')

