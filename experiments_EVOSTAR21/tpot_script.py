from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import sklearn

if sklearn.__version__>='0.22':
    from sklearn.impute import SimpleImputer as Imputer
else:
    from sklearn.preprocessing import Imputer

import numpy as np
import pandas as pd

from shutil import move, copy
import signal
import sys
import os
import time

TIME_PER_TASK=5400
GRACE_PERIOD=100


SEED_LIST=[5249444, 7592240, 4510071, 2688207, 5344429, 6581462, 6381116, 4901935, 1977886, 2591851, 9404208, 4098546]

def parse_open_ml(d_id, seed):
    """Function that processes each dataset into an interpretable form
    Args:
        d_id (int): dataset id
        seed (int): random seed for replicable results
    Returns:
        A tuple of the train / test split data along with the column types
    """

    df = pd.read_csv('../datasets/{0}.csv'.format(d_id))
    df_types = pd.read_csv('../datasets/{0}_types.csv'.format(d_id))

    df_valid = df[~df['target'].isnull()]

    x_cols = [c for c in df_valid.columns if c != 'target']
    X = df_valid[x_cols]
    y = df_valid['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=seed)

    return X_train, X_test, y_train, y_test, df_types


def impute_data(X, X_test):
    
    if np.any(np.isnan(X.values)) or np.any(np.isnan(X_test.values)):
        imputer=Imputer(strategy="median")
        imputer.fit(X)
        X=imputer.transform(X)
        X_test=imputer.transform(X_test)
    else:   
        X=X.values  #TPOT operators need numpy format for been applied
        X_test=X_test.values

    return X, X_test

def execute_exp(d_id, id_trial, seed, subsample=1, METRIC='balanced_accuracy', F_METRIC=balanced_accuracy_score):
    
    def write_end_file():
        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("\n"+200*"-"+"\n")

    try:
        if os.path.exists('exp_checklist.txt'):
            with open('exp_checklist.txt') as log_file:
                print(f"{d_id}_{id_trial}" )
                if f"{d_id}_{id_trial}" in log_file.read():
                    print(f"Skiping problem {d_id}, trial {id_trial}")
                    
                    return 1

        if os.path.exists('{d_id}_{id_trial}'):
            os.rmdir('{d_id}_{id_trial}')


        with open('log_exp.txt', 'a+') as log_file:
            log_file.write(str(d_id)+"_"+str(id_trial)+": "+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+"\n")
      
        p=TPOTClassifier(scoring=METRIC, verbosity=3, n_jobs=-1, random_state=seed,\
                         periodic_checkpoint_folder=f"{d_id}_{id_trial}", \
                         max_time_mins=TIME_PER_TASK//60)

    
        X_train, X_test, y_train, y_test, _ = parse_open_ml(d_id, seed)
        X_train, X_test = impute_data(X_train, X_test)

        def handler(signum, frame):
            print("Maximum time reached.")
            raise SystemExit('Time limit exceeded, sending system exit...')

        signal.signal(signal.SIGALRM, handler)

        start=time.time()

        signal.alarm(TIME_PER_TASK + GRACE_PERIOD)

        try:
            p.fit(X_train, y_train.values)

            signal.alarm(0)

            duration=time.time()-start
            
            print("WRITING RESULTS")
            
            with open('log_exp.txt', 'a+') as file_out:
                file_out.write("Optimization time: "+str(duration)+"\n")

        except KeyboardInterrupt as e:
            with open('log_exp.txt', 'a+') as log_file:

                log_file.write(f"Paused experiment: {e} \n")       

            write_end_file()

            signal.alarm(0)         

            return 0

        except Exception as e:
            with open('log_exp.txt', 'a+') as log_file:
                
                log_file.write("Optimize method finish with error on problem "+str(d_id)+" trial "+str(id_trial)+"\n")
                log_file.write(str(e)+"\n")
                log_file.write(f"Optimization time: {(time.time()-start)} \n")

                print(e)

                write_end_file()

                signal.alarm(0)

                return 1

        try:
            with open('results.txt', 'a+') as results_out:
                results_out.write(str(d_id)+";TPOT;"+str(F_METRIC(y_test.values, p.predict(X_test)))+";"+str(duration)+"\n")
        
        except Exception as e:
            with open('log_exp.txt', 'a+') as log_file:
                log_file.write("Experience error when fit pipeline in problem "+str(d_id)+" trial "+str(id_trial)+"\n")

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
    folder="./tpot_exp"

    if not os.path.exists(folder):
        os.makedirs(folder)

    os.chdir(folder)

    if not os.path.exists('log_exp.txt'):
        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("Experience TPOT\n")            
    else:
        with open('log_exp.txt', 'a+') as log_file:
            log_file.write("\nResuming experiment...\n\n")



    list_exp=[15, 37, 307, 451, 458, 469, 1476, 1485, 1515, 1590, 6332, 23517, 40496, 40499, 40994]
    N_TRIALS=10

    for id_exp in list_exp:
        print("Problem "+str(id_exp))

        for id_trial in range(N_TRIALS):
            print("Trial "+str(id_trial))
        
            status = execute_exp(id_exp, id_trial, SEED_LIST[id_trial])

            if status == 0:
                break

        if status == 0:
            break

    if status == 0:
        if os.path.exists('results.txt'):
            copy('results.txt','results_TPOT.txt')
    else:
        if os.path.exists('results.txt'):
            move('results.txt','results_TPOT.txt')
