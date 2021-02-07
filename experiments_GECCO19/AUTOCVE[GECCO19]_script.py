import pandas as pd
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import sys
import os
import time
import numpy as np
from shutil import move
import signal
TIME_PER_TASK=5400#3600
GRACE_PERIOD=0#60

#SEED_LIST=[5422, 9895, 8342, 4018, 770, 2534, 1213, 6303, 8672, 5397, 514, 9569, 2760, 434, 7223, 2303 ,4184, 6400, 8241, 2129]
SEED_LIST=[2534, 1213, 6303, 8672, 5397, 514, 9569, 2760, 434, 7223, 2303 ,4184, 6400, 8241, 2129]

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


def fit_predict(estimator, X, y, X_test):
    if estimator is None:
        return None
    try:
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
        return estimator.predict(X_test)
    except Exception as e:
        with open('log_exp.txt', 'a+') as file_out:
            file_out.write("Experience error in pipeline:\n"+str(estimator)+"\n")
            file_out.write(str(e)+"\n")
        return None


def execute_exp(d_id, id_trial, seed, subsample=1, METRIC='balanced_accuracy', F_METRIC=balanced_accuracy_score, RUN_TEST=False):
    try:
        with open('log_exp.txt', 'a+') as file_out:
            file_out.write(str(d_id)+"_"+str(id_trial)+": "+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+"\n")


        if RUN_TEST:
            p=AUTOCVEClassifier(generations=10, population_size_components=10, population_size_ensemble=20, grammar='grammarTPOT', max_pipeline_time_secs=5, max_evolution_time_secs=20, n_jobs=-1, random_state=seed, verbose=1)
        else:
            p=AUTOCVEClassifier(generations=100, population_size_components=50, mutation_rate_components=0.9, crossover_rate_components=0.9, population_size_ensemble=50, mutation_rate_ensemble=0.1, crossover_rate_ensemble=0.9, grammar='grammarTPOT', max_pipeline_time_secs=60, max_evolution_time_secs=TIME_PER_TASK, n_jobs=-1, random_state=seed, scoring=METRIC, verbose=1)

        with open('log_exp.txt', 'a+') as file_out:
            file_out.write("Parameters: "+str(p.get_parameters())+"\n")

        X_train, X_test, y_train, y_test, df_types = parse_open_ml(d_id, seed)

        def handler(signum, frame):
            print("Maximum time reached.")
            raise SystemExit('Time limit exceeded, sending system exit...')
        
        signal.signal(signal.SIGALRM, handler)

        signal.alarm(TIME_PER_TASK+GRACE_PERIOD)

        start=time.time()

        try:
            p.optimize(X_train,y_train,subsample_data=subsample)
        except (KeyboardInterrupt, SystemExit) as e:
            print(e)
        duration=time.time()-start

        signal.alarm(0)

        with open('log_exp.txt', 'a+') as file_out:
            file_out.write("Optimization time: "+str(duration)+"\n")


        move("evolution.log", str(d_id)+"_"+str(id_trial)+"_evolution.log")
        move("matrix_sim.log", str(d_id)+"_"+str(id_trial)+"_matrix_sim.log")
        try:
            move("evolution_ensemble.log", str(d_id)+"_"+str(id_trial)+"_evolution_ensemble.log")
        except:
            pass

        with open('pipe_found.txt', 'a+') as file_out:
            file_out.write("Problem: "+str(d_id)+", Trial: "+str(id_trial)+"\n\n")

        try:
            submit=fit_predict(p.get_best_pipeline(),X_train,y_train,X_test)
            if submit is not None:
                with open('pipe_found.txt', 'a+') as file_out:
                    if isinstance(p.get_best_pipeline(), type(Pipeline)):
                        file_out.write("Best pipeline: "+str(p.get_best_pipeline().steps)+"\n")
                    else:
                        file_out.write("Best pipeline: "+str(p.get_best_pipeline())+"\n")

                with open('results.txt', 'a+') as results_out:
                    results_out.write(str(d_id)+";best_pip;"+str(F_METRIC(y_test,submit))+";"+str(duration)+"\n")
        except Exception as e:
            with open('log_exp.txt', 'a+') as file_out:
                file_out.write("Experience error in problem "+str(d_id)+" trial "+str(id_trial)+"\n")
                file_out.write(str(e)+"\n")


        try:
            submit=fit_predict(p.get_voting_ensemble_elite(),X_train,y_train,X_test)
            if submit is not None:
                with open('pipe_found.txt', 'a+') as file_out:
                    file_out.write("Ensemble Elite pipeline: "+str(p.get_voting_ensemble_elite().estimators)+"\n")

                with open('results.txt', 'a+') as results_out:
                    results_out.write(str(d_id)+";ensemble_elite;"+str(F_METRIC(y_test,submit))+";"+str(duration)+"\n")
        except Exception as e:
            with open('log_exp.txt', 'a+') as file_out:
                file_out.write("Experience error in problem "+str(d_id)+" trial "+str(id_trial)+"\n")
                file_out.write(str(e)+"\n")


        try:
            submit=fit_predict(p.get_best_voting_ensemble(),X_train,y_train,X_test)
            if submit is not None:
                with open('pipe_found.txt', 'a+') as file_out:
                    file_out.write("Ensemble AUTOCVE pipeline: "+str(p.get_best_voting_ensemble().estimators)+"\n")

                with open('results.txt', 'a+') as results_out:
                    results_out.write(str(d_id)+";AUTOCVE;"+str(F_METRIC(y_test,submit))+";"+str(duration)+"\n")
        except Exception as e:
            with open('log_exp.txt', 'a+') as file_out:
                file_out.write("Experience error in problem "+str(d_id)+" trial "+str(id_trial)+"\n")
                file_out.write(str(e)+"\n")


        try:
            submit=fit_predict(p.get_voting_ensemble_all(),X_train,y_train,X_test)
            if submit is not None:
                with open('pipe_found.txt', 'a+') as file_out:
                    file_out.write("Ensemble All pipeline: "+str(p.get_voting_ensemble_all().estimators)+"\n")

                with open('results.txt', 'a+') as results_out:
                    results_out.write(str(d_id)+";ensemble_all;"+str(F_METRIC(y_test,submit))+";"+str(duration)+"\n")
        except Exception as e:
            with open('log_exp.txt', 'a+') as file_out:
                file_out.write("Experience error in problem "+str(d_id)+" trial "+str(id_trial)+"\n")
                file_out.write(str(e)+"\n")

    except Exception as e:
        with open('log_exp.txt', 'a+') as file_out:
            file_out.write("Experience error in problem "+str(d_id)+" trial "+str(id_trial)+"\n")
            file_out.write(str(e)+"\n")

    with open('log_exp.txt', 'a+') as file_out:
        file_out.write("\n"+200*"-"+"\n")
    with open('pipe_found.txt', 'a+') as file_out:
        file_out.write("\n"+200*"-"+"\n")


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

    with open('log_exp.txt', 'a+') as file_out:
        file_out.write("Experience "+str(sys.argv[1])+"\n")
        if not RUN_TEST:
            file_out.write("Description: "+str(sys.argv[2])+"\n\n")

    list_exp=[11, 29, 42, 50, 151, 377, 1038, 1459, 1464, 40971]
    for id_exp in list_exp:
        print("Problem "+str(id_exp))
        for id_trial in range(N_TRIALS):
            print("Trial "+str(id_trial))
            execute_exp(id_exp, id_trial, SEED_LIST[id_trial], RUN_TEST=RUN_TEST)

    move('results.txt','results_finished.txt')

