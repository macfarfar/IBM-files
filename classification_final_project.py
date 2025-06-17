import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import (model_selection, ensemble, linear_model, svm,
                     pipeline, neighbors, preprocessing, metrics)
from pathlib import Path
import sys
from datetime import datetime as dt
import os
import json


TESTING = 0
REDUCE_FEATURES = 1


def run_dir_string():
    run_dir = f'run_outputs\\{'TEST-' if TESTING else ''}run-{'reduced-' if REDUCE_FEATURES else ''}{dt.now()}'.replace(
        '.', '-').replace(':', '-')
    try:
        os.mkdir(run_dir)
    except FileExistsError:
        os.mkdir(run_dir + 'x')
    return run_dir


def reduce_features_function(x_all, lassoed_coefficients):
    if x_all.shape[1] > 450:
        mask = np.abs(lassoed_coefficients).sum(
            axis=0)
        drop_list = x_all.columns[pd.Series(mask).apply(lambda x: not bool(x))]
        x_all = x_all.drop(drop_list, axis=1)
        return x_all
    else:
        return None


def simplify_labels(data: pd.DataFrame()):
    group_mapping = {
        'HS': None,
        'Stage_I_II': 'CRC',
        'Stage_0': 'CRC',
        'Stage_III_IV': 'CRC',
        'MP': None
    }
    data['Study.Group'] = data['Study.Group'].apply(lambda x: group_mapping[x] if x in group_mapping else x)
    data.dropna(subset=['Study.Group'], inplace=True)
    return data


def get_data():
    root = 'microbiome-metabolome-curated-data/data/processed_data/YACHIDA_CRC_2019'
    metabolite_data = pd.read_csv(Path(root, 'mtb.tsv'), sep='\t')
    metadata = pd.read_csv('microbiome-metabolome-curated-data/data/processed_data/YACHIDA_CRC_2019/metadata.tsv',
                           sep='\t')
    all_data = pd.concat([
        metabolite_data.set_index('Sample'),
        # pd.read_csv(Path(root, 'species.tsv'), sep='\t').set_index('Sample'),
        # pd.read_csv(Path(root, 'genera.tsv'), sep='\t').set_index('Sample'),
        metadata.set_index('Sample')[
            ['Study.Group', 'BMI', 'Age', 'Gender',
             'Brinkman Index', 'Alcohol']]],
        axis=1)
    all_data = simplify_labels(all_data)
    y_all = all_data.pop('Study.Group')
    x_all = all_data
    x_all.Gender = x_all.Gender.apply(lambda x: 1 if x == 'Male' else 0)
    return x_all, y_all


def train_models(x_all=None, y_all=None, run_dir=run_dir_string(),
                 reduce_features=REDUCE_FEATURES, run_models=None, features_reduced=False):

    if run_models is None:
        run_models = models.keys()

    pipelines = {}
    classifier_string = 'classifier'

    if x_all is None or y_all is None:
        x_all, y_all = get_data()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_all, y_all, test_size=0.3, random_state=42)

    # Redirect stdout to a file
    try:
        with open(Path(run_dir, f'y_test.json'), 'w') as predictions_file:
            json.dump({'predictions': list(y_test)}, predictions_file)
        with open(Path(run_dir, 'log.txt'), 'w') as log_file:
            if not reduce_features or features_reduced:
                print('Switching to text logging')
                sys.stdout = log_file
            for model in run_models:
                # returns early if reduce_features is on
                print(f'Running {model}')
                pipelines[model] = {}
                pipelines[model]['pipeline'] = pipeline.Pipeline(steps=[
                    ('scaling', preprocessing.StandardScaler()),
                    ('classifier', models[model]['model'])])
                if 'cv' in models[model]:
                    cv = models[model]['cv']
                else:
                    cv = model_selection.GridSearchCV
                pipelines[model]['search'] = cv(
                    estimator=pipelines[model]['pipeline'],
                    param_grid={f'{classifier_string}__{p}': models[model]['params'][p] for p in
                                models[model]['params']},
                    cv=10,
                    scoring='accuracy')
                pipelines[model]['search_results'] = pipelines[model]['search'].fit(x_train, y_train)
                print(pipelines[model]['search_results'].best_score_)
                print(pipelines[model]['search_results'].best_estimator_)
                try:
                    y_prob = pipelines[model]['search_results'].predict_proba(x_test)
                except AttributeError:
                    print(f'{model} does not have probability prediction output')
                y_pred = pipelines[model]['search_results'].predict(x_test)
                pipelines[model]['test_accuracy'] = metrics.balanced_accuracy_score(y_test, y_pred)
                print(f'Best accuracy for {model}: {pipelines[model]['test_accuracy']}')
                output_dict = {
                    'predictions': list(y_pred),
                    'accuracy': pipelines[model]['test_accuracy'],
                    'model_search_results': {
                        r: pipelines[model]['search_results'].cv_results_[r].tolist()
                        for r in pipelines[model]['search_results'].cv_results_
                        if not isinstance(
                            pipelines[model]['search_results'].cv_results_[r], list)}}

                if reduce_features and not features_reduced:
                    # if X is over the feature limit, trim lassoed features
                    # else set reduce_features flag to False
                    # rf() returns None if the data set has been reduced already
                    x_reduced = reduce_features_function(
                        x_all, pipelines['logreg']['search_results'].best_estimator_.steps[1][1].coef_)
                    return x_reduced, run_dir

                if features_reduced:
                    model_r = model + '_reduced'
                else:
                    model_r = model
                metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
                plt.title(model_r)
                plt.savefig(Path(run_dir, f'confusion_matrix_{model_r}, '
                                          f'precision={pipelines[model]['search_results'].best_estimator_.score(x_test, y_test)}.png'))
                with open(Path(run_dir, f'search_results_{model_r}.json'), 'w') as results_file:
                    json.dump(output_dict, results_file)

    except KeyboardInterrupt:
        handle_error('STOPPED', run_dir)
    except Exception:
        handle_error('FAILED', run_dir)
    return pipelines, run_dir


def handle_error(name, rd):
    if not os.listdir(rd):
        os.rmdir(rd)
    elif name:
        os.rename(rd, rd.replace('run-', f'{name}-run-'))
    try:
        raise
    except RuntimeError:
        pass


def multi_model_voting(x_all, y_all, pipelines_dict=None, run_dir=run_dir_string(), models_to_run=None):
    if models_to_run is None:
        models_to_run = []
    x_voting_train, x_voting_test, y_voting_train, y_voting_test = model_selection.train_test_split(
        x_all, y_all, test_size=.3, random_state=42)
    if not pipelines_dict:
        pipelines_dict, _ = train_models(x_voting_train, y_voting_train, run_dir=run_dir, features_reduced=True,
                                         run_models=models_to_run)

    classifiers_to_stack = [(
        model, pipelines_dict[model]['search_results'].best_estimator_) for model in pipelines_dict]

    skew_weights = {0: 0.7, 1: 0.3}
    for final_estimator in [(linear_model.LogisticRegression(class_weight=skew_weights), 'logreg'),
                            (ensemble.RandomForestClassifier(), 'rf')]:
        sc = ensemble.StackingClassifier(classifiers_to_stack, final_estimator=final_estimator[0])
        sc.fit(x_voting_train, y_voting_train, )
        y_voting_pred = sc.predict(x_voting_test)
        metrics.RocCurveDisplay.from_estimator(sc, x_voting_test, y_voting_test)
        auc_test = sc.predict_proba(x_voting_test)[:, 1]
        plt.title(f'Stacked ({final_estimator[1]}) '
                  f'AUC: {metrics.roc_auc_score(y_voting_test, auc_test)}')
        plt.savefig(Path(run_dir, f'roc_plot_stacked_{final_estimator[1]}.png'))
        # plt.close()
        metrics.ConfusionMatrixDisplay.from_predictions(y_voting_test, y_voting_pred)
        plt.title(f'Stacked ({final_estimator[1]} Final, precision={sc.score(x_voting_test, y_voting_test)}')
        plt.savefig(Path(run_dir, f'confusion_matrix_stacked_{final_estimator[1]}.png'))

        with open(Path(run_dir, f'stacked_classification_output_{final_estimator[1]}.json'), 'w') as results_file:
            json.dump(
                {'predictions': list(y_voting_pred),
                 'y_test': list(y_voting_test),
                 'accuracy': metrics.balanced_accuracy_score(y_voting_test, y_voting_pred)},
                results_file)


grid_size=10
models = {'logreg': {'model': linear_model.LogisticRegression(penalty='l1', solver='liblinear'),
                     'params': {'C': np.linspace(0.5, 10, grid_size)}},
          'svc': {'model': svm.SVC(probability=False),
                  'params': {
                      'C': np.linspace(0.5, 10, grid_size),
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}},
          'knn': {'model': neighbors.KNeighborsClassifier(),
                  'params': {
                      'n_neighbors': np.linspace(3, 10, grid_size, dtype=int),
                      'weights': ['uniform', 'distance']}},
          'rf': {'model': ensemble.RandomForestClassifier(),
                 'params': {'n_estimators': np.linspace(20, 50, grid_size, dtype=int),
                            'max_depth': np.linspace(10, 16, grid_size, dtype=int),
                            'min_samples_leaf': np.unique(np.linspace(2, 2, grid_size, dtype=int)),
                            'min_samples_split': np.unique(np.linspace(3, 6, grid_size, dtype=int))
                            }},
          }


if __name__ == '__main__':
    run_directory = run_dir_string()
    try:
        x_reduced, run_directory = train_models(reduce_features=True, run_models=['logreg'])
        if TESTING:
            run_models = ['logreg']
        else:
            run_models = models.keys()
        _, y_all = get_data()
        # x_reduced.corr().abs().sum(axis=1).sort_values()
        multi_model_voting(x_reduced, y_all, run_dir=run_directory, models_to_run=run_models)
    except KeyboardInterrupt:
        handle_error('STOPPED', run_directory)
    except Exception:
        handle_error('FAILED', run_directory)
    handle_error('', run_directory)
