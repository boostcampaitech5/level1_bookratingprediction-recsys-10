import time
import argparse
import pandas as pd
import numpy as np
import json
import pickle 
import re
import os

from src.utils import Logger, Setting, models_load
from src.data import Naive_data_load, Naive_data_split, Naive_data_loader
from src.train import train, test

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool


from datetime import datetime



# 현재 시간을 사용하여 파일명 생성
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
result_path = f"/opt/ml/code/submit/{timestamp}_CATBOOST.csv"
folder_path = f"/opt/ml/code/log/{timestamp}_CATBOOST"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def main(args):
    
    Setting.seed_everything(args.seed)
    
    with open('sample_data.pkl', 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    with open('summary_last2.pkl', 'rb') as pkl_file:
        data_cat = pickle.load(pkl_file)
        data_cat['train'] = data_cat['train'].drop(['summary_label'], axis=1)
        data_cat['test'] = data_cat['test'].drop(['summary_label'], axis=1)

    def rmse(real, predict):
        return np.sqrt(np.mean((real-predict) ** 2))

    def mae(real, predict):
        return np.mean(np.abs(real-predict))
    
    print('args.problem={}'.format(args.problem), '------------------------------')
    
    if args.problem == 'Classifier':
        X_train_cat, X_valid_cat, y_train_cat, y_valid_cat = train_test_split(data_cat['train'].drop(['rating'], axis=1), data_cat['train']['rating'], test_size=0.2, shuffle=True, random_state=args.seed)
        
        X_test_cat = data_cat['test']
        

        cat_list = [x for x in X_train_cat.columns.tolist() if x not in ['age', 'year_of_publication']]
        
        
        catboost_cl = CatBoostClassifier(
                                        iterations = args.iterations,
                                        loss_function = args.loss_function, #분류
                                        eval_metric = args.eval_metric, 
                                        verbose= args.verbose ,
                                        early_stopping_rounds= 300,
                                        cat_features= cat_list,
                                        random_seed= args.seed,
                                        task_type='GPU', # Use GPU for training
                                        devices='0:1', # Specify the device(s) to use, in this case, use device 0 and 1
                                        auto_class_weights = args.auto_class_weights,  #클래스 불균형 해결을 위해 클래스 가중치 계산
                                        train_dir = folder_path,
                                        )

        # Fit
        catboost_cl.fit(
            X_train_cat, 
            y_train_cat,
            eval_set = (X_valid_cat, y_valid_cat),
            use_best_model = True,
            save_snapshot = True,
            snapshot_file = '/opt/ml/code/saved_models/{}.cbm'.format(timestamp)
            )

        

        # Predict
        #cat_boost_cl = CatBoostClassifier()
        #cat_boost_cl.load_model('/opt/ml/code/saved_models/{}.cbm'.format(timestamp))

        y_pred = catboost_cl.predict(X_valid_cat)
        print(accuracy_score(y_valid_cat, y_pred))
        print('MAE : ', mae(y_valid_cat, y_pred.squeeze(1)))
        print('RMSE : ', rmse(y_valid_cat, y_pred.squeeze(1)))

        y_last = catboost_cl.predict(X_test_cat)

        params = catboost_cl.get_params()
        with open(f"{folder_path}/model.json", 'w') as f:
            json.dump(params, f)

        # sample_submission.csv 파일 읽기
        submission = pd.read_csv("/opt/ml/data/sample_submission.csv")
        # 예측된 결과를 세 번째 열에 채워 넣기
        submission['rating'] = y_last
        # 결과를 CSV 파일로 저장하기
        result_path = f"/opt/ml/code/submit/{timestamp}_{round(rmse(y_valid_cat, y_pred.squeeze(1)), 3)}_{args.problem}_CATBOOST.csv"
        submission.to_csv(result_path, index=False)

    elif args.problem == 'Regressor':
        X_train_cat, X_valid_cat, y_train_cat, y_valid_cat = train_test_split(data_cat['train'].drop(['rating'], axis=1), data_cat['train']['rating'], test_size=0.2, shuffle=True, random_state=args.seed)
        X_test_cat = data_cat['test'].drop(['rating'], axis=1)
 

        cat_list = ['user_id', 'isbn', 'location_country', 'book_title',
       'book_author', 'year_of_publication', 'publisher', 'language',
       'category']
        #cat_list = [x for x in X_train_cat.columns.tolist() if x not in ['age', 'year_of_publication']]


        catboost_reg = CatBoostRegressor(
            iterations= args.iterations,
            loss_function= args.loss_function,  # 회귀 문제를 위한 손실 함수로 RMSE를 사용합니다.
            eval_metric= args.eval_metric,  # 평가 지표로 RMSE를 사용합니다.
            verbose= args.verbose,
            early_stopping_rounds= 700,
            cat_features= cat_list,
            task_type= 'GPU',  # GPU를 사용하여 학습합니다.
            devices= '0:1',  # 0번과 1번 장치를 사용합니다.
            train_dir = folder_path,
            learning_rate = args.learning_rate,
            depth = args.depth,
            use_best_model = True,
            bootstrap_type = args.bootstrap_type,
            l2_leaf_reg=args.l2_leaf_reg,
            feature_border_type=args.feature_border_type,
            dev_score_calc_obj_block_size = 5000000

        )

        # 학습
        catboost_reg.fit(
            X_train_cat,
            y_train_cat,
            eval_set=(X_valid_cat, y_valid_cat),  # 검증 세트를 지정하여 모델의 성능을 평가합니다.
            use_best_model = True,
            save_snapshot = True,
            snapshot_file = '/opt/ml/code/saved_models/{}.cbm'.format(timestamp)
        )

        params = catboost_reg.get_params()
        with open(f"{folder_path}/model.json", 'w') as f:
            json.dump(params, f)

        # Predict
        #cat_boost_reg = CatBoostRegressor()
        #cat_boost_reg.load_model('/opt/ml/code/saved_models/{}.cbm'.format(timestamp))
        y_pred = catboost_reg.predict(X_valid_cat)
        print("this is X_valid_cat:",  X_valid_cat)
        print('MAE : ', mae(y_valid_cat, y_pred))
        print('RMSE : ', rmse(y_valid_cat, y_pred))



        print("this is X_test_cat:",  X_test_cat)
        y_last = catboost_reg.predict(X_test_cat)
        # sample_submission.csv 파일 읽기
        submission = pd.read_csv("/opt/ml/data/sample_submission.csv")
        # 예측된 결과를 세 번째 열에 채워 넣기
        submission['rating'] = y_last
        # 결과를 CSV 파일로 저장하기
        result_path = f"/opt/ml/code/submit/{timestamp}_{round(rmse(y_valid_cat, y_pred),3)}_CATBOOST.csv"
        submission.to_csv(result_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')

    #인자를 추가할 수 있는 상태로 만듦
    arg = parser.add_argument

    #이 아래는 Classifier에 주로 있는 것 
    arg('--problem', default='Regressor', choices=['Regressor', 'Classifier'], help='문제 유형을 설정합니다. Regressor 또는 Classifier 중 하나를 선택하세요.')
    arg('--iterations', type=int, default = 1000, help= '부스팅 반복 횟수로, 이 값이 높을수록 더 많은 트리가 생성되며 학습이 더 길어집니다.')
    arg('--learning_rate', type=float, default = 0.03, help= '부스팅 스텝별로 사용되는 학습률로, 이 값이 작을수록 모델이 더 느리게 학습됩니다.')
    arg('--depth', type=int, default=6, help='트리의 최대 깊이로, 깊이가 높을수록 모델의 복잡성이 증가합니다.')
    arg('--l2_leaf_reg', type=float, default=3.0, help='L2 규제 계수로, 값이 클수록 모델에 더 많은 규제가 적용되어 과적합을 방지할 수 있습니다.')
    arg('--model_size_reg', type=float, default=0.5, help='모델 크기 규제 계수로, 값이 클수록 모델의 크기가 작아집니다.')
    arg('--rsm', type=float, default=1.0, help='각 부스팅 스텝에서 사용하는 피쳐의 비율로, 값이 작을수록 피쳐를 적게 사용하여 더 빠른 학습이 가능합니다.')
    arg('--border_count', type=int, default=254, help='피쳐의 경계 개수로, 이 값이 클수록 더 많은 분할이 가능해집니다.')
    arg('--feature_border_type', default='GreedyLogSum', help='경계 타입을 결정하는 방법으로, 여러 옵션을 사용할 수 있습니다.')
    arg('--fold_permutation_block_size', type=int, default=None, help='Fold permutation block size를 설정합니다. 정수를 입력하세요.')
    arg('--per_float_feature_quantization', default=None, help='각 부동 소수점 피쳐에 대한 양자화 설정을 정의합니다.')
    arg('--input_borders', default=None, help='사용자 지정 경계를 사용하여 피쳐를 양자화합니다.')
    arg('--output_borders', default=None, help='파일 경로를 지정하여 양자화 된 경계를 출력합니다.')
    arg('--fold_permutation_block', type=int, default=64, help='스토캐스틱 그래디언트 부스팅의 교차 검증 블록 크기를 결정합니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--od_wait', type=int, default=20, help='오버피팅 탐지기가 과적합으로 판단하기 전에 대기할 부스팅 반복 횟수입니다.')  
    arg('--od_type', default='IncToDec', help='오버피팅 탐지 방법으로, IncToDec 또는 Iter 중 하나를 선택할 수 있습니다.')  
    arg('--nan_mode', default='Min', help='결측값 처리 방법으로, Min 또는 Max 중 하나를 선택할 수 있습니다.')  
    arg('--counter_calc_method', default='SkipTest', help='범주형 피쳐의 카운터를 계산하는 방법을 설정합니다.')  
    arg('--gpu_ram_part', type=float, default=0.95, help='GPU 학습 시 사용할 메모리의 비율입니다.')  
    arg('--classes_count', type=int, default=0, help='다중 클래스 분류 문제의 클래스 수를 설정합니다.')  
    arg('--auto_class_weights', default=None, help='클래스 불균형 문제를 해결하기 위해 자동으로 클래스 가중치를 계산합니다. Balanced, SqrtBalanced')  
    arg('--class_weights', default=None, help='클래스 가중치를 사용자 지정 값으로 설정합니다.')  
    arg('--one_hot_max_size', type=int, default=2, help='원-핫 인코딩을 적용할 범주형 변수의 최대 카디널리티를 설정합니다.')  
    arg('--random_strength', type=float, default=1, help='스플릿 결정에 추가되는 노이즈의 양을 결정하는 데 사용되는 임의 계수입니다.')  
    arg('--name', default=None, help='모델의 이름을 설정합니다.')  
    arg('--ignored_features', default=None, help='학습 중 무시할 피쳐의 인덱스 목록을 설정합니다.')  
    arg('--train_dir', default=None, help='학습 중 생성되는 출력 파일이 저장될 디렉토리를 설정합니다.')  
    arg('--custom_metric', default=None, help='사용자 정의 메트릭을 설정합니다.')
    arg('--eval_metric', default=None, help='평가 메트릭을 설정합니다. 설정하지 않으면 손실 함수에 따라 기본 메트릭이 자동으로 선택됩니다.')  
    arg('--bagging_temperature', type=float, default=1.0, help='배깅 온도로, 값이 클수록 더 많은 배깅이 발생합니다.')  
    arg('--save_snapshot', type=bool, default=False, help='학습 중 스냅샷 저장 여부를 결정합니다.') 
    arg('--snapshot_file', default=None, help='스냅샷 파일의 경로를 설정합니다.' ) 
    arg('--snapshot_interval', type=int, default=600, help='스냅샷 저장 간격을 초 단위로 설정합니다.' ) 
    arg('--fold_len_multiplier', type=int, default=2, help='스토캐스틱 그래디언트 부스팅에서 교차 검증 폴드의 길이를 결정하는 데 사용되는 배수입니다.' ) 
    arg('--used_ram_limit', default=None, help='학습 중 사용할 수 있는 RAM 용량의 최대 한도를 설정합니다.' ) 
    arg('--gpu_cat_features_storage', default='CpuPinnedMemory', help='GPU에서 범주형 피쳐를 저장하는 방법을 결정합니다.' ) 
    arg('--data_partition', default='FeatureParallel', help='데이터 분할 방식을 설정합니다. FeatureParallel 또는 DocParallel 중 하나를 선택할 수 있습니다.' ) 
    arg('--metadata', default=None, help='모델에 추가할 메타데이터를 설정합니다.' ) 
    arg('--early_stopping_rounds', default=None, help='검증 세트에 대한 성능이 개선되지 않은 경우, 조기 종료할 최대 반복 횟수를 설정합니다.' ) 
    arg('--min_data_in_leaf', type=int, default=1, help='리프 노드에 포함되어야 하는 최소 데이터 수를 설정합니다. 이 값을 높이면 모델의 복잡도가 줄어들어 과적합을 방지할 수 있습니다.' ) 
    arg('--grow_policy', default='SymmetricTree', help='트리의 성장 정책을 설정합니다. SymmetricTree, Depthwise, 또는 Lossguide 중 하나를 선택할 수 있습니다.' ) 
    arg('--min_child_samples', type=int, default=20, help='스플릿을 생성하기 전에 필요한 최소 샘플 수를 설정합니다.' ) 
    arg('--max_leaves', type=int, default=64, help='트리당 최대 리프 수를 설정합니다.' ) 
    arg('--num_leaves', type=int, default=31, help='트리당 최대 리프 수를 설정합니다. 이 값은 max_leaves와 동일한 기능을 하지만, 다른 라이브러리와의 호환성을 위해 추가되었습니다.' )  
    arg('--leaf_estimation_backtracking', default='AnyImprovement', help='리프 가중치 추정 과정에서 백트래킹을 수행하는 방식을 설정합니다.') 
    arg('--feature_weights', default=None, help='각 피쳐에 대한 가중치를 설정합니다. 이를 통해 모델이 특정 피쳐에 더 집중하도록 조절할 수 있습니다.') 
    arg('--penalties_coefficient', type=float, default=1, help='학습 중 트리에 적용되는 모든 패널티의 계수를 설정합니다.') 
    arg('--first_feature_use_penalties', type=bool, default=False, help='첫 번째 피쳐에 패널티를 적용할지 여부를 결정합니다.') 
    arg('--per_object_feature_penalties', default=None, help='개별 객체에 대해 피쳐 패널티를 적용합니다.') 
    arg('--sparse_features_conflict_fraction', type=float, default=0, help='희소 피쳐 간 충돌 가능성에 대한 패널티를 설정합니다.') 
    arg('--boosting_type', default='Plain', help='부스팅 타입을 설정합니다. 가능한 옵션은 Plain, Ordered입니다.') 
    arg('--task_type', default='CPU', help='학습에 사용할 하드웨어 유형을 설정합니다. 가능한 옵션은 CPU, GPU입니다.')
    #이 아래는 Regressor에 주로 있는것
    # loss-function은 필수인 것으로 하자.
    arg('--loss_function', required=True, default='RMSE', help='회귀문제에서 사용할 수 있는 손실함수는 RMSE, MAE, Poisson, Quantile입니다. 기본은 RMSE입니다. 분류문제에서 사용할 수 있는 손실함수는 Accuracy, AUC, F1, Precision, Recall 입니다. 기본은 MultiClass입니다.')
    arg('--leaf_estimation_iterations', type=int, default=1, help='리프 가중치를 업데이트하는 반복 횟수입니다.')
    arg('--leaf_estimation_method', default='Newton', help='리프 가중치를 업데이트하는 방법입니다. Newton')
    arg('--thread_count', type=int, default=-1, help='병렬 처리를 위한 스레드 수. -1일 경우, 모든 가능한 스레드를 사용합니다.')
    arg('--random_seed', default=None, help='난수 생성 시드. 이 값을 지정하면 결과를 재현할 수 있습니다.')
    arg('--use_best_model', type=bool, default=None, help='True일 경우, 가장 좋은 검증 점수를 기록한 모델을 사용합니다.')
    arg('--best_model_min_trees', type=int, default=1, help='가장 좋은 모델로 사용될 수 있는 최소 트리 개수를 지정합니다.')
    arg('--verbose', type=bool, default=False, help='학습 중에 진행 상황을 출력할지 여부를 결정합니다.')
    arg('--silent', type=bool, default=None, help='True일 경우, 로그 출력을 완전히 비활성화합니다.')
    arg('--logging_level', default='Info', help='로그 출력 레벨을 결정합니다. Silent, Info, Verbose, Debug 중 하나를 선택할 수 있습니다.')
    arg('--metric_period', type=int, default=1, help='로그 출력 주기를 결정합니다. 값이 클수록 출력 빈도가 낮아집니다.')
    arg('--ctr_leaf_count_limit', default=None, help='카테고리 피쳐를 처리하기 위한 최대 리프 수를 지정합니다. None일 경우, 제한이 없습니다.')
    arg('--store_all_simple_ctr', type=bool, default=False, help='모든 단순 카운터를 저장할지 여부를 결정합니다.')
    arg('--max_ctr_complexity', type=int, default=4, help='카운터 복잡도의 최대 값입니다.')
    arg('--has_time', type=bool, default=False, help='시계열 데이터의 경우 True로 설정합니다.')
    arg('--allow_const_label', type=bool, default=False, help='True일 경우, 상수 레이블을 허용합니다.')
    arg('--target_border', default=None, help='분류 문제의 경우, 레이블을 이진으로 변환하기 위한 경계 값을 지정합니다.')
    arg('--bayesian_matrix_reg', type=float, default=0.1, help='베이지안 정규화 행렬의 규제 계수입니다.')
    arg('--priors', default=None, help='사전 확률 정보를 설정합니다. None일 경우, 기본값을 사용합니다.')
    arg('--score_function', default='Cosine', help='점수 함수를 설정합니다. Cosine, L2, NewtonL2, NewtonCosine 중 하나를 선택할 수 있습니다.')
    arg('--bootstrap_type', default='Bayesian', help='부스팅시 사용할 부트스트랩 방법입니다. Bayesian, Bernoulli, MVS, No 중 하나를 선택할 수 있습니다.')
    arg('--subsample', default=None, help='부트스트랩 샘플링 비율입니다. Bernoulli 또는 MVS 부트스트랩 방법을 사용할 때 설정합니다.')
    arg('--sampling_unit', default='Object', help='샘플링 단위를 설정합니다. Object 또는 Group 중 하나를 선택할 수 있습니다.')
    arg('--dev_score_calc_obj_block_size', type=int, default=5000000, help='GPU 학습시 GPU 장치에서 점수 계산 블록 크기를 결정합니다.')
    arg('--max_depth', default=None, help='최대 트리 깊이를 설정합니다. depth와 동일한 역할을 하며, 이 값이 설정되면 depth 값은 무시됩니다.')
    arg('--n_estimators', default=None, help='생성할 트리의 개수를 설정합니다. iterations와 동일한 역할을 하며, 이 값이 설정되면 iterations 값은 무시됩니다.')
    arg('--eta', default=None, help='부스팅 스텝별 사용되는 학습률을 설정합니다. learning_rate와 동일한 역할을 하며, 이 값이 설정되면 learning_rate 값은 무시됩니다.')
    arg('--cat_features', default=None, help='범주형 피쳐의 인덱스 목록입니다. 이를 통해 CatBoost가 범주형 피쳐를 자동으로 인식할 수 있습니다.')
    arg('--score_function_change_interval', type=int, default=10, help='점수 함수 변경 간격입니다. 이 값이 클수록 점수 함수 변경 빈도가 낮아집니다.')
    #Regressor를 쓸 때는 ctr_history를 Sample, 'Group', 'Sample'만 쓸 수 있음.
    arg('--ctr_history_unit', default = 'Sample', help='카운터 업데이트 기록 단위를 설정합니다. SampleRate, LearnEventsCount, TestEventsCount 중 하나를 선택할 수 있습니다.')
    arg('--monotone_constraints', default = None, help='단조 제약 조건을 설정합니다. 이는 피쳐와 예측 사이의 단조 관계를 유지하는 데 도움이 됩니다.')

    #인자 추가 완료
    args = parser.parse_args()
    main(args)