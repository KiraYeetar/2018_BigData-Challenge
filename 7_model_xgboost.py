'''
跑模型
'''
import xgboost as xgb

def get_feature(num, data_label=None):
    register = pd.read_csv(features_path()+'register_feature_'+str(num)+'.csv')
    create = pd.read_csv(features_path()+'create_feature_'+str(num)+'.csv')
    launch = pd.read_csv(features_path()+'launch_feature_'+str(num)+'.csv')
    activity = pd.read_csv(features_path()+'activity_feature_'+str(num)+'.csv')
    feature = pd.merge(register, launch, on='user_id', how='left')
    feature = pd.merge(feature, activity, on='user_id', how='left')
    feature = pd.merge(feature, create, on='user_id', how='left')
    del register
    del create
    del launch

    if data_label is not None:
        label_name = 'label_' + str(num)
        data_label_tmp = data_label[data_label['user_id'].isin(feature['user_id'])]
        data_label_tmp = data_label.loc[:, ['user_id', label_name]]
        data_label_tmp.columns = ['user_id', 'label']
        feature = pd.merge(feature, data_label_tmp, on='user_id', how='left')
    return feature


if __name__ == '__main__':
    # 读标签数据
    data_label = pd.read_csv(basic_path()+'data_label.csv')

    # 读特征数据
    test_x = get_feature('20')
    train_x = get_feature('0', data_label).append(get_feature('1', data_label)).append(
        get_feature('2', data_label)).append(get_feature('3', data_label)).append(
        get_feature('4', data_label)).append(get_feature('5', data_label)).append(
        get_feature('6', data_label)).append(get_feature('7', data_label)).append(
        get_feature('8', data_label)).append(get_feature('9', data_label)).append(
        get_feature('10', data_label))

    train_y = train_x['label']
    test_user = test_x['user_id']

    del train_x['user_id']
    del test_x['user_id']
    del train_x['label']
   
    # XGBOOST 训练
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    params = {
        # 'objective': 'binary:logistic',
        'objective': 'rank:pairwise',
        'eta': 0.03,
        'max_depth': 5,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'min_child_weight': 16,
        'silent': 1,
    }
    bst = xgb.train(params, dtrain, 1500, watchlist = [(dtrain, 'train')])
    pre_label = bst.predict(dtest)

    # 生成结果文件
    pd.DataFrame(data={0:test_user, 1:pre_label}).to_csv('/home/kesci/work/xjy_.txt', index=None, header=None)

	
