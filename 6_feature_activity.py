'''
活动表特征

另外

这个真的非常慢
'''

if __name__ == '__main__':
    up = ups()
    down = downs()

    for feature_num in features_addday_list():
        # 读数据
        register = pd.read_csv(data_path() + 'user_register_log.txt', sep='\t', header=None,
                               names=['user_id', 'register_day', 'register_type', 'device_type'],
                               dtype={0: np.uint32, 1: np.uint8, 2: np.uint16, 3: np.uint16})
        activity = pd.read_csv(data_path() + 'user_activity_log.txt', sep='\t', header=None,
                               names=['user_id', 'act_day', 'page', 'video_id', 'author_id', 'action_type'],
                               dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint32, 4: np.uint32, 5: np.uint8})

        # 基础变量定义
        feature_start = up
        feature_end = down + feature_num
        result_data = split_data(register, 'register_day', 1, feature_end).loc[:, ['user_id', 'register_day']]
        feature_data = split_data(activity, 'act_day', feature_start, feature_end)
        del register
        del activity

        # # # # # # # # #
        # 提特征
        #
        # 活动计数
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='act_day',
                                     aggfunc='count').reset_index().rename(columns={"act_day": 'act_count'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        # 活动的 平均/最大/最小日期 与 注册日期/最大时间 的时间差
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='act_day',
                                     aggfunc='mean').reset_index().rename(columns={"act_day": 'act_mean'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['actmean_red_register'] = result_data['act_mean'] - result_data['register_day']
        result_data['maxday_red_actmean'] = max(result_data['register_day']) - result_data['act_mean']

        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='act_day',
                                     aggfunc=np.max).reset_index().rename(columns={"act_day": 'act_max'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['actmax_red_register'] = result_data['act_max'] - result_data['register_day']
        result_data['maxday_red_actmax'] = max(result_data['register_day']) - result_data['act_max']

        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='act_day',
                                     aggfunc=np.min).reset_index().rename(columns={"act_day": 'act_min'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['actmin_red_register'] = result_data['act_min'] - result_data['register_day']
        result_data['maxday_red_actmin'] = max(result_data['register_day']) - result_data['act_min']
        result_data = result_data.fillna(-1)

        # 最后一天是否有活动
        result_data['act_at_lastday'] = pd.Series(result_data['act_max'] == max(feature_data['act_day'])).replace({True: 1, False: 0})

        # 均值/最大/最小 天数处理
        result_data['act_mean'] = max(feature_data['act_day']) - result_data['act_mean']
        result_data['act_max'] = max(feature_data['act_day']) - result_data['act_max']
        result_data['act_min'] = max(feature_data['act_day']) - result_data['act_min']

        # 观看自己计数
        feature_tmp = pd.pivot_table(feature_data[feature_data['user_id'] == feature_data['author_id']],
                                     index='user_id', values='author_id', aggfunc='count').reset_index().rename(columns={"author_id": 'act_self_count'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        # 活动日期的 方差/峰度/偏度
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='act_day',
                                     aggfunc=np.var).reset_index().rename(columns={"act_day": 'act_var'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='act_day',
                                     aggfunc=pd.Series.kurt).reset_index().rename(columns={"act_day": 'act_kurt'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='act_day',
                                     aggfunc=pd.Series.skew).reset_index().rename(columns={"act_day": 'act_skew'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        # action 的 计数/率
        feature_tmp = feature_data.loc[:, ['user_id', 'action_type', 'act_day']].groupby(['user_id', 'action_type']).count().reset_index().rename(columns={"act_day": 'action_count'})
        for i in range(6):
            fea_name = 'action_' + str(i) + '_count'
            action_tmp = feature_tmp[feature_tmp['action_type'] == i].loc[:, ['user_id', 'action_count']].rename(columns={"action_count": fea_name})
            result_data = pd.merge(result_data, action_tmp, how='left', on='user_id')
        result_data = result_data.fillna(0)
        result_data['action_all'] = (result_data['action_0_count']+result_data['action_1_count']+
                                     result_data['action_2_count']+result_data['action_3_count']+
                                     result_data['action_4_count']+result_data['action_5_count']).replace(0, 1)
        for i in range(6):
            fea_name = 'action_' + str(i) + '_ratio'
            fea_name_2 = 'action_' + str(i) + '_count'
            result_data[fea_name] = result_data[fea_name_2] / result_data['action_all']

        # page 的 计数/率
        feature_tmp = feature_data.loc[:, ['user_id', 'page', 'act_day']].groupby(['user_id', 'page']).count().reset_index().rename(columns={"act_day": 'page_count'})
        for i in range(5):
            fea_name = 'page_' + str(i) + '_count'
            page_tmp = feature_tmp[feature_tmp['page'] == i].loc[:, ['user_id', 'page_count']].rename(columns={"page_count": fea_name})
            result_data = pd.merge(result_data, page_tmp, how='left', on='user_id')
        result_data = result_data.fillna(0)
        result_data['page_all'] = (result_data['page_0_count']+result_data['page_1_count']+
                                   result_data['page_2_count']+result_data['page_3_count']+
                                   result_data['page_4_count']).replace(0, 1)
        for i in range(5):
            fea_name = 'page_' + str(i) + '_ratio'
            fea_name_2 = 'page_' + str(i) + '_count'
            result_data[fea_name] = result_data[fea_name_2] / result_data['page_all']

        del result_data['page_all']
        del result_data['action_all']
        del result_data['register_day']

        # # # # # # # #
        # 保存结果
        result_file_name = 'activity_feature_' + str(feature_num) + '.csv'
        result_data.to_csv(features_path() + result_file_name, index=None)
        print(result_file_name + ' complete!')