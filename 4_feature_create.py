'''
视频创建特征
'''

if __name__ == '__main__':
    up = ups()
    down = downs()

    for feature_num in features_addday_list():
        # 读数据
        register = pd.read_csv(data_path() + 'user_register_log.txt', sep='\t', header=None,
                               names=['user_id', 'register_day', 'register_type', 'device_type'],
                               dtype={0: np.uint32, 1: np.uint8, 2: np.uint16, 3: np.uint16})
        create = pd.read_csv(data_path() + 'video_create_log.txt', sep='\t', header=None,
                             names=['user_id', 'create_day'],
                             dtype={0: np.uint32, 1: np.uint8})

        # 基础变量定义
        feature_start = up
        feature_end = down + feature_num
        result_data = split_data(register, 'register_day', 1, feature_end).loc[:, ['user_id', 'register_day']]
        feature_data = split_data(create, 'create_day', feature_start, feature_end)
        del register
        del create

        # # # # # # # # #
        # 提特征
        #
        # 用户创建视频计数
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='create_day',
                                     aggfunc='count').reset_index().rename(columns={"create_day": 'create_count'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        # 用户创建视频的 平均/最大/最小日期 与 注册日期/最大时间 的时间差
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='create_day',
                                     aggfunc='mean').reset_index().rename(columns={"create_day": 'create_mean'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['createmean_red_register'] = result_data['create_mean'] - result_data['register_day']
        result_data['maxday_red_createmean'] = max(result_data['register_day']) - result_data['create_mean']

        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='create_day',
                                     aggfunc=np.max).reset_index().rename(columns={"create_day": 'create_max'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['createmax_red_register'] = result_data['create_max'] - result_data['register_day']
        result_data['maxday_red_createmax'] = max(result_data['register_day']) - result_data['create_max']

        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='create_day',
                                     aggfunc=np.min).reset_index().rename(columns={"create_day": 'create_min'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['createmin_red_register'] = result_data['create_min'] - result_data['register_day']
        result_data['maxday_red_createmin'] = max(result_data['register_day']) - result_data['create_min']
        result_data = result_data.fillna(-1)

        # 创建最大间隔
        result_data['max_red_min_create'] = result_data['create_max'] - result_data['create_min']

        # 最后一天是否有活动
        result_data['create_at_lastday'] = pd.Series(
            result_data['create_max'] == max(feature_data['create_day'])).replace({True: 1, False: 0})

        # 均值/最大/最小 天数处理
        result_data['create_mean'] = max(feature_data['create_day']) - result_data['create_mean']
        result_data['create_max'] = max(feature_data['create_day']) - result_data['create_max']
        result_data['create_min'] = max(feature_data['create_day']) - result_data['create_min']

        # 间隔的 方差/均值
        feature_data_tmp = feature_data.drop_duplicates(['user_id', 'create_day']).sort_values(
            by=['user_id', 'create_day'])
        feature_data_tmp['create_gap'] = np.array(feature_data_tmp['create_day']) - np.array(
            feature_data_tmp.tail(1).append(feature_data_tmp.head(len(feature_data_tmp) - 1))['create_day'])

        feature_tmp = pd.pivot_table(feature_data_tmp, index='user_id', values='create_gap',
                                     aggfunc=(lambda a: np.average(a[1:]))).reset_index().rename(
            columns={"create_gap": 'create_gap_mean'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data_tmp, index='user_id', values='create_gap',
                                     aggfunc=(lambda a: np.var(a[1:]))).reset_index().rename(
            columns={"create_gap": 'create_gap_var'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        # 是否一直连续/连续到结束
        result_data['always_create'] = [1 if i == 1 else 0 for i in result_data['create_gap_mean']]
        tmp = (result_data['create_at_lastday'] == 1).replace({True: 1, False: 0})
        result_data['always_create_atlast'] = tmp * result_data['always_create']
        del tmp

        # 创建日期的 方差/峰度/偏度
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='create_day',
                                     aggfunc=np.var).reset_index().rename(columns={"create_day": 'create_var'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='create_day',
                                     aggfunc=pd.Series.kurt).reset_index().rename(columns={"create_day": 'create_kurt'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='create_day',
                                     aggfunc=pd.Series.skew).reset_index().rename(columns={"create_day": 'create_skew'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        # 求一天最大创建数
        feature_data['max_create_in_oneday'] = 0
        feature_tmp = pd.pivot_table(feature_data, index=['user_id', 'create_day'], values='max_create_in_oneday',
                                     aggfunc='count').reset_index()
        feature_tmp = pd.DataFrame(feature_tmp.groupby(['user_id'])['max_create_in_oneday'].max()).reset_index()
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data.fillna(0, inplace=True)

        del result_data['register_day']

        # # # # # # # #
        # 保存结果
        result_file_name = 'create_feature_' + str(feature_num) + '.csv'
        result_data.to_csv(features_path() + result_file_name, index=None)
        print(result_file_name + ' complete!')