'''
登录表特征
'''

if __name__ == '__main__':
    up = ups()
    down = downs()

    for feature_num in features_addday_list():
        # 读数据
        register = pd.read_csv(data_path()+'user_register_log.txt', sep='\t', header=None,
                    names=['user_id','register_day','register_type','device_type'],
                    dtype={0: np.uint32, 1: np.uint8, 2: np.uint16, 3: np.uint16})
        launch = pd.read_csv(data_path() + 'app_launch_log.txt', sep='\t', header=None,
                    names=['user_id', 'launch_day'],
                    dtype={0: np.uint32, 1: np.uint8})

        # 基础变量定义
        feature_start = up
        feature_end = down + feature_num
        result_data = split_data(register, 'register_day', 1, feature_end).loc[:, ['user_id', 'register_day']]
        feature_data = split_data(launch, 'launch_day', feature_start, feature_end)
        del register
        del launch

        # # # # # # # # #
        # 提特征
        #
        # 登录计数/登录率
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='launch_day', 
                                     aggfunc='count').reset_index().rename(columns={"launch_day": 'launch_count'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        distance = (max(feature_data['launch_day']) - min(feature_data['launch_day']))
        result_data['launch_ratio'] = result_data['launch_count'] * 1.0 / distance
        result_data = result_data.fillna(0)

        # 登录的 平均/最大/最小日期 与 注册日期/最大时间 的时间差
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='launch_day', 
                                     aggfunc='mean').reset_index().rename(columns={"launch_day": 'launch_mean'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['launchmean_red_register'] = result_data['launch_mean'] - result_data['register_day']
        result_data['maxday_red_launchmean'] = max(result_data['register_day']) - result_data['launch_mean']

        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='launch_day', 
                                     aggfunc=np.max).reset_index().rename(columns={"launch_day": 'launch_max'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data['launchmax_red_register'] = result_data['launch_max'] - result_data['register_day']
        result_data['maxday_red_launchmax'] = max(result_data['register_day']) - result_data['launch_max']

        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='launch_day',
                                     aggfunc=np.min).reset_index().rename(columns={"launch_day": 'launch_min'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        # result_data['launchmin_red_register'] = result_data['launch_min'] - result_data['register_day']
        result_data['maxday_red_launchmin'] = max(result_data['register_day']) - result_data['launch_min']
        result_data = result_data.fillna(-1)

        # 登录最大与最小差
        result_data['max_red_min_launch'] = result_data['launch_max'] - result_data['launch_min']

        # 最后一天是否有活动
        result_data['launch_at_lastday'] = pd.Series(result_data['launch_max'] == max(feature_data['launch_day'])).replace({True: 1, False: 0})

        # 均值/最大/最小 天数处理
        result_data['launch_mean'] = max(feature_data['launch_day']) - result_data['launch_mean']
        result_data['launch_max'] = max(feature_data['launch_day']) - result_data['launch_max']
        result_data['launch_min'] = max(feature_data['launch_day']) - result_data['launch_min']

        # 间隔的 方差/均值/最大
        feature_data_tmp = feature_data.drop_duplicates(['user_id', 'launch_day']).sort_values(by=['user_id', 'launch_day'])
        feature_data_tmp['launch_gap'] = np.array(feature_data_tmp['launch_day']) - np.array(
            feature_data_tmp.tail(1).append(feature_data_tmp.head(len(feature_data_tmp) - 1))['launch_day'])

        feature_tmp = pd.pivot_table(feature_data_tmp, index='user_id', values='launch_gap',
                                     aggfunc=(lambda a: np.average(a[1:]))).reset_index().rename(columns={"launch_gap": 'launch_gap_mean'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data_tmp, index='user_id', values='launch_gap',
                                     aggfunc=(lambda a: np.var(a[1:]))).reset_index().rename(columns={"launch_gap": 'launch_gap_var'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data_tmp, index='user_id', values='launch_gap',
                                     aggfunc=(lambda a: np.max(a[1:]))).reset_index().rename(columns={"launch_gap": 'launch_gap_max'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        # 是否一直连续/连续到结束
        result_data['always_launch'] = [1 if i == 1 else 0 for i in result_data['launch_gap_mean']]
        tmp = (result_data['launch_at_lastday'] == 1).replace({True: 1, False: 0})
        result_data['always_launch_atlast'] = tmp * result_data['always_launch']
        del tmp

        # 登录日期的 方差/峰度/偏度
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='launch_day',
                                     aggfunc=np.var).reset_index().rename(columns={"launch_day": 'launch_var'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='launch_day',
                                     aggfunc=pd.Series.kurt).reset_index().rename(columns={"launch_day": 'launch_kurt'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        feature_tmp = pd.pivot_table(feature_data, index='user_id', values='launch_day',
                                     aggfunc=pd.Series.skew).reset_index().rename(columns={"launch_day": 'launch_skew'})
        result_data = pd.merge(result_data, feature_tmp, on='user_id', how='left')
        result_data = result_data.fillna(0)

        del result_data['register_day']
        
        # # # # # # # #
        # 保存结果
        result_file_name = 'launch_feature_' + str(feature_num) + '.csv'
        result_data.to_csv(features_path() + result_file_name, index=None)
        print(result_file_name + ' complete!')