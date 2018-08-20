'''
注册表特征
'''

if __name__ == '__main__':
    up = ups()
    down = downs()

    for feature_num in features_addday_list():
        # 读数据
        register = pd.read_csv(data_path()+'user_register_log.txt', sep='\t', header=None,
                   names=['user_id','register_day','register_type','device_type'],
                   dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint16})

        # 基础变量定义
        feature_start = up + 0
        feature_end = down + feature_num

        '''
        result_data 是存放特征的结果文件
        feature_data 用于存放被提取的原文件
        *****_tmp  存放临时特征文件
        类似文件后续不再注释
        '''
        result_data = split_data(register, 'register_day', 1, feature_end)
        feature_data = split_data(register, 'register_day', feature_start, feature_end)
        del register

        # # # # # # # # #
        # 提特征(已经包含设备类型、设备类型)
        # 
        # 特征区间最大天数减去注册日期
        result_data['maxday_red_registerday'] = max(feature_data['register_day']) - feature_data['register_day']
        result_data = result_data.fillna(max(feature_data['register_day']))

        del result_data['register_day']

        # # # # # # # # #
        # 保存结果
        result_file_name = 'register_feature_'+str(feature_num)+'.csv'
        result_data.to_csv(features_path()+result_file_name, index=None)
        print(result_file_name+' complete!')

