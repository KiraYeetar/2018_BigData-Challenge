'''
存放标签数据
划几个窗口存几列标签
'''
launch = pd.read_csv(data_path()+'app_launch_log.txt', sep='\t', header=None,
                  names=['user_id', 'launch_day'],
                  dtype={0: np.uint32, 1: np.uint8})
register = pd.read_csv(data_path() + 'user_register_log.txt', sep='\t', header=None,
                  names=['user_id', 'register_day', 'register_type', 'device_type'],
                  dtype={0: np.uint32, 1: np.uint8, 2: np.uint16, 3: np.uint16})

def get_label_list(start_day, end_day):
    result = split_data(launch, 'launch_day', start_day, end_day)['user_id'].drop_duplicates()
    return pd.Series(result)


if __name__ == '__main__':
    up = downs()+1
    down = downs()+7
    data = register.loc[:, ['user_id']]
    for label_num in range(len(features_addday_list())-1):
        label_list = get_label_list(up + label_num, down + label_num)
        label_name = 'label_' + str(label_num)
        data[label_name] = data['user_id'].isin(label_list).replace({True: 1, False: 0})
    data.to_csv(basic_path()+'data_label.csv', index=None)
    print('data_label.csv complete!')







