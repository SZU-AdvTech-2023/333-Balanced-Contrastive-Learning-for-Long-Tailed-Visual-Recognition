from scipy.io import loadmat


# 从MATLAB文件读取CSI数据
csi_data = loadmat('./data/maml_direction/user.mat')['csi_data']


def query_csi(activities_list, query_labels, dataset_type='gait'):
    query_item_list = []
    query_label_list = []

    for index, item in enumerate(activities_list):
        # query the specific labels, ?: all
        labels_length = len(query_labels)
        process_flag = 1
        if dataset_type == 'gait':
            # userid, trkid, rptid, rxid, timeid
            labels_item = extract_label(item)
        elif dataset_type == 'activity':
            # userid, gesture_type, torso_location, face_orientation, repetition, rxid
            labels_item = extract_label_activity(item)

        for i in range(labels_length):
            if (query_labels[i] == '?') or (labels_item[i] in query_labels[i]):
                process_flag *= 1
            else:
                process_flag *= 0
        # pass the file if not met the query label
        if process_flag == 0:
            continue
        else:
            activities_sample = item

        query_item_list.append(activities_sample)
        query_label_list.append(labels_item)


    myquery_item_list = []
    myquery_label_list = []
    for i in range(0, 51):
        for index, item in enumerate(query_label_list):
            # labels_item = extract_label(item)
            if item[2]==i:
                activities_sample = item
                myquery_item_list.append(query_item_list[index])
    # print(myquery_item_list)


    return myquery_item_list, query_label_list

