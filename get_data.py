import pandas as pd
import numpy as np
from data_iterator import dataIterator

def get_id(id_dict, val):
    if val not in id_dict:
        return 0
    return id_dict[val]

def get_date_interval(days_info):
    days_info = days_info.split(';')
    day_list = []
    for x in days_info:
        x, _ = x.split(':')
        day_list.append(x)
    day_list.sort()
    ret = {}
    for x in day_list:
        if x not in ret:
            ret[x] = len(ret)
    return ret

# dict list
effect_dict = {}
adgroup_dict = {}
pos_dict = {} # 资源位字典
direct_dict = {}
member_dict = {}
campaign_dict = {}
item_dict = {}
cate_dict = {}
commodity_dict = {}
node_dict = {}

def getData(
        file='ad_action_state',
        batch_size=128,
        shuffle_each_epoch=True,
    ):
    df = pd.read_csv(file, sep='$')

    df['adgroup_id'] = pd.Series(df['adgroup_id'], dtype=np.str)

    # build dict
    for _, row in df.iterrows():
        adgroup_id = row['adgroup_id']
        if adgroup_id not in adgroup_dict:
            adgroup_dict[adgroup_id] = len(adgroup_dict)

        effect_info = row['effect_data'].split(';')
        for s in effect_info:
            _, s = s.split(':')
            s = s.split(',')
            for t in s:
                t, _ = t.split('=')
                if t not in effect_dict:
                    effect_dict[t] = len(effect_dict)
            
        pos_info = row['pos_ratio'].split(';')
        for s in pos_info:
            _, s = s.split(':')
            s = s.split(',')
            for t in s:
                t, _ = t.split('=')
                if t not in pos_dict:
                    pos_dict[t] = len(pos_dict)
                    
        direct_info = row['direct_type_price'].split(';')
        for s in direct_info:
            _, s = s.split(':')
            s = s.split(',')
            for t in s:
                t, _ = t.split('=')
                if t not in direct_dict:
                    direct_dict[t] = len(direct_dict)
                    
        ad_feature = row['ad_feature'].split(';')
        for s in ad_feature:
            _, s = s.split(':')
            s = s.split(',')
            for t in s:
                name, num = t.split('=')
                if name == 'member_id':
                    if num not in member_dict:
                        member_dict[num] = len(member_dict)
                elif name == 'campaign_id':
                    if num not in campaign_dict:
                        campaign_dict[num] = len(campaign_dict)
                elif name == 'adgroup_id':
                    if num not in adgroup_dict:
                        adgroup_dict[num] = len(adgroup_dict)
                elif name == 'item_id':
                    if num not in item_dict:
                        item_dict[num] = len(item_dict)
                elif name == 'cate_id':
                    if num not in cate_dict:
                        cate_dict[num] = len(cate_dict)
                elif name == 'commodity_id':
                    if num not in commodity_dict:
                        commodity_dict[num] = len(commodity_dict)
                elif name == 'node_id':
                    if num not in node_dict:
                        node_dict[num] = len(node_dict)

    train_set = []
    test_set = []

    # parse data
    for _, row in df.iterrows():
        data = []
        
        # label
        data.append(row['label'])
        data.append(row['label'])
        
        # ad feature
        adgroup_id = get_id(adgroup_dict, row['adgroup_id'])
        data.append(adgroup_id)
        
        ad_feature = (row['ad_feature'].split(';')[0]).split(':')[1]
        ad_feature = ad_feature.split(',')
        for x in ad_feature:
            name, entry = x.split('=')
            fid = 0
            if name == 'member_id':
                fid = get_id(member_dict, entry)
                data.append(fid)
            elif name == 'campaign_id':
                fid = get_id(campaign_dict, entry)
                data.append(fid)
            elif name == 'item_id':
                fid = get_id(item_dict, entry)
                data.append(fid)
            elif name == 'item_price':
                item_price = float(entry) / 100.0
                data.append(item_price)
            elif name == 'cate_id':
                fid = get_id(cate_dict, entry)
                data.append(fid)
            elif name == 'commodity_id':
                fid = get_id(commodity_dict, entry)
                data.append(fid)
            elif name == 'node_id':
                fid = get_id(node_dict, entry)
                data.append(fid)
        
        days_dict = get_date_interval(row['effect_data'])
        days_num = len(days_dict)
        
        # effect data
        effect_list = [[0.0] * len(effect_dict) for _ in range(days_num)]
        effect_data = row['effect_data'].split(';')
        mmin = np.array([1000000000.0 for i in range(len(effect_dict))])
        mmax = np.array([0.0 for i in range(len(effect_dict))])
        for x in effect_data:
            day, entry = x.split(':')
            if day not in days_dict:
                continue
            day = days_dict[day]
            entry = entry.split(',')
            for o, y in enumerate(entry):
                name, num = y.split('=')
                num = float(num)
                name = get_id(effect_dict, name)
                mmin[o] = min(mmin[o], num)
                mmax[o] = max(mmax[o], num)
                effect_list[day][name] = num

        # normalized
        for o, x in enumerate(effect_list):
            effect_arr = np.array(effect_list[o])
            effect_arr = (effect_arr - mmin) / np.array([max(mmax[i] - mmin[i], 0.0000000001) for i in range(len(mmax))])
            effect_list[o] = effect_arr.tolist()
        data.append(effect_list)
        
        # pos_ratio
        pos_list = [[0.0] * len(pos_dict) for _ in range(days_num)]
        pos_data = row['pos_ratio'].split(';')
        for x in pos_data:
            day, entry = x.split(':')
            if day not in days_dict:
                continue
            day = days_dict[day]
            entry = entry.split(',')
            for y in entry:
                name, num = y.split('=')
                num = float(num)
                name = get_id(pos_dict, name)
                pos_list[day][name] = num
        data.append(pos_list)
        
        # direct info
        direct_list = [[0.0] * len(direct_dict) for _ in range(days_num)]
        direct_mask = [[0.0] * len(direct_dict) for _ in range(days_num)]
        direct_data = row['direct_type_price'].split(';')
        for x in direct_data:
            day, entry = x.split(':')
            if day not in days_dict:
                continue
            day = days_dict[day]
            entry = entry.split(',')
            for y in entry:
                name, num = y.split('=')
                num = float(num)
                name = get_id(direct_dict, name)
                direct_list[day][name] = num
                direct_mask[day][name] = 1.0
        data.append(direct_list)
        data.append(direct_mask)
        
        # sorted by time, haven't consider specific time
        # actions info
        direct_type_list = [[] for _ in range(days_num)]
        direct_val_list = [[] for _ in range(days_num)]
        pos_type_list = [[] for _ in range(days_num)]
        pos_val_list = [[] for _ in range(days_num)]
        actions_data = row['actions'].split(';')
        for x in actions_data:
            day, entry = x.split(':')
            if day not in days_dict:
                continue
            day = days_dict[day]
            entry = entry.split(',')
            for y in entry:
                if len(y) == 0:
                    continue
                a, b = y.split('-', 1)
                if a == '修改定向':
                    a, b = b.split('->')
                    aa = a.split('-')[0] # direct type
                    bb = a.split('-')[-1]
                    cc = b.split('-')[0]
                    bb = float(bb) / 100.0 # old price
                    cc = float(cc) / 100.0 # new price
                    direct_type_list[day].append(get_id(direct_dict, aa))
                    direct_val_list[day].append(cc - bb)
                if a == '新增定向':
                    b = b.split('-')
                    aa = b[0]
                    if aa == '67' or aa == '66' or aa == '32' or aa == '16384':
                        bb = b[3]
                    else:
                        bb = b[2]
                    bb = float(bb) / 100.0
                    direct_type_list[day].append(len(direct_dict) + get_id(direct_dict, aa))
                    direct_val_list[day].append(bb)
                #if a == '移除定向':
                #    b = b.split('-')
                #    aa = b[0]
                #    bb = b[2]
                #    if aa == '67' or aa == '66' or aa == '32' or aa == '16384':
                #        bb = b[3]
                #    else:
                #         bb = b[2]
                #    bb = float(bb) / 100.0
                #    direct_type_list[day].append(len(direct_dict) + len(direct_dict) + get_id(direct_dict, aa))
                #    direct_val_list[day].append(bb)
                if a == '新增资源位':
                    b = b.split('-')
                    aa = b[0]
                    bb = b[2]
                    bb = float(bb) / 100.0
                    if aa == '23':
                        pos_type_list[day].append(0)
                        pos_val_list[day].append(bb)
                    if aa == '24':
                        pos_type_list[day].append(1)
                        pos_val_list[day].append(bb)
                    if aa == '25':
                        pos_type_list[day].append(2)
                        pos_val_list[day].append(bb)
                if a == '修改资源位':
                    a, b = b.split('->')
                    aa = a.split('-')[0]
                    bb = a.split('-')[-1]
                    cc = b.split('-')[0]
                    bb = float(bb) / 100.0
                    cc = float(cc) / 100.0
                    if aa == '23':
                        pos_type_list[day].append(3)
                        pos_val_list[day].append(cc - bb)
                    if aa == '24':
                        pos_type_list[day].append(4)
                        pos_val_list[day].append(cc - bb)
                    if aa == '25':
                        pos_type_list[day].append(5)
                        pos_val_list[day].append(cc - bb)
                #if a == '移除资源位':
                #    b = b.split('-')
                #    aa = b[0]
                #    bb = b[2]
                #    bb = float(bb) / 100.0
                #    if aa == '23':
                #        pos_type_list[day].append(6)
                #        pos_val_list[day].append(bb)
                #    if aa == '24':
                #        pos_type_list[day].append(7)
                #        pos_val_list[day].append(bb)
                #    if aa == '25':
                #        pos_type_list[day].append(8)
                #        pos_val_list[day].append(bb)
        data.append(direct_type_list)
        data.append(direct_val_list)
        data.append(pos_type_list)
        data.append(pos_val_list)
        
        data_type = row['version']
        if data_type == 'train':
            train_set.append(data)
        elif data_type == 'test':
            test_set.append(data)
    print("train num: %d." % len(train_set))
    print("test num: %d." % len(test_set))
    train_data = dataIterator(train_set, batch_size=batch_size, shuffle_each_epoch=shuffle_each_epoch)
    test_data = dataIterator(test_set, batch_size=batch_size, shuffle_each_epoch=False)
    return train_data, test_data