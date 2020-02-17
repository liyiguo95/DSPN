import numpy as np
from get_data import *

def get_n():
    n_adgroup = len(adgroup_dict)
    n_member = len(member_dict)
    n_campaign = len(campaign_dict)
    n_item = len(item_dict)
    n_cate = len(cate_dict)
    n_commodity = len(commodity_dict)
    n_node = len(node_dict)
    n_effect = len(effect_dict)
    n_pos = len(pos_dict)
    n_direct = len(direct_dict)
    n_direct_action = 3 * n_direct
    n_pos_action = 3 * n_pos
    return n_adgroup, n_member, n_campaign, \
           n_item, n_cate, n_commodity, \
           n_node, n_effect, n_pos, n_direct, \
           n_direct_action, n_pos_action


def prepare_data(inp):
    

    batch = len(inp)
    # label
    ylabel = np.array(np.zeros((batch, 2), dtype=np.int32))
    y_ = np.array(np.zeros((batch, 2), dtype=np.int32))
    for i, x in enumerate(inp):
        ylabel[i][1 - x[0]] = 1
        y_[i][1 - x[0]] = 1
    
    # ad feature
    adgroup = [x[2] for x in inp]
    adgroup = np.array(adgroup)
    member = [x[3] for x in inp]
    member = np.array(member)
    campaign = [x[4] for x in inp]
    campaign = np.array(campaign)
    item = [x[5] for x in inp]
    item = np.array(item)
    item_price = [x[6] for x in inp]
    item_price = np.array(item_price)
    cate = [x[7] for x in inp]
    cate = np.array(cate)
    commodity = [x[8] for x in inp]
    commodity = np.array(commodity)
    node = [x[9] for x in inp]
    node = np.array(node)
    
    # effect data
    maxday = max([len(x[10]) for x in inp]) # effect day is reference
    
    effect_num = len(effect_dict)
    effect = np.array(np.zeros((batch, maxday, effect_num)))
    effect_mask = np.array(np.zeros((batch, maxday)))
    effect_id = [[[i for i in range(effect_num)] for j in range(maxday)] for k in range(batch)]
    effect_id = np.array(effect_id)
    for i, y in enumerate(inp):
        x = y[10]
        x = np.array(x)
        effect[i, :x.shape[0], :x.shape[1]] = x
        effect_mask[i, :x.shape[0]] = 1.0
        
    # pos_ratio
    pos_num = len(pos_dict)
    pos = np.array(np.zeros((batch, maxday, pos_num)))
    pos_mask = np.array(np.zeros((batch, maxday)))
    pos_id = [[[i for i in range(pos_num)] for j in range(maxday)] for k in range(batch)]
    pos_id = np.array(pos_id)
    for i, y in enumerate(inp):
        x = y[11]
        x = np.array(x)
        pos[i, :x.shape[0], :x.shape[1]] = x
        pos_mask[i, :x.shape[0]] = 1.0
    
    # direct info
    # direct
    direct_num = len(direct_dict)
    direct_id = [[[i for i in range(direct_num)] for j in range(maxday)] for k in range(batch)]
    direct_id = np.array(direct_id)
    direct = np.array(np.zeros((batch, maxday, direct_num)))
    for i, y in enumerate(inp):
        x = y[12]
        x = np.array(x)
        direct[i, :x.shape[0], :x.shape[1]] = x

    # direct mask
    direct_mask = np.array(np.zeros((batch, maxday, direct_num)))
    for i, y in enumerate(inp):
        x = y[13]
        x = np.array(x)
        direct_mask[i, :x.shape[0], :x.shape[1]] = x # x is mask here, it's right

    # actions: no cut off
    # direct
    direct_max_len = max(1, max([max([len(y) for y in x[14]]) for x in inp]))
    direct_action_type = np.array(np.zeros((batch, maxday, direct_max_len), dtype=np.int32))
    direct_action_value = np.array(np.zeros((batch, maxday, direct_max_len)))
    direct_action_mask = np.array(np.zeros((batch, maxday, direct_max_len)))
    for i, z in enumerate(inp):
        y = z[14]
        for j, x in enumerate(y):
            x = np.array(x)
            direct_action_type[i, j, :x.shape[0]] = x
            direct_action_mask[i, j, :x.shape[0]] = 1.0
        y = z[15]
        for j, x in enumerate(y):
            x = np.array(x)
            direct_action_value[i, j, :x.shape[0]] = x
        
    # pos
    pos_max_len = max(1, max([max([len(y) for y in x[16]]) for x in inp]))
    pos_action_type = np.array(np.zeros((batch, maxday, pos_max_len), dtype=np.int32))
    pos_action_value = np.array(np.zeros((batch, maxday, pos_max_len)))
    pos_action_mask = np.array(np.zeros((batch, maxday, pos_max_len)))
    for i, z in enumerate(inp):
        y = z[16]
        for j, x in enumerate(y):
            x = np.array(x)
            pos_action_type[i, j, :x.shape[0]] = x
            pos_action_mask[i, j, :x.shape[0]] = 1.0
        y = z[17]
        for j, x in enumerate(y):
            x = np.array(x)
            pos_action_value[i, j, :x.shape[0]] = x

    # print("adgroup:")
    # print(type(adgroup[0]))
    # print(adgroup)
    # print("member:")
    # print(type(member[0]))
    # print(member)
    # print("campaign:")
    # print(type(campaign[0]))
    # print(campaign)
    # print("item:")
    # print(type(item[0]))
    # print(item)
    # print("item_price:")
    # print(type(item_price[0]))
    # print(item_price)
    # print("cate:")
    # print(type(cate[0]))
    # print(cate)
    # print("commodity:")
    # print(type(commodity[0]))
    # print(commodity)
    # print("node:")
    # print(type(node[0]))
    # print(node)
    # print("effect:")
    # print(type(effect[0][0][0]))
    # print(effect)
    # print("effect_mask:")
    # print(type(effect_mask[0][0][0]))
    # print(effect_mask)
    # print("pos:")
    # print(type(pos[0][0][0]))
    # print(pos)
    # print("pos_mask:")
    # print(type(pos_mask[0][0][0]))
    # print(pos_mask)
    # print("direct:")
    # print(type(direct[0][0][0]))
    # print(direct)
    # print("direct_mask:")
    # print(type(direct_mask[0][0][0]))
    # print(direct_mask)
    # print("direct_action_type:")
    # print(type(direct_action_type[0][0][0]))
    # print(direct_action_type)
    # print("direct_action_value:")
    # print(type(direct_action_value[0][0][0]))
    # print(direct_action_value)
    # print("direct_action_mask:")
    # print(direct_action_mask.shape)
    # print(type(direct_action_mask[0][0][0]))
    # print(direct_action_mask)
    # print("pos_action_type:")
    # print(type(pos_action_type[0][0][0]))
    # print(pos_action_type)
    # print("pos_action_value:")
    # print(type(pos_action_value[0][0][0]))
    # print(pos_action_value)
    # print("pos_action_mask:")
    # print(pos_action_mask.shape)
    # print(type(pos_action_mask[0][0][0]))
    # print(pos_action_mask)
    return ylabel, y_, adgroup, member, campaign, item, item_price, cate, commodity, node, \
           effect_id, effect, effect_mask, \
           pos_id, pos, pos_mask, \
           direct_id, direct, direct_mask, \
           direct_action_type, direct_action_value, direct_action_mask, \
           pos_action_type, pos_action_value, pos_action_mask