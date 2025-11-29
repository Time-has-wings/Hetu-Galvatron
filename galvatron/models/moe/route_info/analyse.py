import json

file_name_list = [
    'rank_0.json',
    'rank_1.json',
    'rank_2.json',
    'rank_3.json',
    'rank_4.json',
    'rank_5.json',
    'rank_6.json',
    'rank_7.json',
]

if __name__ == '__main__':
    # 最终需要得到每层，每个gbsz下，每个专家的token数
    # 以及，每层，每个mbsz下，每个专家的token数

    info = {}
    info_mbsz = {}

    for file_name in file_name_list:
        with open(file_name, 'r') as f:
            data = json.load(f)
            print(f'file_name: {file_name}')
            # print(f'num of routes: {len(data)}')
            # print(f'num of routes with 0: {len([x for x in data if x["num"] == 0])}')
            # 每16组进行求和
            layer_id_list = [0, 1, 2, 3]
            for layer_id in layer_id_list:
                if layer_id not in info.keys():
                    info[layer_id] = {}
                if layer_id not in info_mbsz.keys():    
                    info_mbsz[layer_id] = {}
                layer_id_key = str(layer_id)
                all_list = [value for key, value in data[layer_id_key].items()]

                # 每16组进行求和
                gbsz = -1
                for i in range(0, len(all_list), 16):
                    gbsz += 1
                    sum_value = [sum(values) for values in zip(*all_list[i:i+16])]
                    if gbsz not in info[layer_id].keys():
                        info[layer_id][gbsz] = sum_value
                    else:
                        info[layer_id][gbsz].extend(sum_value)
                
                # 查看mbsz的值
                mbsz = -1
                for i in range(0, len(all_list)):
                    mbsz += 1
                    if mbsz not in info_mbsz[layer_id].keys():
                        info_mbsz[layer_id][mbsz] = all_list[i]
                    else:
                        info_mbsz[layer_id][mbsz].extend(all_list[i])
    # print(info)
    # print(info)

    for layer_id in info.keys():
        for gbsz in info[layer_id].keys():
            print(f'layer_id: {layer_id}, gbsz: {gbsz}, sum_value: {info[layer_id][gbsz]}')

    for layer_id in info_mbsz.keys():
        for mbsz in info_mbsz[layer_id].keys():
            print(f'layer_id: {layer_id}, mbsz: {mbsz}, all_value: {info_mbsz[layer_id][mbsz]}')