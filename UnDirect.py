import networkx as nx
import csv
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import itertools


# 分词函数
def participle(sentence):
    # 进行分词
    tmp_seg_list = word_tokenize(sentence)
    # 去除标点和停用词
    sr = stopwords.words('english')
    punctuate = [',', '.', '!', ';', '?', '，', '。', ' ', '"', '“', '”']
    no_s = []
    for w in tmp_seg_list:
        if w not in sr and w not in punctuate:
            no_s.append(w)
    return no_s


# 获取根据数值的差距的确定跳转概率（差距越小概率越大）
def get_value_prob(node, alt_arr, attr, value_dict):
    min_diff = 0
    if attr == 'year':
        min_diff = 3
    else:
        min_diff = 4
    tmp_value = value_dict[node][attr]
    diff_arr = []
    # 计算得出数值相差列表
    for paper in alt_arr:
        tmp_diff = abs(value_dict[paper][attr] - tmp_value)
        diff_arr.append(tmp_diff)
    if min(diff_arr) > min_diff:
        return False
    weighted_list = []
    sum_diff = sum(diff_arr)
    for t in diff_arr:
        if t == 0:
            weighted_list.append(sum_diff / 0.1)
        else:
            weighted_list.append(sum_diff / t)
    # 如果没有差异则随机
    if sum(weighted_list) == 0:
        return [1/len(weighted_list) for i in range(len(weighted_list))]
    # 将权重进行归一化转换为概率输出
    prob_list = []
    sum_w = sum(weighted_list)
    for w in weighted_list:
        prob_list.append(w / sum_w)
    return prob_list


# 根据分词相似性确定跳转概率(与父节点的相同分词越多跳转概率越大，分词函数可以自己进行修改和优化)
def get_seg_prob(node, alt_arr, attr, value_dict):
    node_seg = value_dict[node][attr]
    node_seg_len = len(node_seg)
    # 如果当前节点没有分词
    if node_seg_len == 0:
        return False
    similar_percent = []
    for suc_id in alt_arr:
        tmp_seg = value_dict[suc_id][attr]
        tmp_count = 0
        for w in tmp_seg:
            if w in node_seg:
                tmp_count += 1
        similar_percent.append(tmp_count / node_seg_len)
    sum_w = sum(similar_percent)
    # 如果当前权重都为0，或者最大相似性小于30% 则中断游走
    if sum_w == 0 or max(similar_percent) < 0.3:
        return False
    prob_list = []
    for w in similar_percent:
        prob_list.append(w/sum_w)
    return prob_list


# 获取会议类型的跳转概率（拥有相同会议类型的节点给予80%的跳转概率，其他类型的节点均分20%的跳转概率）
def get_conf_prob(node, alt_arr):
    node_conf = info_dict[node]['conf']
    conf_dict = {}  # 用来计算每种会议类型各有几篇论文
    conf_type = []  # 用来判断子节点有几种类型
    prob_list = []  # 输出的跳转概率数组
    suc_conf_arr = [info_dict[suc_id]['conf'] for suc_id in alt_arr]
    # 判断子节点没有与当前节点相同的
    if node_conf not in suc_conf_arr:
        return False
    else:
        for suc_id in alt_arr:
            suc_conf = info_dict[suc_id]['conf']
            if suc_conf not in conf_dict:
                conf_type.append(suc_conf)
                conf_dict[suc_conf] = [suc_id]
            else:
                conf_dict[suc_conf].append(suc_id)
        for tmp_conf in suc_conf_arr:
            if tmp_conf == node_conf:
                prob_list.append(8 / len(conf_dict[tmp_conf]))
            else:
                prob_list.append((3 - len(conf_type)) / len(conf_dict[tmp_conf]))
        return prob_list


# 加权随机选择
def random_p(weight_list, alt_arr):
    all_zero = True
    for w in weight_list:
        if w != 0:
            all_zero = False
    if all_zero is False:
        tmp_arr = [[alt_arr[i], weight_list[i]] for i in range(len(alt_arr))]
        sorted_list = sorted(tmp_arr, key=lambda item: item[1])
        sum_w = 0
        for t in sorted_list:
            sum_w += t[1]
        tmp_num = random.random() * sum_w
        tmp_sum = 0
        for t in sorted_list:
            tmp_sum += t[1]
            if tmp_sum >= tmp_num:
                return t[0]
    else:
        return random.choice(alt_arr)


# 根据属性值确定下一节点
def get_next_node(value_arr, alt_arr):
    if len(value_arr) == 0:
        return random.choice(alt_arr)
    else:
        weight_list = [0 for _ in range(len(value_arr[0]))]
        attr_types = len(value_arr)
        for tmp_arr in value_arr:
            for i in range(len(tmp_arr)):
                weight_list[i] += tmp_arr[i] / attr_types
        next_node = random_p(weight_list, alt_arr)
        return next_node


# 根据所给的节点和参数选择，进行概率游走，返回一条路径

"""step 用来决定是否限制步数（默认不限制，若不限制时，游走路径内不能出现重复的点）
   ret 用来决定路径内是否可以出现重复的点"""


def walk_by_multi(g, node, value_dict, steps=None, ret=False, conf=False, aff=False, abt=False, year=False, cited=False):
    path = [node]
    next_node = node
    alt_arr = [suc_id for suc_id in g.successors(node)]
    for pre_id in g.predecessors(node):
        alt_arr.append(pre_id)
    tmp_steps = 1  # 记录游走步数
    while len(alt_arr) > 0:
        prob_arr = []
        """若概率数组返回False则代表备选集内没有较好的子节点，则跳出"""
        if conf is True:
            tmp_pro = get_conf_prob(next_node, alt_arr)
            if tmp_pro is not False:
                prob_arr.append(tmp_pro)
            else:
                break
        if abt is True:
            tmp_pro = get_seg_prob(next_node, alt_arr, attr='abt', value_dict=value_dict)
            if tmp_pro is not False:
                prob_arr.append(tmp_pro)
            else:
                break
        if aff is True:
            tmp_pro = get_seg_prob(next_node, alt_arr, attr='aff', value_dict=value_dict)
            if tmp_pro is not False:
                prob_arr.append(tmp_pro)
            else:
                break
        if year is True:
            tmp_pro = get_value_prob(next_node, alt_arr, attr='year', value_dict=value_dict)
            if tmp_pro is not False:
                prob_arr.append(tmp_pro)
            else:
                break
        if cited is True:
            tmp_pro = get_value_prob(next_node, alt_arr, attr='cited', value_dict=value_dict)
            if tmp_pro is not False:
                prob_arr.append(tmp_pro)
            else:
                break
        next_node = get_next_node(prob_arr, alt_arr)
        if next_node in path:
            break
        if steps is not None:  # 如果限制了步数,则判断是否可以回头和步数是否超标来决定跳出与否
            if ret is False:
                if next_node in path:
                    break
            else:
                if tmp_steps >= steps:
                    break
        else:
            if next_node in path:
                break
        tmp_steps += 1  # 成功选择下一节点则步数加一
        path.append(next_node)
        alt_arr = [suc_id for suc_id in g.successors(next_node)]
        for pre_id in g.predecessors(node):
            alt_arr.append(pre_id)
    return path


# 所有节点都作为起始点进行游走
def walk_all(g, all_node, value_dict, all_path, steps=None, ret=False, conf=True, aff=False, abt=False, year=False, cited=False):
    for node in all_node:
        tmp_path = walk_by_multi(g, node, value_dict, steps, ret, conf, aff, abt, year, cited)
        if len(tmp_path) > 1:
            all_path.append(tmp_path)
    return all_path


# 进行多次概率游走
def walk_times(times, g, all_node, value_dict, steps=None, ret=False, conf=True, aff=False, abt=False, year=False, cited=False):
    all_path = []
    for _ in range(times):
        all_path = walk_all(g, all_node, value_dict, all_path, steps=steps, ret=ret, conf=conf, aff=aff, abt=abt, year=year, cited=cited)
    return all_path


def write_path(file_name, out_data):
    of = open(file_name, 'w', newline='')
    csv.writer(of).writerows(out_data)
    of.close()




f_info = open('data/ieee_visC.csv', 'r')
f_link = open('data/link.csv', 'r')
data_info = csv.reader(f_info)
data_link = csv.reader(f_link)

# get the edge and node list of the net
edge_list = []
all_id = []
for row in data_link:
    edge_list.append(row)
    if row[0] not in all_id:
        all_id.append(row[0])
    if row[1] not in all_id:
        all_id.append(row[1])


# get the information dict

info_dict = dict()
index = 0
for row in data_info:
    if index > 0:
        tmp_dict = dict()
        tmp_dict['year'] = int(row[1])
        tmp_dict['conf'] = row[0]
        tmp_dict['abt'] = participle(row[9])
        tmp_dict['cited'] = int(row[-1])
        tmp_afs = row[11].strip().split(',')
        tmp_afs = [aff.strip() for aff in tmp_afs]
        tmp_dict['aff'] = tmp_afs
        info_dict[row[7]] = tmp_dict
    index += 1


# 将所有节点都进行游走
G = nx.DiGraph()
G.add_edges_from(edge_list)

# tmp_path = walk_times(5, G, all_id, info_dict, None, ret=True, conf=True, aff=False, abt=False, year=False, cited=False)
# write_path('test.csv', tmp_path)

# example
tmp_all_path = walk_all(G, all_id, info_dict, conf=True, aff=True, abt=True, year=True, cited=True)
write_path('TEST/test.csv', tmp_all_path)


# 多属性组合批量游走
# attr_list = ['conf', 'aff', 'abt', 'year', 'cited']  # ['conf', 'aff', 'abt', 'year', 'cited']
# walk_times_list = [5, 10]
# steps_limit = [5, 10]  # [5, 10, None]
# ret_limit = [True, False]
# index = [0, 3, 4]
# #
# for wt in walk_times_list:
#     for sl in steps_limit:
#         for rl in ret_limit:
#             for tf in range(len(index) + 1):
#                 comb = [False, False, False, False, False]
#                 if tf != 0:
#                     comb[index[tf - 1]] = True
#                 tmp_all_path = walk_times(wt, G, all_id, info_dict, steps=sl, ret=rl, conf=comb[0], aff=comb[1], abt=comb[2], year=comb[3], cited=comb[4])
#                 fileName = ''
#                 count = 0
#                 for i in range(5):
#                     if comb[i] is True:
#                         count += 1
#                         if count > 1:
#                             fileName += '_' + attr_list[i]
#                         else:
#                             fileName += attr_list[i]
#                 if count == 0:
#                     fileName += 'random'
#                 dirName = 'new_corpus/wt_' + str(wt) + '/sl_' + str(sl) + '/rl_' + str(rl) + '/'
#                 pathName = dirName + fileName + '/' + fileName + '_path.csv'
#                 write_path(pathName, tmp_all_path)
#                 print(pathName)

