#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import argparse
from collections import OrderedDict
from pexpect import pxssh


def send_command(s, cmd):
    s.sendline(cmd)
    s.prompt()
    return str(s.before).replace('\\r\\n', '\r\n')
    #print(str(s.before).replace('\\r\\n', '\r\n'))


def connect(host, user, password):
    try:
        s = pxssh.pxssh()
        s.login(host, user, password)
        print("ssh login {} succeed.".format(host))
        return s
    except:
        print('ssh login {} error'.format(host))
        exit(0)


def get_info(nvidia_info):
    lines = nvidia_info.split('\r\n')[1:-1]
    lines.reverse()
    usage_info = []
    # RE: 匹配nvidia-smi中的 '|   1  Tesla K40m On | 00000000:04:00.0 Off |    0 |'
    # 提取显卡型号和显卡列表用
    re_list = "\|\s+(\d{1,2})\s+(.+?)\s+(On|Off)\s+\|\s+\d+?:"
    # RE: 匹配nvidia-smi中的'| 2 142239 C python 10036MiB |'
    # 提取显卡目前的使用情况
    re_usage = "\|\s+(\d{1,2})\s+(\d+)\s+\S+\s+\S+\s+(\d+MiB)\s+\|"
    # RE: 匹配'| 50%   83C    P2   172W / 250W |  10313MiB / 12196MiB | 75% Default |'
    # 提取显存大小
    re_mem = "\d+MiB\s+/\s+(\d+MiB)\s+\|\s+\d+%"
    gpu_mem_list = []
    gpu_mem_dict = {}
    # 记录显卡列表
    gpu_list = OrderedDict()
    for line in lines:
        match_list = re.search(re_list, line)
        match_usage = re.search(re_usage, line)
        match_mem = re.search(re_mem, line)
        if match_list:
            gpu_list[int(match_list.group(1))] = match_list.group(2)
        if match_usage:
            gpu_id = int(match_usage.group(1))
            pid = match_usage.group(2)
            used_mem = match_usage.group(3)
            usage_info.append([gpu_id, pid, used_mem])
        if match_mem:
            gpu_mem_list.append(match_mem.group(1))
    assert len(gpu_list) == len(gpu_mem_list)
    for gpu_id, mem in zip(gpu_list, gpu_mem_list):
        gpu_mem_dict[gpu_id] = mem

    # 如果该显卡被使用标注1
    gpu_mask = [0] * (max(gpu_list) + 1)
    for gpu_id, _, _ in usage_info:
        gpu_mask[gpu_id] = 1
    # return: {'3': 'Tesla K40m', '2': 'Tesla K40m', '1': 'Tesla K40m', '0': 'Tesla K40m'},
    # [['3', '142249', '10036MiB'], ['2', '142239', '10036MiB'], ['0', '123458', '7569MiB']]
    # [1, 0, 1, 1]
    # {'3': '11441MiB'}
    return gpu_list, usage_info, gpu_mask, gpu_mem_dict


def get_pid2ut(ps_info):
    lines = ps_info.split('\r\n')
    pid2ut = {}
    lc = 1
    for line in lines[1:-1]:
        item = line.split()
        pid2ut[item[1]] = (item[0], item[9])
        lc += 1
    return pid2ut


def main():
    cmd = argparse.ArgumentParser(description='gpu usage.')
    cmd.add_argument('--user', type=str, default="xhu", help='user name.')
    cmd.add_argument('--password', type=str, default="huxiao", help='password.')
    args = cmd.parse_args()
    # node_list = [
    #     'gpu05', 'gpu08', 'gpu09', 'gpu10', 'gpu11', 'gpu12', 'gpu13', 'gpu14', 'gpu15', 'gpu16'
    # ]
    node_list = [
        'gpu10', 'gpu11', 'gpu12', 'gpu13', 'gpu14', 'gpu15', 'gpu16'
    ]
    node_list.reverse()
    # node_list = ['gpu05']
    info = {node: [] for node in node_list}
    # 所有节点的gpu使用情况
    # 结构:node: [gpu mask]
    all_gpu_mask = {}
    # 所有的节点的gpu型号
    # 结构:node:{0: Titan}
    all_gpu_type = {}
    # {node:{gpu_id: mem}}
    all_gpu_mem = {}
    for node in node_list:
        s = connect(node, args.user, args.password)
        nvidia_info = send_command(s, 'nvidia-smi')
        ps_info = send_command(s, 'ps aux')

        # user & time
        pid2ut = get_pid2ut(ps_info)
        # gpu_list为dict, e.g.{'0': 'Titan xp'i}
        gpu_list, gpu_load_info, gpu_mask, gpu_mem_dict = get_info(nvidia_info)
        all_gpu_mask[node] = gpu_mask
        all_gpu_type[node] = gpu_list
        all_gpu_mem[node] = gpu_mem_dict
        if not gpu_load_info:
            info[node].append(["", '0', '0', '0', '0', '0'])
            continue
        for gpu_id, pid, used_mem in gpu_load_info:
            assert pid in pid2ut
            info[node].append([pid2ut[pid][0], pid, gpu_id, gpu_list[gpu_id], used_mem, pid2ut[pid][1]])

        #关闭该node的ssh连接
        s.logout()
    # sort by gpu id
    for node_k, node_v in info.items():
        node_v.sort(key=lambda item: item[2])

    for node_k, node_v in info.items():
        print(node_k + ":")
        if not node_v[0][0]:
            print("***empty load !***.")
            continue
        for item in node_v:
            print("user: {}\tpid: {}\tgpu_id: {}\tgpu_type: {}\tusage: {} / {}\ttime: {}.".format(item[0], item[1], item[2], item[3],
                                                                               item[4], all_gpu_mem[node_k][item[2]], item[5]))

    print("\n********empty load info********\n")
    all_empty_load = {}
    for node, mask in all_gpu_mask.items():
        for i, m in enumerate(mask):
            if m == 0:
                if node not in all_empty_load:
                    all_empty_load[node] = []
                all_empty_load[node].append("gpu_id: {}\tgpu_type: {}\tgpu_mem: {}".format(i, all_gpu_type[node][i], all_gpu_mem[node][i]))

    for node, empty_info in all_empty_load.items():
        print(node + ':')
        print("\n".join(empty_info))
        print("\n")

if __name__ == '__main__':
    main()
