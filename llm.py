import numpy as np
import random
import requests
import json
import dashscope
from http import HTTPStatus
import time
import os

dashscope.api_key = 'Your key'

def callMixtral(query):
    url = "http://192.168.1.101:11434/api/generate"
    query = {"role": "user", "content": query}
    # data = {"model": "mistral", "prompt": str(query)}
    data = {"model": "mixtral", "prompt": str(query)}
    headers = {"Content-Type": "application/json"}

    try:
        # url = "http://ganglia.act.buaa.edu.cn/"
        response = requests.post(url, json=data, verify=False)
        temp = response.text.split('\n')
        res = ""
    # print(temp)
        for i in temp:
            if i == "":
                continue
            res += json.loads(i)["response"]
        print(res)
        return res
    except requests.RequestException as e:
        print(f"请求发生异常: {e}")
        return None


# 调用QWen模型
def call_with_messages(content):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': content}]
    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_max,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
        return response["output"]["choices"][0]["message"]["content"]
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return None


def get_trajectory(success, drop_times, steps):
    if success == 1:
        prompt = "the agent successfully got the key and reached the gate."
    else:
        prompt = "the agent failed to reach the gate."
    prompt += f"it dropped the key {drop_times} times.\n"
    prompt += f"it took {steps} steps totally."
    return prompt


def get_trajectory_QWen(success_exit, success_lava, steps, is_fall_lava):
    if success_exit == 1:
        assert success_lava == 1    # 两者必定同时成功
        prompt = "the agent successfully crossed the lava and reached the exit."
    elif success_exit == 0 and success_lava == 1:
        prompt = "the agent successfully crossed the lava but failed to reach the exit."
    else:
        prompt = "the agent failed to cross the lava and failed to reach the exit."
    prompt += f"it took {steps} steps totally."
    if is_fall_lava:
        prompt += "it fell into the lava and died.\n"
    else:
        prompt += "it didn't fall into the lava.\n"
    return prompt



def get_prompt(success, drop_times, steps, is_drop_times=False, is_Three=False):
    background = "In a 2 grid world, there is a key and a gate. The agent needs to first get the key and then it can reach the gate and exit.\n"
    target = ("We hope that the agent can get the key and reach the gate successfully in the least steps.\n")
    if is_drop_times and is_Three:
        target = "We aim for the agent to drop the key exactly three times and successfully reach the gate. Ideally, the number of drops should be as close to three as possible.\n"
              #" First of all, success is the most important, if one is success and another one is falied, you don't need to consider other things.\nThen if two trajectory both success or fail, a smaller steps is better. when two trajectories' steps are equal, we hope that the agent can drop the key less times.\n")#
    mid = "here are two trajectories of the agent. Please choose the better one.\n"
    prompt0 = f"0.{get_trajectory(success[0], drop_times[0], steps[0])}"
    prompt1 = f"1.{get_trajectory(success[1], drop_times[1], steps[1])}"
    after = "please give me which one is better or they are equal. \nif the first one is better, you should return 0. otherwise, you should return 1. if they are equal, you should return 2.\n"
    struct = "Your answer should be structured as follows: \n answer:<your answer(a number)>"
    if is_drop_times:
        struct = "You should just return a number. Your answer should be structured as follows: \n <your answer(a number)>"
    return background + target + mid + prompt0 + prompt1 + after + struct

def get_prompt_QWen(success_exit, success_lava, steps, is_fall_lava, is_mixtral=False):
    background = "In a two-dimensional world, there exists a green exit and a lava strip. The agent is required to first cross the lava strip and then exit through the green exit.\n"
    target = "We desire that the agent accomplish this by taking as few steps as possible while avoiding falling into the lava, as doing so would result in its demise.\n"
    mid = "here are two trajectories of the agent. Please choose the better one.\n"
    prompt0 = f"0.{get_trajectory_QWen(success_exit[0], success_lava[0], steps[0], is_fall_lava[0])}"
    prompt1 = f"1.{get_trajectory_QWen(success_exit[1], success_lava[1], steps[1], is_fall_lava[1])}"
    after = "please give me which one is better or they are equal. \nif the first one is better, you should return 0. otherwise, you should return 1. if they are equal, you should return 2.\n"
    struct = "Your answer should be structured as follows: \n<your answer(a number)>"
    if is_mixtral:
        struct = "Your answer should be structured as follows: \n answer:<your answer(a number)>"
    return background + target + mid + prompt0 + prompt1 + after + struct

def getAnswer(res):
    # 用正则表达式提取出回答，找到answer:后面的数字
    # print(res)
    res = res.split('\n')
    for i in res:
        i = i.strip()
        if i.startswith("answer:"):
            return i[7:].strip()
        elif i.startswith("Answer:"):
            return i[7:].strip()
    return None


def getAnswer_QWen(res):
    # res合法的时候必定是数字
    if res is None:
        print(res)
        return None
    if not res.isdigit():
        print(res)
        return None
    return int(res)

def get_label(trajectory0, trajectory1):
    success = [trajectory0[0], trajectory1[0]]
    drop_times = [trajectory0[1], trajectory1[1]]
    steps = [trajectory0[2], trajectory1[2]]
    prompt = get_prompt(success, drop_times, steps)
    res = callMixtral(prompt)
    res = getAnswer(res)
    if not res.isdigit():
        res = 5
    else:
        res = int(res)
    print(f"res: {res}")
    if res is None:
        raise Exception("res is None")
    return res

def get_label_unlock_QWen(trajectory0, trajectory1, is_drop_times=True, is_Three=False):
    # success, drop_times, steps
    success = [trajectory0[0], trajectory1[0]]
    drop_times = [trajectory0[1], trajectory1[1]]
    steps = [trajectory0[2], trajectory1[2]]
    prompt = get_prompt(success, drop_times, steps, is_drop_times=is_drop_times, is_Three=is_Three)
    print(prompt)
    res = call_with_messages(prompt)
    res = getAnswer_QWen(res)
    retry_times = 5
    while res is None and retry_times > 0:
        print("res is None")
        res = call_with_messages(prompt)
        res = getAnswer_QWen(res)
        retry_times -= 1
    if res is None:
        print("#--------------放弃此次答题-----------------#")
    return res

def get_label_QWen(trajectory0, trajectory1):
    # success_exit, steps, success_lava, is_fall_lava
    success_exit = [trajectory0[0], trajectory1[0]]
    success_lava = [trajectory0[2], trajectory1[2]]
    steps = [trajectory0[1], trajectory1[1]]
    is_fall_lava = [trajectory0[3], trajectory1[3]]
    prompt = get_prompt_QWen(success_exit, success_lava, steps, is_fall_lava)
    res = call_with_messages(prompt)
    res = getAnswer_QWen(res)
    retry_times = 5
    while res is None and retry_times > 0:
        print("res is None")
        res = call_with_messages(prompt)
        res = getAnswer_QWen(res)
        retry_times -= 1
    if res is None:
        print("#--------------放弃此次答题-----------------#")
    return res

def get_label_lava_mixtral(trajectory0, trajectory1):
    # success_exit, steps, success_lava, is_fall_lava
    success_exit = [trajectory0[0], trajectory1[0]]
    success_lava = [trajectory0[2], trajectory1[2]]
    steps = [trajectory0[1], trajectory1[1]]
    is_fall_lava = [trajectory0[3], trajectory1[3]]
    prompt = get_prompt_QWen(success_exit, success_lava, steps, is_fall_lava, is_mixtral=True)
    print(prompt)
    res = callMixtral(prompt)
    res = getAnswer(res)
    if not res.isdigit():
        res = 5
    else:
        res = int(res)
    print(f"res: {res}")
    if res is None:
        raise Exception("res is None")
    return res


def llm4unlock():
    data = np.load("data.npy")
    num = len(data)
    # 成功的数据是第一个元素是1的
    success_data = data[:50]
    # 失败的数据是第一个元素是0的
    fail_data = data[50:]
    # 采样数目
    sample_num = 250
    sample_trajectory = []
    labels = []
    # i = 50
    # # 成功+失败采样25, 成功+成功采样25, 其余采样200
    # while i > 0:
    #     random_index = np.random.randint(success_data.shape[0])
    #     success = success_data[random_index]
    #     random_index = np.random.randint(fail_data.shape[0])
    #     fail = fail_data[random_index]
    #     label = get_label_unlock_QWen(success, fail)
    #     if label == 0 or label == 1:
    #         sample_trajectory.append([success, fail])
    #         labels.append(label)
    #         print(f"success: {success}, fail: {fail}, label: {label}")
    #     else:
    #         print("error: label is not 0 or 1")
    #         continue
    #     i -= 1
    #     # 停一秒
    #     time.sleep(1)
    #     print(f"剩余次数: {i}/25")
    # sample_trajectory = np.array(sample_trajectory)
    # labels = np.array(labels)
    # np.save("./data/sample_trajectory_QWen_unlock_success.npy", sample_trajectory)
    # np.save("./data/labels_QWen_unlock_success.npy", labels)
    # exit()
    i = 50
    while i > 0:
        random_index = np.random.randint(success_data.shape[0])
        success = success_data[random_index]
        random_index = np.random.randint(success_data.shape[0])
        fail = success_data[random_index]
        label = get_label_unlock_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue
        i -= 1
        time.sleep(1)
        print(f"剩余次数: {i}/25")

    i = 50
    # 成功+失败采样25, 成功+成功采样25, 其余采样200
    while i > 0:
        random_index = np.random.randint(success_data.shape[0])
        success = success_data[random_index]
        random_index = np.random.randint(fail_data.shape[0])
        fail = fail_data[random_index]
        label = get_label_unlock_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue
        i -= 1
        # 停一秒
        time.sleep(1)
        print(f"剩余次数: {i}/25")


    i = 200
    while i > 0:
        random_index = np.random.randint(fail_data.shape[0])
        success = fail_data[random_index]
        random_index = np.random.randint(fail_data.shape[0])
        fail = fail_data[random_index]
        label = get_label_unlock_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue
        i -= 1
        time.sleep(1)
        print(f"剩余次数: {i}/200")
    sample_trajectory = np.array(sample_trajectory)
    labels = np.array(labels)
    np.save("./data/sample_trajectory_QWen_unlock.npy", sample_trajectory)
    np.save("./data/labels_QWen_unlock.npy", labels)
    with open("sample_trajectory.txt", "w") as f:
        for i in range(len(sample_trajectory)):
            f.write(f"{sample_trajectory[i]}: {labels[i]}\n")

def llm4lava():
    data = np.load("data.npy")
    num = len(data)
    # 首先是成功的数据
    double_success_data = data[data[:, 0] == 1]
    # 其次是失败了但是到第二阶段的数据掉入lava的数据
    fail_die_data = data[(data[:, 2] == 1) & (data[:, 0] == 0) & (data[:, 3] == 1)]
    # 然后是失败了没掉入lava的数据 这个没有
    fail_live_data = data[(data[:, 2] == 1) & (data[:, 0] == 0) & (data[:, 3] == 0)]
    # 最后是失败了掉入lava的数据
    fail_fail_data = data[(data[:, 2] == 0) & (data[:, 3] == 1)]
    print(f"成功的数据: {double_success_data.shape[0]}, 失败的数据: {fail_die_data.shape[0]}, 没掉入lava的数据: {fail_live_data.shape[0]}, 双输lava的数据: {fail_fail_data.shape[0]}")
    sample_num = 250
    sample_trajectory = []
    labels = []
    i = 50
    # 成功+成功采样50
    while i > 0:
        random_index = np.random.randint(double_success_data.shape[0])
        success = double_success_data[random_index]
        random_index = np.random.randint(double_success_data.shape[0])
        fail = double_success_data[random_index]
        label = get_label_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue

        i -= 1
        print(f"成功成功剩余次数: {i}/25")
    i = 50
    # 成功+失败没掉入lava采样50
    while i > 0:
        random_index = np.random.randint(double_success_data.shape[0])
        success = double_success_data[random_index]
        random_index = np.random.randint(fail_live_data.shape[0])
        fail = fail_live_data[random_index]
        label = get_label_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue
        i -= 1
        print(f"成功失败没掉入lava剩余次数: {i}/25")
    i = 50
    # 没掉入lava+掉入lava采样50
    while i > 0:
        random_index = np.random.randint(fail_live_data.shape[0])
        success = fail_live_data[random_index]
        random_index = np.random.randint(fail_die_data.shape[0])
        fail = fail_die_data[random_index]
        label = get_label_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue
        i -= 1
        print(f"失败没掉入lava+掉入lava剩余次数: {i}/50")
    i = 50
    # 掉入lava和双输lava采样50
    while i > 0:
        random_index = np.random.randint(fail_die_data.shape[0])
        success = fail_die_data[random_index]
        random_index = np.random.randint(fail_fail_data.shape[0])
        fail = fail_fail_data[random_index]
        label = get_label_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue
        i -= 1
        print(f"掉入lava+双输lava剩余次数: {i}/50")
    # 双输双输采样100
    i = 100
    while i > 0:
        random_index = np.random.randint(fail_fail_data.shape[0])
        success = fail_fail_data[random_index]
        random_index = np.random.randint(fail_fail_data.shape[0])
        fail = fail_fail_data[random_index]
        label = get_label_QWen(success, fail)
        if label == 0 or label == 1:
            sample_trajectory.append([success, fail])
            labels.append(label)
            print(f"success: {success}, fail: {fail}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            continue
        i -= 1
        print(f"双输双输剩余次数: {i}/100")
    sample_trajectory = np.array(sample_trajectory)
    labels = np.array(labels)
    np.save("./data/sample_trajectory_QWen.npy", sample_trajectory)
    np.save("./data/labels_QWen.npy", labels)
    with open("sample_trajectory_QWen.txt", "w") as f:
        for i in range(len(sample_trajectory)):
            f.write(f"{sample_trajectory[i]}: {labels[i]}\n")

def ask_unlock_from_mixtral_and_QWen():
    data = np.load("./data/sample_trajectory.npy")
    mixtral_labels = np.load("./data/labels.npy")
    QWen_labels = []
    target_path = "./difference/Mixtral_unlock_QWen.npy"
    for traject0, traject1 in data:
        label = get_label_unlock_QWen(traject0, traject1)
        if label == 0 or label == 1:
            QWen_labels.append(label)
            print(f"success: {traject0}, fail: {traject1}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            QWen_labels.append(2)
            print(f"success: {traject0}, fail: {traject1}, label: {label}")
            continue
        time.sleep(1)
    QWen_labels = np.array(QWen_labels)
    np.save(target_path, QWen_labels)
    # 计算两者的差异
    for i in range(len(mixtral_labels)):
        if mixtral_labels[i] != QWen_labels[i]:
            print(f"i: {i}, mixtral: {mixtral_labels[i]}, QWen: {QWen_labels[i]}")

def show_difference():
    mixtral_labels = np.load("./data/labels.npy")
    QWen_labels = np.load("./difference/Mixtral_unlock_QWen.npy")
    data = np.load("./data/sample_trajectory.npy")
    for i in range(len(mixtral_labels)):
        if mixtral_labels[i] != QWen_labels[i]:
            print(f"i: {i}, mixtral: {mixtral_labels[i]}, QWen: {QWen_labels[i]}")
            print(f"success: {data[i][0]}, fail: {data[i][1]}")


def ask_lava_from_QWen_and_mixtral():
    data = np.load("./data/sample_trajectory_QWen.npy")
    QWen_labels = np.load("./data/labels_QWen.npy")
    mixtral_labels = []
    target_path = "./difference/QWen_lava_mixtral.npy"
    i = 0
    for traject0, traject1 in data:
        label = get_label_lava_mixtral(traject0, traject1)
        if label == 0 or label == 1:
            mixtral_labels.append(label)
            print(f"success: {traject0}, fail: {traject1}, label: {label}")
        else:
            print("error: label is not 0 or 1")
            mixtral_labels.append(2)
            continue
        i += 1
        time.sleep(1)
        print(f"剩余次数: {i}/25")
    mixtral_labels = np.array(mixtral_labels)
    np.save(target_path, mixtral_labels)


if __name__ == '__main__':
    # ask_unlock_from_mixtral_and_QWen()
    # show_difference()
    #ask_lava_from_QWen_and_mixtral()
    show_difference_QWen_lava_mixtral()
