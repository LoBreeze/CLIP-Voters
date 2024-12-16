import os
import logging
import yaml
import numpy as np

def read_yaml(file_path):
    """
    读取 YAML 文件并返回其内容。
    
    :param file_path: YAML 文件的路径
    :return: 解析后的 YAML 数据（字典形式）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file does not exist: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            raise ValueError(f"Error while parsing YAML file: {exc}")
        
        
def setup_log(args, hparams, log_dir):
    """
    设置日志记录器。

    :param args: 包含日志配置的参数对象，需包含 `logs`属性。
    :param time_format: 自定义时间格式，默认格式为 '%Y-%m-%d %H:%M:%S'。
    :return: 配置好的日志记录器。
    """
    # 创建日志记录器
    log = logging.getLogger(__name__)
    
    
    # 设置日志格式，使用自定义时间格式
    time_format='%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(f'%(asctime)s : %(message)s', datefmt=time_format)
        
    # 文件处理器
    fileHandler = logging.FileHandler(os.path.join(log_dir, "id_ood_eval_info.log"))
    fileHandler.setFormatter(formatter)
    
    # 控制台处理器
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    
    # 配置日志记录器
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    
    # 输出测试日志
    log.info(f"{'='*10} Start Testing {'='*10}")
    log.info(f"Args:  {args}")
    log.info(f"Hyperparameters:  {hparams}")
    log.info(f"{'='*10}==============={'='*10}")
    
    return log


def save_as_dataframe(args, result_dir, out_datasets, fpr_list, auroc_list, aupr_list):
    fpr_list = [float('{:.2f}'.format(100*fpr)) for fpr in fpr_list]
    auroc_list = [float('{:.2f}'.format(100*auroc)) for auroc in auroc_list]
    aupr_list = [float('{:.2f}'.format(100*aupr)) for aupr in aupr_list]
    import pandas as pd
    import time
    data = {k:v for k,v in zip(out_datasets, zip(fpr_list,auroc_list,aupr_list))}
    data['AVG'] = [np.mean(fpr_list),np.mean(auroc_list),np.mean(aupr_list) ]
    data['AVG']  = [float('{:.2f}'.format(metric)) for metric in data['AVG']]
    # Specify orient='index' to create the DataFrame using dictionary keys as rows
    df = pd.DataFrame.from_dict(data, orient='index',
                       columns=['FPR95', 'AUROC', 'AUPR'])
    time_str = time.strftime("%m-%d__%H:%M")
    df.to_csv(os.path.join(result_dir, f'resluts.csv'))