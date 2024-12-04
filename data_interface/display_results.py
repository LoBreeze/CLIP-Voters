import numpy as np
import sklearn.metrics as sk

recall_level_default = 0.95

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1)) # 认为是正类的样本，以及对应的得分
    neg = np.array(_neg[:]).reshape((-1, 1)) # 认为是负类的样本，以及对应的得分
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1 # 正类标签为1，负类标签为0
    try:
        # 在正类标签时，看其得分
        auroc = sk.roc_auc_score(labels, examples)
    except ValueError:
        if np.all(labels == 1):  # 如果全是正类
            auroc =  1.0 if np.all(examples > 0.5) else 0.0
        else:  # 如果全是负类
            auroc = 1.0 if np.all(examples < 0.5) else 0.0
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)


    return auroc, aupr, fpr


def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))
    #print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    #print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    #print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(aurocs, auprs, fprs, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*np.mean(fprs), 100*np.mean(aurocs), 100*np.mean(auprs)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*np.std(fprs), 100*np.std(aurocs), 100*np.std(auprs)))
    #print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    #print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    #print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))


def show_performance_comparison(pos_base, neg_base, pos_ours, neg_ours, baseline_name='Baseline',
                                method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos_base: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg_base: 0's class scores generated by the baseline
    '''
    auroc_base, aupr_base, fpr_base = get_measures(pos_base[:], neg_base[:], recall_level)
    auroc_ours, aupr_ours, fpr_ours = get_measures(pos_ours[:], neg_ours[:], recall_level)

    print('\t\t\t' + baseline_name + '\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
        int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours))
    print('AUROC:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * auroc_base, 100 * auroc_ours))
    print('AUPR:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * aupr_base, 100 * aupr_ours))
    # print('FDR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
    #     int(100 * recall_level), 100 * fdr_base, 100 * fdr_ours))
    
    
    
    
# 误分类
# E-AURC和AURC
# 拒绝：模糊拒绝（降低将unclaer 误分类的风险）和 OOD拒绝 （ 降低 OOD 分类到 known class 的风险）
 # 公式 （3） （4）
 # 有一个阈值，小于阈值则拒绝，r(x) = 1; 大于阈值则接受，r(x) = 0，认为是已知分类, argmax p(x)
# AURC, EAURC计算
def calc_aurc_eaurc(softmax, correct):
    correctness = np.array(correct)  # 转换为NumPy数组
    softmax_max = np.array(softmax)  # 转换为NumPy数组
    # 将softmax值和正确性按softmax值排序
    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x: x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)  # 解压排序后的值
    # 计算风险和覆盖率
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    # 计算AURC和EAURC
    aurc, eaurc = aurc_eaurc(risk_li)
    return aurc, eaurc  # 返回结果

# 计算覆盖率和风险
def coverage_risk(confidence, correctness):
    # confidence: softmax值, from high to low
    risk_list = []  # 风险列表
    coverage_list = []  # 覆盖率列表
    risk = 0  # 初始化风险
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)  # 计算覆盖率
        coverage_list.append(coverage)
        if correctness[i] == 0:  # 如果预测错误
            risk += 1  # 风险增加
        risk_list.append(risk / (i + 1))  # 计算风险率
    return risk_list, coverage_list  # 返回风险和覆盖率列表

# 计算AURC和EAURC
def aurc_eaurc(risk_list):
    r = risk_list[-1]  # 最终风险
    risk_coverage_curve_area = 0  # 初始化面积
    optimal_risk_area = r + (1 - r) * np.log(1 - r)  # 计算最佳风险面积
    for risk_value in risk_list:  # 遍历风险列表
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))  # 累加面积
    aurc = risk_coverage_curve_area  # AURC
    eaurc = risk_coverage_curve_area - optimal_risk_area  # EAURC
    print("AURC {0:.2f}".format(aurc * 1000))  # 输出AURC
    print("EAURC {0:.2f}".format(eaurc * 1000))  # 输出EAURC
    return aurc, eaurc  # 返回结果
    
    
    