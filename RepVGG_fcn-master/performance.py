import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from numpy.ma.extras import mask_rows

if __name__ == '__main__':
# -----------------------------------------------------max_mAP------------------------------------------------------#
    network = ['fcn8s']
    result_path = '/storage/sjpark/vehicle_data/precision_recall_per_class_p_threshold/Repvgg/a0/max_mAP/'

    image_sizes = ['256']

    expected_class = ['constructionguide', 'trafficdrum']
    class_names = glob(os.path.join(result_path, image_sizes[0], network[0],  'precision', '*'))
    class_names = [class_names[i].split('/')[-1] for i in range(len(class_names))]
    class_names.sort()

    probs = glob(os.path.join(result_path, image_sizes[0], network[0], 'precision', class_names[0], '*'))
    probs = [probs[i].split('/')[-1][:-4].split('_')[-1] for i in range(len(probs))]
    probs.sort()

    precision_dict = dict()
    recall_dict = dict()

    empty_classes = []
    for net_name in network:
        precision_dict[net_name] = dict()
        recall_dict[net_name] = dict()
        for img_s in image_sizes:
            precision_dict[net_name][img_s] = dict()
            recall_dict[net_name][img_s] = dict()
            for name in class_names:
                file_names = glob(os.path.join(result_path, img_s, net_name,  'precision', name, '*'))
                if len(file_names) == 0:
                    empty_classes.append(name)
                    pass
                precision_dict[net_name][img_s][name] = dict()
                recall_dict[net_name][img_s][name] = dict()
                for prob in probs:
                    precision_dict[net_name][img_s][name][prob] = dict(result_values=[], ap=[])
                    recall_dict[net_name][img_s][name][prob] = dict(result_values=[], ar=[])

    empty_classes = list(np.unique(empty_classes))
    for name_class in empty_classes:
        class_names.remove(name_class)

    for i in expected_class:
        class_names.remove(i)

    for net_name in network:
        for img_s in image_sizes:
            for name in class_names:
                for prob in probs:
                    file_path = os.path.join(result_path, img_s, net_name, 'precision', name, name + '_' + prob + '.txt')
                    with open(file_path, 'r') as f:
                        result_values = f.readlines()
                    result_values = [float(result_values[i].split('\n')[0]) for i in range(len(result_values))]
                    precision_dict[net_name][img_s][name][prob]['result_values'] = result_values
                    precision_dict[net_name][img_s][name][prob]['ap'] = np.sum(result_values) / len(result_values)
                    #
                    file_path = os.path.join(result_path,  img_s, net_name, 'recall', name, name + '_' + prob + '.txt')
                    with open(file_path, 'r') as f:
                        result_values = f.readlines()
                    result_values = [float(result_values[i].split('\n')[0]) for i in range(len(result_values))]
                    recall_dict[net_name][img_s][name][prob]['result_values'] = result_values
                    recall_dict[net_name][img_s][name][prob]['ar'] = np.sum(result_values) / len(result_values)
#----------------------------------------------------------------------------------------------------------------------#

# -----------------------------------------------------avr_mAP---------------------------------------------------------#

    network = ['fcn8s']
    result_path = '/storage/sjpark/vehicle_data/precision_recall_per_class_p_threshold/Repvgg/a0/avr_mAP/'

    image_sizes = ['256']

    expected_class = ['constructionguide', 'trafficdrum']
    class_names = glob(os.path.join(result_path, image_sizes[0], network[0], 'precision', '*'))
    class_names = [class_names[i].split('/')[-1] for i in range(len(class_names))]
    class_names.sort()

    probs = glob(os.path.join(result_path, image_sizes[0], network[0], 'precision', class_names[0], '*'))
    probs = [probs[i].split('/')[-1][:-4].split('_')[-1] for i in range(len(probs))]
    probs.sort()

    precision_dict_avr = dict()
    recall_dict_avr = dict()

    empty_classes = []
    for net_name in network:
        precision_dict_avr[net_name] = dict()
        recall_dict_avr[net_name] = dict()
        for img_s in image_sizes:
            precision_dict_avr[net_name][img_s] = dict()
            recall_dict_avr[net_name][img_s] = dict()
            for name in class_names:
                file_names = glob(os.path.join(result_path, img_s, net_name, 'precision', name, '*'))
                if len(file_names) == 0:
                    empty_classes.append(name)
                    pass
                precision_dict_avr[net_name][img_s][name] = dict()
                recall_dict_avr[net_name][img_s][name] = dict()
                for prob in probs:
                    precision_dict_avr[net_name][img_s][name][prob] = dict(result_values=[], ap=[])
                    recall_dict_avr[net_name][img_s][name][prob] = dict(result_values=[], ar=[])

    empty_classes = list(np.unique(empty_classes))
    for name_class in empty_classes:
        class_names.remove(name_class)

    for i in expected_class:
        class_names.remove(i)

    for net_name in network:
        for img_s in image_sizes:
            for name in class_names:
                for prob in probs:
                    file_path = os.path.join(result_path, img_s, net_name, 'precision', name, name + '_' + prob + '.txt')
                    with open(file_path, 'r') as f:
                        result_values = f.readlines()
                    result_values = [float(result_values[i].split('\n')[0]) for i in range(len(result_values))]
                    precision_dict_avr[net_name][img_s][name][prob]['result_values'] = result_values
                    precision_dict_avr[net_name][img_s][name][prob]['ap'] = np.sum(result_values) / len(result_values)
                    #
                    file_path = os.path.join(result_path, img_s, net_name, 'recall', name, name + '_' + prob + '.txt')
                    with open(file_path, 'r') as f:
                        result_values = f.readlines()
                    result_values = [float(result_values[i].split('\n')[0]) for i in range(len(result_values))]
                    recall_dict_avr[net_name][img_s][name][prob]['result_values'] = result_values
                    recall_dict_avr[net_name][img_s][name][prob]['ar'] = np.sum(result_values) / len(result_values)
#----------------------------------------------------------------------------------------------------------------------#


    mAP = dict()
    mAP_avr = dict()
    for net_name in network:
        mAP[net_name] = dict()
        mAP_avr[net_name] = dict()
        for prob in probs:
            mAP[net_name][prob] = []
            mAP_avr[net_name][prob] = []
            x = 0
            y=0
            for name in class_names:
                x += precision_dict[net_name]['256'][name][prob]['ap']
                y += precision_dict_avr[net_name]['256'][name][prob]['ap']
            mAP[net_name][prob].append(x / len(class_names))
            mAP_avr[net_name][prob].append(y / len(class_names))

    for name in class_names:
        print(precision_dict['fcn8s']['256'][name]['0.4']['ap'])
    #
    #
    # #
    val = []
    for prob in probs:
        val.append(mAP['fcn8s'][prob][0])
    #
    val2 = []
    for prob in probs:
        val2.append(mAP_avr['fcn8s'][prob][0])
    #
    plt.plot(probs, val, marker='o' ,label='max_mAP')
    plt.plot(probs, val2, marker='o' ,label='avr_mAP')
    plt.xlabel('threshold_probability')
    plt.ylabel('mAP')
    plt.title('mAP per threshold_probability')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), title="mAP", ncol=1)
    plt.imshow()
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------F1-score_for_max_mAP----------------------------------------------------------#
    f1_score = {}
    marco_f1_score = {}
    for name in class_names:
        for prob in probs:
            x = 2 * (precision_dict[network[0]][image_sizes[0]][name][prob]['ap'] * recall_dict[network[0]][image_sizes[0]][name][prob]['ar']) / (precision_dict[network[0]][image_sizes[0]][name][prob]['ap'] + recall_dict[network[0]][image_sizes[0]][name][prob]['ar'])
            if x > 0:
                f1_score.setdefault(name, {}).setdefault(prob, []).append(x)
                marco_f1_score.setdefault(prob, []).append(x)
            else:
                f1_score.setdefault(name, {}).setdefault(prob, []).append(0)
                marco_f1_score.setdefault(prob, []).append(0)

    for prob in probs:
        marco_f1_score[prob] = sum(marco_f1_score[prob]) / len(marco_f1_score[prob])
        # ----------------------------------------------------------------------------------------------------------------------#

# ----------------------------------------------------F1-score_for_avr_mAP----------------------------------------------------------#
    f1_score_avr = {}
    marco_f1_score_avr = {}
    for name in class_names:
        for prob in probs:
            x = 2 * (precision_dict_avr[network[0]][image_sizes[0]][name][prob]['ap'] *
                     recall_dict_avr[network[0]][image_sizes[0]][name][prob]['ar']) / (
                            precision_dict_avr[network[0]][image_sizes[0]][name][prob]['ap'] +
                            recall_dict_avr[network[0]][image_sizes[0]][name][prob]['ar'])
            if x > 0:
                f1_score_avr.setdefault(name, {}).setdefault(prob, []).append(x)
                marco_f1_score_avr.setdefault(prob, []).append(x)
            else:
                f1_score_avr.setdefault(name, {}).setdefault(prob, []).append(0)
                marco_f1_score_avr.setdefault(prob, []).append(0)

    for prob in probs:
        marco_f1_score_avr[prob] = sum(marco_f1_score_avr[prob]) / len(marco_f1_score_avr[prob])
# ----------------------------------------------------------------------------------------------------------------------#

# ----------------------------------------------------------per_f1_score_curve-------------------------------------------------------#
        # f1_score | f1_score_avr
        f1_scores = []
        plt.figure(figsize=(12, 8))

        for i in range(len(class_names)):
            for j in range(len(probs)):
                f1_scores.append(f1_score[class_names[i]][probs[j]][0])
            plt.plot(probs, f1_scores, marker='o', label=f'Class {class_names[i]}')
            f1_scores.clear()

        plt.xlabel('Probability Threshold')
        plt.ylabel('F1-score')
        plt.title('F1-score_curve')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Classes", ncol=1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
# ----------------------------------------------------------------------------------------------------------------------#

# ----------------------------------------------------------all_f1_score_curve-------------------------------------------------------#
        # f1_score | f1_score_avr
        f1_scores = []
        f1_scores2 = []
        plt.figure(figsize=(12, 8))

        for i in range(len(class_names)):
            for j in range(len(probs)):
                f1_scores.append(f1_score[class_names[i]][probs[j]][0])
                f1_scores2.append(f1_score_avr[class_names[i]][probs[j]][0])
            plt.plot(probs, f1_scores, marker='o', label=f'Class {class_names[i]} - max_mAP')
            plt.plot(probs, f1_scores2, marker='x',  label=f'Class {class_names[i]}- avr_mAP')
            f1_scores.clear()
            f1_scores2.clear()

        plt.xlabel('Probability Threshold')
        plt.ylabel('F1-score')
        plt.title('F1-score_curve')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Classes", ncol=1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
# ----------------------------------------------------------------------------------------------------------------------#
