import os
import numpy as np
import pandas as pd

from configs import settings


def execute():
    result_dir = os.path.join(settings.root_dir, 'results')
    datasets = os.listdir(result_dir)
    global_result_dict = {}
    forget_result_dict = {}
    test_result_dict = {}
    train_result_dict = {}
    transform_result_dict = {}
    test_forget_result_dict = {}
    test_retain_result_dict = {}

    for dataset in datasets:
        global_result_dict[dataset] = {}
        forget_result_dict[dataset] = {}
        test_result_dict[dataset] = {}
        train_result_dict[dataset] = {}
        transform_result_dict[dataset] = {}
        test_forget_result_dict[dataset] = {}
        test_retain_result_dict[dataset] = {}

        data_dir = os.path.join(result_dir, dataset)
        forget_ratios = os.listdir(data_dir)
        for forget_ratio in forget_ratios:
            global_result_dict[dataset][forget_ratio] = {}
            forget_result_dict[dataset][forget_ratio] = {}
            test_result_dict[dataset][forget_ratio] = {}
            train_result_dict[dataset][forget_ratio] = {}
            transform_result_dict[dataset][forget_ratio] = {}
            test_forget_result_dict[dataset][forget_ratio] = {}
            test_retain_result_dict[dataset][forget_ratio] = {}

            ratio_dir = os.path.join(data_dir, forget_ratio)
            uni_names = os.listdir(ratio_dir)
            for uni_name in uni_names:
                if uni_name.endswith('.npy'):
                    train_file = uni_name
                    train_file_path = os.path.join(ratio_dir, uni_name)
                    train_result = np.load(train_file_path, allow_pickle=True)
                    train_result_type = os.path.splitext(train_file)[0].split('_')[-1]
                    if '_global.npy' in train_file:
                        train_result_dict[dataset][forget_ratio][train_result_type] = train_result.item()['global']
                    if '_forget.npy' in train_file:
                        train_result_dict[dataset][forget_ratio][train_result_type] = train_result.item()['global']
                    if '_test.npy' in train_file:
                        train_result_dict[dataset][forget_ratio][train_result_type] = train_result.item()['global']
                    if '_test_forget.npy' in train_file:
                        train_result_dict[dataset][forget_ratio][train_result_type] = train_result.item()['global']
                    if '_test_retain.npy' in train_file:
                        train_result_dict[dataset][forget_ratio][train_result_type] = train_result.item()['global']

                else:
                    global_result_dict[dataset][forget_ratio][uni_name] = {}
                    forget_result_dict[dataset][forget_ratio][uni_name] = {}
                    test_result_dict[dataset][forget_ratio][uni_name] = {}
                    transform_result_dict[dataset][forget_ratio][uni_name] = {}
                    test_forget_result_dict[dataset][forget_ratio][uni_name] = {}
                    test_retain_result_dict[dataset][forget_ratio][uni_name] = {}

                    uni_dir = os.path.join(ratio_dir, uni_name)
                    result_files = os.listdir(uni_dir)
                    for result_file in result_files:
                        result_file_path = os.path.join(uni_dir, result_file)
                        # load data
                        result = np.load(result_file_path, allow_pickle=True)

                        result_type = result_file.split('_')[-2]
                        if result_type not in global_result_dict[dataset][forget_ratio][uni_name].keys():
                            global_result_dict[dataset][forget_ratio][uni_name][result_type] = {}

                        if result_type not in forget_result_dict[dataset][forget_ratio][uni_name].keys():
                            forget_result_dict[dataset][forget_ratio][uni_name][result_type] = {}

                        if result_type not in test_result_dict[dataset][forget_ratio][uni_name].keys():
                            test_result_dict[dataset][forget_ratio][uni_name][result_type] = {}

                        if result_type not in transform_result_dict[dataset][forget_ratio][uni_name].keys():
                            transform_result_dict[dataset][forget_ratio][uni_name][result_type] = {}

                        if result_type not in test_forget_result_dict[dataset][forget_ratio][uni_name].keys():
                            test_forget_result_dict[dataset][forget_ratio][uni_name][result_type] = {}

                        if result_type not in test_retain_result_dict[dataset][forget_ratio][uni_name].keys():
                            test_retain_result_dict[dataset][forget_ratio][uni_name][result_type] = {}

                        if '_global.npy' in result_file_path:
                            global_result_dict[dataset][forget_ratio][uni_name][result_type] = result.item()['global']
                        if '_forget.npy' in result_file_path:
                            forget_result_dict[dataset][forget_ratio][uni_name][result_type] = result.item()['global']
                        if '_test.npy' in result_file_path:
                            test_result_dict[dataset][forget_ratio][uni_name][result_type] = result.item()['global']
                        if '_transform.npy' in result_file_path:
                            transform_result_dict[dataset][forget_ratio][uni_name][result_type] = result.item()['global']
                        if '_testforget.npy' in result_file_path:
                            test_forget_result_dict[dataset][forget_ratio][uni_name][result_type] = result.item()['global']
                        if '_testretain.npy' in result_file_path:
                            test_retain_result_dict[dataset][forget_ratio][uni_name][result_type] = result.item()['global']

        csv_dir = os.path.join(settings.root_dir, 'results_csv')
        os.makedirs(csv_dir, exist_ok=True)

        new_order_index = ['RL', 'GA', 'IU', 'BU', 'GA_l1', 'SalUn', 'fisher', 'UNSC', 'DELETE']
        new_order_header = ['ul', 'ft', 'neverecall', 'SSR', 'SSGS', 'SSPR']

        # save df global
        for dataset in global_result_dict.keys():
            for forget_ratio in global_result_dict[dataset].keys():
                df = pd.DataFrame(global_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_global.csv' % (dataset, forget_ratio))
                df = df.reindex(new_order_header, columns=new_order_index)
                df.to_csv(csv_path, index=True, header=True)

        # save df forget
        for dataset in forget_result_dict.keys():
            for forget_ratio in forget_result_dict[dataset].keys():
                df = pd.DataFrame(forget_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_forget.csv' % (dataset, forget_ratio))
                df = df.reindex(new_order_header, columns=new_order_index)
                df.to_csv(csv_path, index=True, header=True)

        # save df test
        for dataset in test_result_dict.keys():
            for forget_ratio in test_result_dict[dataset].keys():
                df = pd.DataFrame(test_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_test.csv' % (dataset, forget_ratio))
                df = df.reindex(new_order_header, columns=new_order_index)
                df.to_csv(csv_path, index=True, header=True)

        # save df test_retain
        for dataset in test_retain_result_dict.keys():
            for forget_ratio in test_retain_result_dict[dataset].keys():
                df = pd.DataFrame(test_retain_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_test_retain.csv' % (dataset, forget_ratio))
                df = df.reindex(new_order_header, columns=new_order_index)
                df.to_csv(csv_path, index=True, header=True)

        # save df test_forget
        for dataset in test_forget_result_dict.keys():
            for forget_ratio in test_forget_result_dict[dataset].keys():
                df = pd.DataFrame(test_forget_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_test_forget.csv' % (dataset, forget_ratio))
                df = df.reindex(new_order_header, columns=new_order_index)
                df.to_csv(csv_path, index=True, header=True)

        # save df forget_transform
        for dataset in transform_result_dict.keys():
            for forget_ratio in transform_result_dict[dataset].keys():
                df = pd.DataFrame(transform_result_dict[dataset][forget_ratio])
                csv_path = os.path.join(csv_dir, 'result_%s_%s_transform.csv' % (dataset, forget_ratio))
                df = df.reindex(new_order_header, columns=new_order_index)
                df.to_csv(csv_path, index=True, header=True)

        # save df train
        df = pd.DataFrame(train_result_dict)
        csv_path = os.path.join(csv_dir, 'result_train.csv')
        df.to_csv(csv_path, index=True, header=True)


if __name__ == "__main__":
    execute()
