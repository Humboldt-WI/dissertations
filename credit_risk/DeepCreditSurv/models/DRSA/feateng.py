import argparse
import random


# build feat_index
def build_feat_index(feat_index_f, features, df):
    feat_index = {}
    max_index = 0
    feat_index_lines = []

    # truncate
    feat_index["0:truncate"] = max_index
    feat_index_lines.append("0:truncate\t{}\n".format(max_index))
    max_index += 1
    low, high, multipliers = find_width(df, features)
    for i in range(len(features)):
        feat_num = i + 1
        # other
        feat_index["{}:other".format(feat_num)] = max_index
        feat_index_lines.append("{}:other\t{}\n".format(feat_num, max_index))
        max_index += 1
        # normal values

        for value in range(low[i], high[i]):
            feat_index["{}:{}".format(feat_num, value)] = max_index
            feat_index_lines.append("{}:{}\t{}\n".format(feat_num, value, max_index))
            max_index += 1

    # write feat_index file:
    with open(feat_index_f, 'w') as f:
        f.writelines(feat_index_lines)

    return feat_index, feat_index_lines, multipliers


def find_width(df, features):
    dist = df[features].max() - df[features].min()
    nunique = df[features].nunique()
    min_val = df[features].min()
    max_val = df[features].max()
    low = []
    high = []
    multipliers = []
    for i in range(len(features)):
        if nunique[i] - dist[i] > 1:
            if min_val[i] >= 0:
                low.append(0)
                multiplier = nunique[i] / max_val[i]
                multipliers.append(multiplier)
            else:
                multiplier = nunique[i] / dist[i]
                multipliers.append(multiplier)
                low.append(round(min_val[i] * multiplier - 1))
            high.append(round(max_val[i] * multiplier + 2))
        else:
            multiplier = 1
            multipliers.append(multiplier)
            if min_val[i] == 0:
                low.append(0)
            else:
                low.append(round(min_val[i] - 1))
            high.append(round(max_val[i] + 2))

    return low, high, multipliers


# build yzbx data
def build_yzbx_data(feat_index, raw_train, raw_test, yzbx_train, yzbx_test, features, multipliers):
    fnames_i = [raw_train, raw_test]
    fnames_o = [yzbx_train, yzbx_test]

    for j in range(len(fnames_i)):
        yzbx_list = []
        with open(fnames_i[j], 'r') as f:
            lines = f.readlines()
            for line in lines:
                x_items = []
                # truncate
                x_items.append("0:1")
                line_items = line.split(',')
                for i in range(len(line_items) - 2):
                    feat_num = i + 1
                    feat_name = features[i]
                    feat_val = int(float(line_items[i]) * multipliers[i])
                    key = "{}:{}".format(feat_num, feat_val)
                    if key not in feat_index.keys():
                        key = "{}:other".format(feat_num)
                    x_items.append("{}:1".format(feat_index[key]))
                # get b and z
                # t from hour to day
                t = int(float(line_items[-2]))
                e = int(float(line_items[-1]))
                if e == 1:
                    z = t
                    b = random.randint(z + 1, 2 * (z + 1))
                else:
                    b = t
                    z = random.randint(b + 1, 2 * (b + 1))
                # dummy y
                y = 0
                yzbx = ' '.join([str(y), str(z), str(b)] + x_items)
                yzbx_list.append(yzbx + '\n')

        with open(fnames_o[j], 'w') as f:
            f.writelines(yzbx_list)


# calculate features' statistical: max value, min value, mean value, and different value counts
def feat_stat(train, test, features):
    feat_dict = {}
    for feat in features:
        feat_dict[feat] = set()

    # train
    with open(train, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(',')
            for i in range(len(items) - 2):
                feat_name = features[i]
                feat_dict[feat_name].add(float(items[i]))
    # test
    with open(test, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(',')
            for i in range(len(items) - 2):
                feat_name = features[i]
                feat_dict[feat_name].add(float(items[i]))

    # print result
    for feat in features:
        stats = feat_dict[feat]
        print("{} result:".format(feat))
        print("max value:{0}    min value:{1}   distinct cnt:{2}".format(max(stats), min(stats), len(stats)))


