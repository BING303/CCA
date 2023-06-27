from itertools import groupby
import os
import json
import datetime
import pickle
from unicodedata import category
import pandas as pd
import numpy as np
from sklearn import preprocessing

filter_min = 5
dividing_line = 0.8

data_directory = './datasets/Video_Games/'
category_name = 'Video_Games'
data_path = os.path.join(data_directory,'Video_Games.json')

users_id, items_id, ratings, reviews, times = [],[],[],[],[]
np.random.seed(2022)

# chang the time format
def str_to_days(s):
    st_date = datetime.date(1970, 1, 1)
    cur_date = datetime.date(
        int(s.split(', ')[1]), int(s.split(', ')[0].split(' ')[0]),
        int(s.split(', ')[0].split(' ')[1]))
    return (cur_date - st_date).days

with open(data_path,'r') as f:
    for line in f:
        js = json.loads(line)
        if str(js['reviewerID']) == 'unknown':
            print("unknown")
            continue
        if str(js['asin']) == 'unknown':
            print("unknown2")
            continue
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID']))  
        items_id.append(str(js['asin']) )
        ratings.append(float(js['overall']))
        times.append((int(js['unixReviewTime'])))
        # times.append(str_to_days(js['reviewTime']))

data = pd.DataFrame({
    'user_id': pd.Series(users_id),
    'item_id': pd.Series(items_id),
    'ratings': pd.Series(ratings),
    'reviews': pd.Series(reviews),
    'times': pd.Series(times)
})[['user_id', 'item_id', 'ratings', 'reviews', 'times']]


print("============ %s ============" % data_path)
print("================= raw info =============================")
print("#users: %d" % len(data.user_id.unique()))
print("#items: %d" % len(data.item_id.unique()))
print("#actions: %d" % len(data))


# =====================================================================================
# drop duplicated user-item pairs
data.drop_duplicates(subset=['user_id','item_id'], keep='first', inplace=True)

# discard cold-start items
count_i = data.groupby('item_id').user_id.count()
item_keep = count_i[count_i >= filter_min].index
data = data[data['item_id'].isin(item_keep)]

# discard cold-start users
count_u = data.groupby('user_id').item_id.count()
user_keep = count_u[count_u >= filter_min].index
data = data[data['user_id'].isin(user_keep)]

print("========================================================")
print("============== drop some data ==========================")
# output statistical information
n = len(data.user_id.unique())
m = len(data.item_id.unique())
p = len(data)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = data.groupby(['user_id']).item_id.count()
print('sequence length:')
print(count_u.describe())

# =====================================================================================
# sort by time
data.sort_values(by=['times'], kind='mergesort', inplace=True)

# =====================================================================================
# find the time dividing line
time = list(data.times)
div_line = time[int(len(time)*dividing_line)]

df1 = data[data.times<=div_line]
df2 = data[data.times>div_line]
df1_user = set(df1.user_id)
df2_user = set(df2.user_id)

user_m = set(df1_user & df2_user)
user_ml = set(df1_user | user_m)

df_m = data[data.user_id.isin(list(user_m))]
df_ml = data[data.user_id.isin(list(user_ml))]

print("==============================================================")
print("============= find the time dividing line (ml) ===============")
print("#users: %d" % len(user_ml))
print("#items: %d" % len(df_ml.item_id.unique()))
print("#actions: %d" % len(df_ml))
print("density: %.4f" % (len(df_ml)/len(user_ml)/len(df_ml.item_id.unique())))

count_dfml = df_ml.groupby(['user_id']).item_id.count()
print('sequence length:')
print(count_dfml.describe())

# =====================================================================================
# split the data
train_ml = df1[df1['user_id'].isin(list(user_ml))]
test_ml = df2[df2['user_id'].isin(list(user_ml))]

print("========================================================")
print("============== split the data ==========================")
print("#train actions: %d" % train_ml.shape[0])
print("#test actions: %d" % test_ml.shape[0])
print("#split rate: %.4f " % (test_ml.shape[0]/(train_ml.shape[0]+test_ml.shape[0])))

# =====================================================================================
# calculate the length of the sequence
def cal_len(df):
    return len(df)

a = train_ml.groupby('user_id').apply(cal_len)
a = a.reset_index()
a.columns = ['user_id', 'train_len']

b = test_ml.groupby('user_id').apply(cal_len)
b = b.reset_index()
b.columns = ['user_id', 'test_len']

res = pd.merge(a, b, how='inner', on=['user_id'])
# =====================================================================================
# filter the data --> the length of the sequence >= 5
user_l = list(user_ml -user_m)
user_seq5 = list(res[res.train_len>=5].user_id)
user_seq5 = user_l + user_seq5
df_seq5 = data[data.user_id.isin(list(user_seq5))]

# ================================================================================
# split data into test set, valid set and train set

df_train = df_seq5[df_seq5.times<=div_line]
df_test = df_seq5[df_seq5.times>div_line]


train = df_train[df_train['user_id'].isin(list(user_seq5))]
train['flag'] = list(['train']*train.shape[0])

test_valid = df_test[df_test['user_id'].isin(list(user_seq5))]

# split the test dataset
def cal_valid(df):
    return df.head(1)

test_valid = test_valid.sort_values('times')
test_valid['flag'] = test_valid.index

valid = test_valid.groupby('user_id').apply(cal_valid)

test_index = set(test_valid.flag.unique()) - set(valid.flag.unique())
test = test_valid[test_valid['flag'].isin(list(test_index))]

valid['flag'] = list(['valid']*valid.shape[0])
test['flag'] = list(['test']*test.shape[0])

# drop cold-start items in valid set and test set
valid = valid[valid.item_id.isin(train.item_id)]
test = test[test.user_id.isin(valid.user_id) & (
    test.item_id.isin(train.item_id) | test.item_id.isin(valid.item_id))]


# reset the user/item id
df_concat = pd.concat([train,test,valid],axis=0)

le = preprocessing.LabelEncoder()
df_concat['user_id'] = le.fit_transform(df_concat['user_id'])+1
df_concat['item_id'] = le.fit_transform(df_concat['item_id'])+1

train = df_concat[df_concat.flag=='train']
valid = df_concat[df_concat.flag=='valid']
test = df_concat[df_concat.flag=='test']

# print the data static
users_num = len(train.user_id.unique())
items_num = len(train.item_id.unique())
action_num = train.shape[0] + test.shape[0] + valid.shape[0]

print("========================================================")
print("=========== train_len >=5 ================")
print("#users: %d" % users_num)
print("#items: %d" % items_num)
print("#actions: %d" % action_num)
print("density: %.4f" % (action_num/items_num/users_num))
print("#train actions: %d" % train.shape[0])
print("#test&valid users: %d" % len(test_valid.user_id.unique()))
print("#test&valid actions: %d" % (test.shape[0]+valid.shape[0]))
print("#split rate: %.4f " % ((test.shape[0]+valid.shape[0])/(train.shape[0]+ \
                             (test.shape[0]+valid.shape[0]))))
print("#valid actions  %d " % (valid.shape[0]))
print("#test acitons  %d " % ( test.shape[0]))
print("#valid : test  %.4f " % (valid.shape[0]/(test.shape[0]+valid.shape[0])))


# =====================================================================================
# save the item_seq to txt
video_games_train = train.loc[:,['user_id','item_id']]
video_games_train.to_csv(data_directory+category_name+'_train.txt',sep=' ', index=False,header=False)

video_games_test = test.loc[:,['user_id','item_id']]
video_games_test.to_csv(data_directory+category_name+'_test.txt',sep=' ', index=False,header=False)

video_games_valid = valid.loc[:,['user_id','item_id']]
video_games_valid.to_csv(data_directory+category_name+'_valid.txt',sep=' ', index=False,header=False)

# =====================================================================================
# processing the timestamp
def PreprocessData_Games(df):
    # for col in ("user", "item"):
    #     df[col] = df[col].astype(np.int32)
    df['ts'] = pd.to_datetime(df['ts'],unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))
    df['year']-=df['year'].min()
    df['year']/=df['year'].max()
    df['month']/=12
    df['day']/=31
    df['dayofweek']/=7
    df['dayofyear']/=365
    df['week']/=4

    df.fillna(0,inplace=True)

    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows() :
      userid = int(row['user'])
      itemid = int(row['item'])

      year = row['year'] 
      month = row['month'] 
      day = row['day'] 
      dayofweek = row['dayofweek'] 
      dayofyear = row['dayofyear'] 
      week = row['week'] 
      DATEINFO[(userid,itemid)] = [year, month, day, dayofweek, dayofyear, week]

    return df, DATEINFO 

df_concat = pd.concat([train,test,valid],axis=0)
cxt = df_concat.loc[:,['user_id','item_id','times']]
cxt.rename(columns={'user_id':'user','item_id':'item','times':'ts'},inplace=True)

df_time,DATEINFO = PreprocessData_Games(cxt)
with open(data_directory+'CXTDictSasRec_Beauty.dat','wb') as f:
    pickle.dump(DATEINFO, f)

# =====================================================================================
# save the review document
df_review = df_concat.loc[:,['user_id','item_id','reviews']]

with open(data_directory+category_name+'_review.dat','wb') as f:
    pickle.dump(df_review, f)


# # =====================================================================================
# pick the negetive item
sample_pop = False
sample_num = 100
sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

# sample according to popularity
if sample_pop == True:
    sr_item2pop = df_seq5.item_id.value_counts(sort=True, ascending=False)
    arr_item = sr_item2pop.index.values
    arr_pop = sr_item2pop.values

    def get_negative_sample(pos):
        neg_idx = ~np.in1d(arr_item, pos)
        neg_item = arr_item[neg_idx]
        neg_pop = arr_pop[neg_idx]
        neg_pop = neg_pop / neg_pop.sum()

        return np.random.choice(neg_item, size=sample_num, replace=False, p=neg_pop)

    arr_sample = df_negative.user_id.apply(
        lambda x: get_negative_sample(sr_user2items[x])).values

# sample uniformly
else:
    arr_item = df_concat.item_id.unique()
    arr_sample = df_negative.user_id.apply(
        lambda x: np.random.choice(
            arr_item[~np.in1d(arr_item, sr_user2items[x])], size=sample_num, replace=False)).values

# output negative data
df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis='columns')
df_negative.to_csv(data_directory+ "%s_negative.csv"%category_name, header=False, index=False)

