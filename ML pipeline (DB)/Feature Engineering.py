#1.Import Libraries & Load Code:
import pandas as pd
import numpy as np
df=pd.read_csv('')
df

df['SR']=pd.to_numeric(df['SR'],errors='coerce')
print(df['SR'].dtypes)

#2.Create New Features:

#is_out--
df['is_out']=df['Out/Not_out'].apply(lambda x:1 if x=='Out' else 0)
print(df['is_out'])

#Batting Average:
df['batting_average']=df['runs']/df['is_out'].replace(0,np.nan)
print(df['batting_average'])

#boundary Runs:
df['boundary_runs']=df['4s']*4+df['6s']*6
df

#Boundary Percentage:
df['boundary_percentage']=np.where(df['runs']>0,(df['boundary_runs']/df['runs'])*100,0)
df

#Dot Balls:
df['dot_balls']=df['balls']-(df['4s']+df['6s']+((df['runs']-df['boundary_runs'])//1))
df
#Balls per boundary:
df['balls_per_boundary']=np.where((df['4s']+df['6s'])>0,df['balls']/(df['4s']+df['6s']),df['balls']))
df
#Runs per Ball:
df['runs_per_ball']=np.where(df['balls']>0,df['runs']/df['balls'],0)
df
#Power Hitting Index:
df['power_hitting_index']=np.where(df['runs']>0,(df['6s']*6)/df['runs'],0)
df

#3.Aggregated Player Stats
player_stats = df.groupby("batsmanName").agg({
    "runs":"sum",
    "balls":"sum",
    "4s":"sum",
    "6s":"sum",
    "is_out":"sum"
}).reset_index()

player_stats["batting_average"] = player_stats["runs"] / player_stats["is_out"].replace(0, np.nan)
player_stats["strike_rate"] = np.where(player_stats["balls"] > 0, (player_stats["runs"]/player_stats["balls"])*100, 0)
player_stats["boundary_runs"] = player_stats["4s"]*4 + player_stats["6s"]*6
player_stats["boundary_percentage"] = np.where(player_stats["runs"]>0,
                                               (player_stats["boundary_runs"]/player_stats["runs"])*100,
                                               0)
print("\nPlayer Level Aggregated Stats:\n", player_stats.head())

#short summary:----------
# Example: Create interaction terms, normalize numerical columns
scaler= StandardScaler()
# Step 1: Scale all features except the last column
scaled_features= scaler.fit_transform(df.drop(df.columns[-1],axis=1))
# Step 2: Put scaled features back into a DataFrame with proper column names
x=pd.DataFrame(scaled_features,columns=df.columns[:-1])
# Step 3: Define target variable as the last column
y=df[df.columns[-1]]

✅ Extra Note:
তুমি চাইলে y-কে DataFrame হিসেবেও নিতে পারো, কোনো সমস্যা নেই:
y = df[[df.columns[-1]]]   # double bracket দিলে DataFrame
তাহলে y হবে DataFrame (2D)
আর df[df.columns[-1]] দিলে হবে Series (1D)

📌 Summary
x = multiple features → অবশ্যই DataFrame দরকার।
y = single target column → Series রাখাই সহজ, তাই constructor লাগেনি।
চাইলে y কে DataFrame বানানো যায় double bracket দিয়ে।

🔹 np.where() এর structure
np.where(condition, value_if_true, value_if_false)
যদি condition সত্যি হয় → value_if_true ব্যবহার করবে।
যদি condition মিথ্যা হয় → value_if_false ব্যবহার করবে।

🔹 আমাদের কোডে
Condition: df['runs'] > 0
True case: (df['boundary_runs']/df['runs'])*100 → অর্থাৎ রান হলে boundary percentage বের করা হবে।
False case: 0 → যদি রান = 0 হয়, তখন percentage বের করলে divide by zero হয়ে error আসবে। তাই safe value হিসাবে 0 বসানো হয়েছে।

groupby() → একই ব্যাটসম্যানের সব খেলা একত্রিত করে।
.agg() → গ্রুপের উপর নির্দিষ্ট ফাংশন (sum) প্রয়োগ করে।
reset_index() → গ্রুপের নামকে কলাম হিসেবে ফেরত আনে।
নতুন কলামগুলো → ব্যাটিং এভারেজ, স্ট্রাইক রেট, boundary_runs, boundary_percentage হিসাব করে।



ধরো আমাদের কাছে একটি DataFrame আছে:

import pandas as pd

data = {
    "batsmanName": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
    "runs": [10, 20, 30, 40, 50],
    "balls": [5, 10, 15, 20, 25]
}

df = pd.DataFrame(data)
print(df)


Output:

  batsmanName  runs  balls
0       Alice    10      5
1         Bob    20     10
2       Alice    30     15
3         Bob    40     20
4     Charlie    50     25

১. groupby ছাড়া reset_index()
grouped = df.groupby("batsmanName").sum()
print(grouped)


Output:

           runs  balls
batsmanName            
Alice        40     20
Bob          60     30
Charlie      50     25


লক্ষ্য করো, batsmanName এখন index।

তুমি যদি grouped["batsmanName"] করতে চাও → error দেবে।

২. reset_index() ব্যবহার করলে
grouped_reset = grouped.reset_index()
print(grouped_reset)


Output:

  batsmanName  runs  balls
0       Alice    40     20
1         Bob    60     30
2     Charlie    50     25


এখন batsmanName আবার সাধারণ কলাম হিসেবে এসেছে।

তুমি চাইলে এই কলামে আবার filter, sort, বা merge করতে পারবে।
groupby() করলে grouping column index হয়ে যায়।
reset_index() দিলে সেটা আবার সাধারণ column হয়।


ধরা যাক, আমাদের কাছে একটি DataFrame df আছে। ধরো df এর structure এরকম:

   runs  balls  4s  6s  is_out
0    10      5   1   0       1
1    20     10   2   1       1
2    15      8   1   0       0


এখানে ধরো আমরা is_out কে predict করতে চাই। তাহলে আমাদের target column হবে:

target = "is_out"

১. X = df.drop(target, axis=1)

df.drop(target, axis=1) মানে হলো df থেকে is_out column মুছে ফেলা, এবং বাকি সব column রেখে feature set তৈরি করা।

এখানে axis=1 মানে column drop করা (axis=0 হলে row drop)।

X = df.drop(target, axis=1)
print(X)


Output:

   runs  balls  4s  6s
0    10      5   1   0
1    20     10   2   1
2    15      8   1   0


এটাই আমাদের input features বা X।

২. y = df[target]

y হচ্ছে আমাদের target variable বা যা predict করতে চাই।

এখানে আমরা is_out predict করব।

y = df[target]
print(y)


Output:

0    1
1    1
2    0
Name: is_out, dtype: int64


এটাই আমাদের output label বা y।