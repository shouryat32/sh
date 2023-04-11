import ijson
import json
import tarfile
from collections import defaultdict
import pandas as pd
from mpi4py import MPI
import sys


# Define the MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the file names
file_name=sys.argv[1]
print(file_name)
tar_filename_twitter = 'twitter-data-small.json.tar.gz'
json_filename_twitter = 'twitter-data-small.json'
tar_filename_location = 'sal.json.tar.gz'
json_filename_location = 'sal.json'
# Extract the JSON file from the compressed tar file
if rank == 0:
    with tarfile.open(tar_filename_twitter, 'r:gz') as tar:
        tar.extract(json_filename_twitter)
with tarfile.open(tar_filename_location, 'r:gz') as tar:
    tar.extract(json_filename_location)   
# Wait for rank 0 to extract the file before continuing
comm.Barrier()
# Read the JSON data in parallel
chunk_size = 10000
chunks = defaultdict(list)
with open(json_filename_twitter, "r", encoding="utf8") as file:
    for i, chunk in enumerate(ijson.items(file, "item")):
        # Distribute the chunks across processes
        if i % size == rank:
            chunks[i // chunk_size].append(chunk)
# Concatenate the data into a Pandas DataFrame
df = pd.DataFrame()
for chunk in chunks.values():
    df_chunk = pd.json_normalize(chunk)
    df = pd.concat([df, df_chunk], ignore_index=True)
# Select the columns of interest
selected_cols = ['data.author_id', 'includes.places']
df_selected = df.loc[:, selected_cols]
df_selected.rename(columns={'includes.places':'places' ,'data.author_id':'author_id'},inplace=True)
df_selected['places']=df_selected['places'].astype(str)
df_selected['places']=df_selected['places'].apply(lambda y:((y.split(",")[0]).split(":")[1]).replace("'",""))
with open(json_filename_location, 'r',encoding='utf-8') as g:
    loc = json.load(g)
#Select gcc of all greater city areas
final_loc=defaultdict(list)
for x,y in loc.items():
    if(y['gcc'][1] =="g"):
        final_loc[y['gcc']].append(x)
# match full_name to greater cities 
def matching(text1):
    max_key=""
    m_string=""
    c2=""
    for key,value in final_loc.items():
        for doc2 in value:
            if(text1 in doc2):
                    max_key=key
                    m_string=doc2
    return max_key
location_count=defaultdict(int)
greater_area_count=defaultdict(str)
for row in df_selected.itertuples(index=False):
        area_belong=matching(row.places.lower())
        location_count[area_belong]+=1
        oldval=str(greater_area_count.get(row.author_id,''))
        greater_area_count[row.author_id]+=","+area_belong if greater_area_count[row.author_id] else area_belong
print("Greater Capital City \t Number of Tweets Made")
comm.Barrier()
# Count the number of tweets made by each author
value_counts = dict(df_selected["author_id"].value_counts())
# Gather the results to the root process
results_taks2=comm.gather(location_count,root=0)
results = comm.gather(value_counts, root=0)
results_task3=comm.gather(dict(greater_area_count),root=0)
# Print the results on the root process
if rank == 0:
    #for task 1
    my_dict_task1={} 
    for i, value_counts in enumerate(results): 
        for j, (key, val) in enumerate(value_counts.items()): 
            if(key not in my_dict_task1 ):
                my_dict_task1[key]=val
            elif (key in my_dict_task1):
                my_dict_task1[key]+=val
    my_dict_task1 = dict(sorted(my_dict_task1.items(), key=lambda x: x[1], reverse=True))
    i=1
    print("Rank \t Author Id  \t\t Number of Tweets Made")
    for key,val in my_dict_task1.items():
        if(i<=10):
            print(f"{i:<9}{key:<24}{val}")
            i=i+1
        else:
            break
    #for task 2
    my_dict_task2={} 
    for i, value_counts in enumerate(results_taks2): 
        for j,(key, val) in enumerate(value_counts.items()): 
            if key!='':
                if(key not in my_dict_task2 ):
                    my_dict_task2[key]=val
                elif (key in my_dict_task2):
                    my_dict_task2[key]+=val
    my_dict_task2 = dict(sorted(my_dict_task2.items(), key=lambda x: x[1], reverse=True))
    print("Greater Capital City \t Number of Tweets Made")
    for key,val in my_dict_task2.items():
        print(f"{key:<18}","\t",f"{val:<18}")
    #for task 3
    my_dict_task3={}
    for i, value_counts in enumerate(results_task3): 
        for j, (key, val) in enumerate(value_counts.items()): 
            if key!='':
                if key not in my_dict_task3:
                    my_dict_task3[key]=val
                elif (key in my_dict_task3):
                    my_dict_task3[key]+=val
    final_val=defaultdict(dict)
    for key,val in my_dict_task3.items():
        temp=val.split(",")
        for x in temp:
            if(x!=""):
                final_val[key][x]=temp.count(x)
    final_val=dict( sorted(final_val.items(), key=lambda x: len(x[1])))
    print("Rank \t Author  \t  Id Number of Unique City Locations and #Tweet")
    count=1
    for key,value in final_val.items():
        if count<=10:
            y=', '.join([f'#{v}{k[1:]}' for k,v in value.items()] )
            print(f"{count:<6}\t{key:<18}\t{len(value)}(#{sum(value.values())} tweets - {y})")
            count+=1
        else:
            break