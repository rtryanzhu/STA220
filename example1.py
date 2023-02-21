import pandas as pd
import requests as req
import requests
import json
import time
from tqdm import tqdm 
# progress bar to track
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# set up query
# make a search, look for search.action in network
response = req.post("https://ucannualwage.ucop.edu/wage/search.action", data = {
        "_search": "false", 
        "nd": "",
        "rows": "200",
        "page": '1',
        "sidx": "EAW_LST_NAM",
        "sord": "asc",
        "year": "2021",
        "location": "Davis",
        "firstname": "", 
        "lastname": "",
        "title": "prof",
        "startSal": "",
        "endSal": ""
    })
# for all falcuties, leave names blank

result = json.loads(response.text.replace("\'", "\"")) # decode
int(result["total"])
# get total number of pages 

data = pd.DataFrame()
for i in range(17):
    response = req.post("https://ucannualwage.ucop.edu/wage/search.action", data = {
        "_search": "false", 
        "nd": "",
        "rows": "200",
        "page": i,
        "sidx": "EAW_LST_NAM",
        "sord": "asc",
        "year": "2021",
        "location": "Davis",
        "firstname": "", 
        "lastname": "",
        "title": "prof",
        "startSal": "",
        "endSal": ""
    })
    result = json.loads(response.text.replace("\'", "\"")) # decode
    df = pd.DataFrame([x["cell"] for x in result["rows"]])
    data = data.append(df)
 
privaticed = data[data.FullName.str.contains(r'[*]')] # Any row with asterisk in FullName

index = data[data.FullName.str.contains(r'[*]')]['Index'] # Get row indices of privaticed cases
index_list = index.tolist()
drop_list = [int(i) for i in index_list]
drop_list = [x-1  for x in drop_list] # Python indicing starts from 0


def fetch_new(name_list, start_case,end_case):
    title_list = ['Supervisor','Professor']
    dept = pd.DataFrame()

    for i in tqdm(range(start_case, end_case)):
        name = name_list[i].split()[0] + ' ' + name_list[i].split()[-1]
        
        try:
            response = requests.get("https://org.ucdavis.edu/directory-search/person?", params = {
                "query": name
            }, timeout = 2).json()
                                
            if len(response)==0: # When response is empty
                res_df = pd.DataFrame({'name':[name],'department':[None]})
                dept = dept.append(res_df)
                
            else:
                res_df = pd.DataFrame(response)
                
                if res_df.shape[0] > 1: # In case of A-Z and A-B-Z
                    for i in range(res_df.shape[0]):
                         if (((res_df.iloc[i,0].split()[0].upper() +' ' +res_df.iloc[i,0].split()[-1].upper()) == name)
                            and res_df.iloc[i,11] is not None
                            and any(x in res_df.iloc[i,11] for x in title_list)):
                                result = pd.DataFrame({'name':[name],'department':[res_df.iloc[i,1]]})
                                dept = dept.append(result)
                else:
                    result = pd.DataFrame({'name':[name],'department':[res_df.iloc[0,1]]})
                    dept = dept.append(result)
           
        except :
            pass # Skip any person with any error
            
    return (dept)
  
 
dept_full = fetch_new(full_name_list_new,0,len(full_name_list_new))

to_merge_1 = data_rm_mid_clean.iloc[:,[3,4,6,10]]
to_merge_1.head()

x = pd.merge(to_merge_1,dept_full, how='inner',on=['FullName'])


def find_difference_in_columns(col_long, col2_short):
    missing_names = []
    n = len(col_long)
    for i in range(n):
        if col_long[i] in col2_short.values:
            pass
        else:
            missing_names.append(col_long[i])
    return missing_names

ucdhs_list = ['UCDHS','MED','Med']
for i in range(full_table.shape[0]):
    if any(x in full_table.iloc[i,4] for x in ucdhs_list):
        full_table.iloc[i,4] = 'UCDHS'
        
median_pay_by_dept = full_table.groupby(['Department'])['Gross'].median()
median_pay_by_dept.sort_values(by=['Median of Gross Pay of Professors'],ascending=False)

median_pay_by_dept = pd.DataFrame(median_pay_by_dept)
