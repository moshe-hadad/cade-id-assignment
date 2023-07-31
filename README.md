# Project Title

Case ID to Activity Assignment, Under Interleaved Network Traffic Data

## Description

This project is part of the Data Science Lab course in the University of Haifa.  
The aim of this project is to present a method for assigning a process instance case id to specific packets in 
network traffic data which feature multiple process instance executions.  
The method steps are:  
1) Identify case id attributes on the training data
2) Collect and cluster attributes' values from the network data (interleaved data) and addign each group with a number 
as a case id
3) Assign a case id number to packets containing the suitable values from the group
4) Evaluate clustering using clustering metrics

The network traffic data set are taken from this repo : https://github.com/HaifaUniversityBPM/traffic-data-to-event-log

## Getting Started
To run this project install pyton 3.1 and all dependencies from requirements.txt

The main module is main.py it contains all the steps of the project
1) Loading the data
2) Pre-processing 
3) Feature Engineering
4) Feature Selection
5) Data imputing 
6) Clustering 
7) Case id assignment 
8) Evaluation  

The main module will perform all steps above on the data set contained in the data folder.
After Pre-processing, Feature Engineering and Data imputing the resulted data sets are saved in the processed_data 
folder  
After the Case id assignment step the resulted data set is saved as final_results.csv under the processed_data folder    
After all steps are done the evaluation are printed to the console in the form of : 
Rand Score : 0.896  
Homogeneity : 0.6666476220992672  
Completeness : 0.4883063207156309  
V_measure : 0.5637077557702908  


There are three flags above the main function which controls the steps 

PRE_PROCESS - Controls if to run Pre-processing step  
FEATURE_ENGINEERING - Controls if to run the feature engineering step  
IMPUTING - Controls if to run the imputing step  
If the above flags are False, the program will load previously processes data from the processes_data folder and 
perform the next steps.  

### Executing program
Run the main.ipynb notebook or run the main.py module.

### Project structure
- data - Contains the original network traffic data. Isolated and Interleaved data sets  
- data_for_tests - Contain small sample of data for tests
- notebooks - Contain jupyter notebooks which were used for exploring and understanding the data
  - main.ipynb - Contain the main steps to execute, same as main.py but with explenations.
- process_data - Contains the data sets after each step of processing. 
  - benchmark.csv - benchmark containing the results of the original method
  - clusters.csv - the results of the last run of the clustering step
  - correlation.csv - contains the results of the last correlation step
  - All csv with dtype prefix are for loading the data sets with the correct data types (handled by utilities.py)
  - final_results.csv - the final results of running main after all the steps are run.
  - interleaved_df_engineered.csv - the interleaved data set after the feature engineering step
  - interleaved_df_imputed.csv - the interleaved data set after the imputing step
  - interleaved_df_processed.csv - the interleaved data set after the pre-processing step
  - isolated_df_engineered.csv - the isolated data set after the feature engineering step
  - isolated_df_processed.csv - the isolated data set after the pre-processing step
- src
  - case_id_assignment
    - assignment.py - Contain the functions for the case id assignment step
    - clustering.py - Contains the functions for the clustering step
    - evaluation.py - Contains the function for the evaluation metrics
    - feature_engineering.py - Contains the functions for feature engineering
    - httputil.py - Contains utilities to parse HTTP and HTML
    - imputing.py - Contains function for the imputing step
    - main.py - The main module to run. Contains all the steps for this project
    - sqlutil.py - Contains utilities to parse SQL queries
    - utilities.py - Contain general utilities function

#### Schema:
All files share the basic shame schema (with additional columns as a result of the feature engineering step)  
The files contains the following schema:

| Field name      | Field Description                                                                                                                                                                                                                                                                                                     |
| ----------- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|FileName | Name of the recording file this data is originated from.                                                                                                                                                                                                                                                              |
|BusinessActivity | Contains the name of the business process activity. This column contains data only for the training data set, the evaluation data set has the value of the business process since the activity is unknown.                                                                                                            |
|InstanceNumber | Contains the case id of the business process (instance number). This column contains data only for the training data set, the evaluation data set contains garbage value in this column since the case id is unknown.                                                                                                 |
|sniff_time | The recording time (used as a time stamp)                                                                                                                                                                                                                                                                             |
| frame.number | The order of the packets in the data (added by wireshark)                                                                                                                                                                                                                                                             |
| synthetic_sniff_time	 | Since the interval between packets is very low (milliseconds) and most of the analytics tools are sensitive to micro seconds granularity, we manipulated the sniffing time to show difference between packets even in macro seconds granularity, while keeping the original relative time between packets.            | 
| synthetic_sniff_time_str | A string representation of synthetic_sniff_time.                                                                                                                                                                                                                                                                      |
| session_generalized | A string combination of the actors participated in the session.                                                                                                                                                                                                                                                       |
| HighestLayerProtocol | Contain the protocol name of the highest layer e.g HTTP, PGSQL etc.                                                                                                                                                                                                                                                   |
| MessageType_WithRole | Contains a string representation of the source machine role, the destination machine role and the type of message they exchanged. e.g. End Point (HR Manager)->Odoo Application:[HttpRequest:POST /xmlrpc/2/common HTTP/1.1\r\n]                                                                                      | 
| MessageType | The type of the message exchanged in this packet. e.g. PgsqlRequest:Simple query                                                                                                                                                                                                                                      |
| MessageAttributes	 | The attributes transferred in this packet. In the feature engineering phase, this column is parsed to extract additional attributes. **MessageAttributes column was omitted from the recognition_results data sets"                                                                                                   |
| query_type | Contais the type of query (INSERT, UPDATE) for the PGSQL packets. Empty for HTTP packets.                                                                                                                                                                                                                             |
| session_class	 | Protocol name (http, pgsql, smtp) exist from R2 and above.                                                                                                                                                                                                                                                            |
| filter_flag | Left over from the filtering stages contains the value True from R2 and above.                                                                                                                                                                                                                                        |
| query	 | For PGSQL packets, contains the query for the data base. HTTP packets has no value. applicable for R2 and above.                                                                                                                                                                                                      |
| tables | For PGSQL packets, contains the list of tables impacted by this packets. exist from R2 and above.                                                                                                                                                                                                                     |
| event	 | A concise string representation of the message carried out in this packet. e.g. PgsqlRequest:Simple query:SELECT:['pg_database']. Exist from R2 and above.                                                                                                                                                            |
| event_with_roles	 | A concise str representation of the source, destination, message and roles in this packet. e.g. Odoo Application->db Server/Mail Server: [PgsqlRequest:Simple query:SELECT:['base_registry_signaling', 'base_cache_signaling']] .Exist from R2 and above. **This column was hevely used in the classification phase** |
| noise_event | Left over from the filtering stages, a flag to mark packets as noise.                                                                                                                                                                                                                                                 |


##### parsing PGSQL
At the feature engineering stage we extract over 100 attributes from packets representing PGSQL and HTTP. The list of attributes is too long to be presented here but we can give some examples:
For a PGSQL packet containing the following query 
```    
INSERT INTO "res_users_log" ("id", "create_uid", "create_date", "write_uid", "write_date") VALUES (nextval('res_users_log_id_seq'), 6, (now() at time zone 'UTC'), 6, (now() at time zone 'UTC')) RETURNING id
```

We extracted the following columns and values :

* id = nextval('res_users_log_id_seq')
* create_uid =   6  
* create_date = now() at time zone 'UTC'
* write_uid = 6
* write_date =  now() at time zone 'UTC'

##### parsing tables column
Additional feature are extracted from the tables columns if the query column contains an id.
e.g. if the query is an update query
```    
UPDATE  "mail_message" SET "create_uid=2", "create_date=8", "write_uid=9", "write_date=8" WHERE ID=2
```
We extract the following feature mail_message_id = 2 

##### parsing HTTP
We did the same for the message attributes column from the HTTP packets. When it contained request/response parameters, we parsed the message and extracted parameters and their values as additional columns. Not all parameters were selected.
As an example, for an HTTP packet containing the following request parameter xml:
    
    <methodCall>
        <methodName>execute_kw
            <params>
                <param><value>odoo01</param>
                <param><value>6</param>
                <param><value>123456789</param>
                <param><value>hr.applicant</param>
                <param><value>create</param>
                <param><value><value>
                    <member><name>name<value>Head of Research</member>
                    <member><name>partner_name<value>Daniel Duncan</member>
                    <member><name>email_from<value>daniel.duncan@gmail.com</member>
                    <member><name>job_id<value>4</member></struct></data></param>
            </params>
    </methodCall>

We extracted the following columns and values :

* request_method_call = execute_kw
* file_data =   ['execute_kw', 'odoo01', '6', '123456789', 'hr.applicant', 'create', 'name', 'Head of Research', 'partner_name', 'Daniel Duncan', 'email_from', 'daniel.duncan@gmail.com', 'job_id', '4']  
* selective_filter_data = hr.applicant_create

## Authors

Moshe Hadad

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg\]

