#!/bin/bash

# datafilsvägarna
datafiles=(
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_54309277.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_54318476.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_54357508.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_57578092.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_57578142.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_57583126.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_57584900.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_57615043.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_57615431.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/IndianFleetData/BatteryFleetData/device_57622858.csv"
)


new1MinAggdatafiles=(
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle1_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle2_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle3_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle4_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle5_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle6_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle7_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle8_1min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle9_1min.csv"
)

new5MinAggdatafiles=(
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle1_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle2_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle3_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle4_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle5_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle6_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle7_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle8_5min.csv"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/DataLoadAndConcatination/vehicle9_5min.csv"
)

fieldVehicleDatafiles_folders=(
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle1"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle2"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle3"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle4"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle5"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle6"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle7"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle8"
"/Users/Axel/Documents/Master/MasterThesis/DataSets/FieldVehicleData/Christian_validation_data/vehicle9"
)


michgan_folder=(
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_1.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_2.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_3.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_4.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_5.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_6.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_7.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_8.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_9.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_10.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_11.csv"
    "/Users/Axel/Documents/Master/MasterThesis/DataSets/Michigan/axel/cell_12.csv"
)

nasa_folder=(
    "/Users/Axel/fedn/examples/server-functions/new_data/B0005_part1.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0005_part2.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0006_part1.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0006_part2.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0007_part1.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0007_part2.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0046.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0047.csv"
    "/Users/Axel/fedn/examples/server-functions/new_data/B0048.csv"
)




# colaborated session from Viktor
# tokens=(
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMDcxLCJpYXQiOjE3NDQ4OTEwNzEsImp0aSI6ImE3NjE2MjM5ZjQ5YjRlNmNiZmQxMzI4NjYxOGYxYjg3IiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.5v4muxzjIothujbUb2LC5PTha3yWMUj7haVNFEgicVs"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMTQ1LCJpYXQiOjE3NDQ4OTExNDUsImp0aSI6ImYxY2NjOTU1YjEyMDRmNjA5OTk2NjNkMWFmZTk5MTNlIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.yybXaiMLQBXJY7gnzmK2AHa6me_qIzNIEsD16R3ujfs"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMTU2LCJpYXQiOjE3NDQ4OTExNTYsImp0aSI6IjkwNzZmNDhkZWUzOTRjM2NiYWVkODhhN2I1NzYwN2ZhIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.farI4lM6zcxBRTY5vey7H43IhsxvJp2RBeWXCxqhxIc"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMTc2LCJpYXQiOjE3NDQ4OTExNzYsImp0aSI6IjU1NGVhMDBiMmJjYTQxZGZhMWY3MDZmNTBlNWM0MzhjIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.mALUsOVJnVwmdG2lYpwM5AT0oxkoj3Erknf6A64nXE0"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMTg1LCJpYXQiOjE3NDQ4OTExODUsImp0aSI6Ijc1NzkzZmNmNzViMzRlYTE5NTc5ZmMwYWQ0MjdjN2E2IiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.-AZgjkwVVh6xC-VPIG0gKx8a3leOPOP0bXpAuOVhB3k"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMjE3LCJpYXQiOjE3NDQ4OTEyMTcsImp0aSI6IjU1ZTkzNjJlODcwMjQ1YTU4OTAzMzY3YjY5ZjdiMmVjIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.MiNPBFMl7LdMpXEfmRtUjeLWtkitbewKDTt7Y96OEJg"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMjI5LCJpYXQiOjE3NDQ4OTEyMjksImp0aSI6Ijk1Y2MyOTczZmIyMzRiOWFhMjk4MjNhMjUxOTc0OThmIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.ZstJ2oPfou_i84otNaC72Hu9sB1c1GlgiUYrPU8B_k0"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMjM2LCJpYXQiOjE3NDQ4OTEyMzYsImp0aSI6IjlhYWYwYzMzOTljMzRmZmFiYmQ2ZTNjOTZhMzE0MzFhIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.SH6_Wp_RihwGpbHK0vZeXSpe9OT6dbA7DmuVcQacwII"
#     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMjQ3LCJpYXQiOjE3NDQ4OTEyNDcsImp0aSI6Ijg1NmJmYWU4OTY4NzQyY2E5ZDUyNmMxZmI1YjY5NmU3IiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoiYXhlbC14dGIifQ.4i-Ozii41m0pIxa8VAvO1LXwmTlNBdZanDZamjvbiDQ"
# )

tokens=(
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NTA0LCJpYXQiOjE3NDU3NjU1MDQsImp0aSI6IjgwZTliMTlhZDdhNzQ5MTBhM2I0ZjhiNmQyZjM4ZDVjIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.9KQg5ES27Is1tap3LmKNiQtR0-l-q5_fElpqPbdxDp4"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NTMzLCJpYXQiOjE3NDU3NjU1MzMsImp0aSI6ImM4ZWM2NDU1OTRhNjQwNWM5MmMwYWY5MmYzMTc3Y2ZmIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.xvaY82Amcsl1ViqHDQ0cGQsOtRGgK9-q5aZD_q7nf4M"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NTQ1LCJpYXQiOjE3NDU3NjU1NDUsImp0aSI6IjA1MTEyYzY2Y2RlZjQzZGI5N2NlYTAzNjNiNTNjNmZlIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.tQ1ztqz-YyxuGO_ZEtqWGv49Smc2tQXC6qQWreYyjII"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NTU3LCJpYXQiOjE3NDU3NjU1NTcsImp0aSI6IjI2NzBiZTY0MDY4NjQ3YzM5ZWEzZTYwNGNiZWJjMzQyIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.zCME909dGmrdF2lSAxMxgk-1qPwzA6glelNe4UOem1I"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NTY2LCJpYXQiOjE3NDU3NjU1NjYsImp0aSI6IjA2NmZiYzQzZTlmMTQwNTFiMTFkYmFjN2RlMmMzZjNmIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.F5_JJ6-5XmV0XzrNZdHZ73ROFuKSqiDhURITOAxKsSQ"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NTg0LCJpYXQiOjE3NDU3NjU1ODQsImp0aSI6ImVmZTNhY2MxZGUwYTRmZTJhZGEzYzE2NDQ0OWRiZDk3IiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.PG1cZ9_a54N2ZBf4D2J8krD8YEpqZbc1x7xROSDwXjk"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NTkzLCJpYXQiOjE3NDU3NjU1OTMsImp0aSI6IjQ0MTVjOTYwOTMwYTQ0YmE4YzBiMTYxMDEzZGM1NTY3IiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.SNuBAUSYblvMVLGDAG90DttetKSL-ZiFwMr8KwhadO0"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NjE1LCJpYXQiOjE3NDU3NjU2MTUsImp0aSI6ImIyMzgxM2UxMGEwNzQxZjBhYzVjOWJjY2QzMGQ0NjVlIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.mGeS-Qyv-DEq5-frYwnOuokFXpXtfLE5zHzlMGxjjbo"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NjM3LCJpYXQiOjE3NDU3NjU2MzcsImp0aSI6ImY5N2MxYjE1NTVkZTQwMzBiNWNlYzA3MGE5ODRlMzliIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.i2oqZblpm-V4ylFyDpB3yQdVto6rtiksPjY0rlSlrPA"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NjU3LCJpYXQiOjE3NDU3NjU2NTcsImp0aSI6ImRiNGRkYzI3NTIyYTQ4ZjQ4YmQ0MTA5NDI2ZTU1N2Q5IiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2ZhcHJpbC1qZ2YifQ.oGTL7N-2m-KMdOOZl8snpKnttnuh8nZjYPgGgiftz5w"

)

client_id=(
"client-Asterix-1"
"client-Obelix-2"
"client-Idifix-3"  
"client-Miraculix-4"
"Jimmy-Neutron-5"
"Spongebob-6"
"Patrick-7"
"Trump-8"
"Joe-Biden-9"
)

#eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ2MjU5Mzg3LCJpYXQiOjE3NDM2NjczODcsImp0aSI6ImZmNzkwY2Y2NGExMTRlNjVhMjI0N2IxZTdhNTkzNDBhIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoic2VydmVyZnVuY3Rpb25tb2RpZmNhdGlvbi1udHAifQ.yW121LJH4IGn70w2ED5qtYCx4m5fp_ioQkyOz0dgJ98 --client-id 8ba3ae09-60b0-46a2-9a87-507a4d83948d



#export FEDN_PACKAGE_EXTRACT_DIR=client/package
#export FEDN_PACKAGE_EXTRACT_DIR=package 
#export CHUNK_SIZE=80000 # this is the default chunk size, change if file size changes
for i in {0..8}
do
    echo "Starting client $i with dataset ${nasa_folder[$i]}"
    
    export FEDN_DATA_PATH=${nasa_folder[$i]}
    #fedn client start --api-url http://localhost:8092 --local-package --client-id ${client_id[$i]}
    #fedn client start --api-url api.fedn.scaleoutsystems.com/axel-xtb-fedn-reducer --token ${tokens[$i]} --client-id ${client_id[$i]} &
    fedn client start --api-url api.fedn.scaleoutsystems.com/sfapril-jgf-fedn-reducer --token ${tokens[$i]} --client-id ${client_id[$i]} &
    sleep 1
done
echo "All clients started!"
