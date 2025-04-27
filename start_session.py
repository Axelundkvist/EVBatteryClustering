from fedn import APIClient
from server_functions import ServerFunctions
from datetime import datetime


client = APIClient(host="api.fedn.scaleoutsystems.com/sfapril-jgf-fedn-reducer", 
                   token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ4MzU3NDM2LCJpYXQiOjE3NDU3NjU0MzYsImp0aSI6IjY2NDFkOTU4MzM3NTQ4ZWY4ZTE5OTE1NWFkMzFlMTAzIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJzZmFwcmlsLWpnZiJ9.rUFRhJRN_5QZcc9790YzcQP1VmZTAAyxugnuH1TZJig"
                   ,secure=True)

session_name = datetime.now().strftime("session-%Y%m%d_%H%M%S")




#client = APIClient(host="localhost", port=8092)

#client.set_active_package("client/package.tgz", helper="numpyhelper", name="battery-client")
#client.set_active_model("seed.npz")
# lägg in tidsberoende variabel för sessions namnet

res = client.start_session(name = session_name,rounds=3,server_functions=ServerFunctions)
#print(f"session started successfully")
print(res)


  