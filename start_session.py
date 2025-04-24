from fedn import APIClient
from server_functions import ServerFunctions

client = APIClient(host="api.fedn.scaleoutsystems.com/axel-xtb-fedn-reducer", 
                   token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ3NDgzMDM5LCJpYXQiOjE3NDQ4OTEwMzksImp0aSI6IjhkM2Q5NTI5NTM3ZTRiMjhiODM0YzkzM2VhMTJkZTVjIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJheGVsLXh0YiJ9.BGUqYY8Kg8SFl_ydGaZRw-2h-iiFm6DbRb6dJjq72DM"
                   ,secure=True)





#client = APIClient(host="localhost", port=8092)

#client.set_active_package("client/package.tgz", helper="numpyhelper", name="battery-client")
#client.set_active_model("seed.npz")

res = client.start_session(name = "session-1",rounds=5,server_functions=ServerFunctions)
#print(f"session started successfully")
print(res)


  