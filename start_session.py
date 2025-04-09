from fedn import APIClient
from server_functions import ServerFunctions

client = APIClient(host="api.fedn.scaleoutsystems.com/projectnumber-hkb-fedn-reducer", 
                   token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ2NjA2MDg1LCJpYXQiOjE3NDQwMTQwODUsImp0aSI6ImNhNmY2MTRlYmQ5YzQyYjA4YmM0YzlhMmE4ZDVhMDVjIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJwcm9qZWN0bnVtYmVyLWhrYiJ9.FvZUE9Ntg4TQja0zpKdM5QCFj9_hyyCEhdu1Tz5-7ws"
                   ,secure=True)

client.set_active_package("client/package.tgz", helper="numpyhelper", name="battery-client")
client.set_active_model("seed.npz")


res = client.start_session(name = "temp_only_cluster",rounds=2,server_functions=ServerFunctions)
print(f"session started successfully")
print(res)

print(ServerFunctions.aggregate())
print(ServerFunctions.client_selection())


