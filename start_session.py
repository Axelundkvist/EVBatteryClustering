from fedn import APIClient
from server_functions import ServerFunctions

client = APIClient(host="api.fedn.scaleoutsystems.com/sessiontesting-kcn-fedn-reducer", 
                   token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ2OTUzMTY1LCJpYXQiOjE3NDQzNjExNjUsImp0aSI6ImE3NDk2OGI5NGU4NjRkZTA5M2QwMTlhODkwMjZjNzcyIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJzZXNzaW9udGVzdGluZy1rY24ifQ.L5bQPxwC93IAmncvzp8sKdmCeNNOoOdTfy88DUo2o0o"
                   ,secure=True)

client.set_active_package("client/package.tgz", helper="numpyhelper", name="battery-client")
client.set_active_model("seed.npz")


res = client.start_session(name = "MassivelyRefinedClusterAlg",rounds=2,server_functions=ServerFunctions)
print(f"session started successfully")
print(res)


