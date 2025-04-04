from fedn import APIClient
from server_functions import ServerFunctions

client = APIClient(host="api.fedn.scaleoutsystems.com/serverfunctionmodifcation-ntp-fedn-reducer", 
                   token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ2MjU4NDQ1LCJpYXQiOjE3NDM2NjY0NDUsImp0aSI6ImVkNmJiMDRlMzBiMjRmOTJhYTMxZTE2YmIzMDhkM2JlIiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJzZXJ2ZXJmdW5jdGlvbm1vZGlmY2F0aW9uLW50cCJ9.Q71vacRXLFieX2Xo_lyUikY6pE8SWWvwvQW3DBcCAog"
                   ,secure=True)

client.set_active_package("client/package.tgz", helper="numpyhelper", name="battery-client")
client.set_active_model("seed.npz")


res = client.start_session(rounds=2,server_functions=ServerFunctions)
print(f"session started successfully")
print(res)

