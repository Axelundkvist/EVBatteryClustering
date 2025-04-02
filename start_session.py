from fedn import APIClient
from server_functions import ServerFunctions

client = APIClient(host="api.fedn.scaleoutsystems.com/evbatteryclusteringprojectnewattempt-zhk-fedn-reducer", 
                   token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQ2MTc5ODEwLCJpYXQiOjE3NDM1ODc4MTAsImp0aSI6IjI4YzhhZWZmOGZiZTQ1NjE5YWNkNDMwNDBmMmI0ZTc4IiwidXNlcl9pZCI6MTIxMywiY3JlYXRvciI6ImF4ZWxwMDBAaG90bWFpbC5jb20iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJldmJhdHRlcnljbHVzdGVyaW5ncHJvamVjdG5ld2F0dGVtcHQtemhrIn0.Ryy1AgpiejCJdzyn2iDauSwNM9OVbDMs8n10uCVj3K0", secure=True)

client.set_active_package("client/package.tgz", helper="numpyhelper", name="battery-client")
client.set_active_model("seed.npz")


res = client.start_session(rounds=2,server_functions=ServerFunctions)
print(f"session started successfully")
print(res)


