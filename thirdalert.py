from trycourier import Courier

client = Courier(auth_token="pk_prod_S419EP2DZKMP1GGQKG3MGKK97G5M")
#sarbjeet@pu.ac.in
resp = client.send_message(
  message={
    "to": {
      "email": "nitish2mahajan@gmail.com",
    },
    "content": {
      "title": "ALERT!",
      "body": "There is a voilation at sector 25, {{name}}",
    },
    "data": {
      "name": "Prof",
    },
    "routing": {
      "method": "single",
      "channels": ["email"],
    },
  }
)
#print(resp['messageId'])
