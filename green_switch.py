import greenswitch
import sys
import time
print('Initiate NahSanchar module')
# fs = greenswitch.InboundESL(host='192.168.50.6')
fs = greenswitch.InboundESL(host='192.168.100.167', port=8021, password='Master12@#')
#print(fs.connect())
# if(fs.connect()):
#     response = fs.connect()
# print(response)
retries = 5
success = False
def initiate_call():
    global success
    while not success:
        try:
            fs.connect()
            print(fs.connect())
            success = True
        except Exception as e:
            wait = retries * 30;
            print ('Error! Waiting %s secs and re-trying...' % wait)
            sys.stdout.flush()
            time.sleep(wait)
            retries += 1
def call(numbers):
    if(success==True):

        for number in numbers:
            link = 'bgapi originate {origination_caller_id_number='+str(number)+'}user/'+str(number)+'@192.168.100.167 7010 XML 192.168.100.167 CALLER_ID_NAME CALLER_ID_NUMBER'
            r = fs.send(link)
            # r = fs.send('bgapi originate {origination_caller_id_number=7017}user/7017@192.168.18.120 9999 XML 192.168.18.120 CALLER_ID_NAME CALLER_ID_NUMBER')
            # print(r)
    else:
        print("could not connect")

if __name__=="__main__":
    initiate_call()
    call([9004])
