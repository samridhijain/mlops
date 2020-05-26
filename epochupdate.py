with open('/fold1/epoch.txt') as f:
  epochs=int(f.readline())

epochs+=1

f= open("/fold1/epoch.txt","w+")
f.write(str(epochs))
f.close()
