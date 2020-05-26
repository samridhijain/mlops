f= open("epoch.txt","r")
epochs=int(f.read())
f.close()

epochs+=1

f= open("epoch.txt","w+")
f.write(str(epochs))
f.close()
