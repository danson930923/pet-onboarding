class Car(object):
        def __init__(self,make,model,color):
            self.make=make;
            self.model=model;
            self.color=color;
            self.owner_number=0
        def show(self):
            print(self.make + self.model + self.color)
        def sell(self):
            self.owner_number=self.owner_number+1

A = Car(model="AC", make="Ho", color="B")
A.model = "D"
print(A.show())

with open("test.txt") as File:
    file_stuff = File.read();
print(File.closed)
print (file_stuff)

for i in range(3):
    print(i)