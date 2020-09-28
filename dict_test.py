

dict= {'x':[2,3], 'y':[3, 4], 'z':[4, 5]}

for arg in dict:

    [arg1, arg2]= dict[arg]
    list= dict[arg]
    print(arg1, arg2)
    print(list)