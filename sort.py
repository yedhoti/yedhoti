name = "yogesh"

a = name[0]  
b = name[1] 
c = name[2]  
d = name[3]  
e = name[4]  
f = name[5]  


characters = [a, b, c, d, e, f]


letter_count = {}

for letter in characters:
    if letter not in letter_count:  
        letter_count[letter] = name.count(letter)

print(letter_count)