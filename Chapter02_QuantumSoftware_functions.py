

## Berstein-Vazirani function

def secretBitStringFunction(x):
    s = '11010'
    if (len(x) != len(s)):
        print('Error: Length of secret string is', len(s))
        return 0
    a = 0
    for i in range(len(s)):
        a = a + int(s[i])*x[i]
    return a%2