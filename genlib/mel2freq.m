function f = mel2freq(m)
f = 700.*(exp(m/1127.01048) - 1);