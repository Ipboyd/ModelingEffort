function m = freq2mel(f)
m = 1127.01048.*log(f./700 + 1);