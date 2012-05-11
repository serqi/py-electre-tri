from ElectreTri import Criterium, LimitProfile, Category, ElectreTriSimple


BAYES = Criterium("BAYES")
SPAMCOP = Criterium("SPAMCOP")
RAZOR = Criterium("RAZOR")
PYZOR = Criterium("PYZOR")

lp1 = LimitProfile("lp1")

SPAM = Category(1, name = "SPAM", lp_inf = lp1)
HAM = Category(2, name = "HAM", lp_sup = lp1)



criteria = [BAYES, SPAMCOP, RAZOR, PYZOR]
categories = [SPAM, HAM]


ets = ElectreTriSimple(criteria, categories)

ets.save("spamfilter.ets")


    
    