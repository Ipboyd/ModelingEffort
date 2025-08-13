import numpy as np

def adam_update(m,v,p,t,beta1,beta2,lr,eps,out_grad):
   
    t += 1

    #Caclulate first and second moment
    #m = [beta1*m[ms,:] + (1-beta1) * out_grad[ms,:] for ms in range(len(m))]
    #v = [beta2*v[vs,:] + (1-beta2) * (out_grad[vs,:]**2) for vs in range(len(v))]

    m = beta1*m + (1-beta1) * out_grad
    v = beta2*v + (1-beta2) * (out_grad**2)

    #print('m')
    #print(m)
    #print('v')
    #print(v)

    #m_hat = [m[ms,:]/(1 - beta1 ** t) for ms in range(len(m))]
    #v_hat = [v[vs,:]/(1 - beta2 ** t) for vs in range(len(v))]

    m_hat = m/(1 - beta1 ** t)
    v_hat = v/(1 - beta2 ** t)

    #print('m')
    #print(m_hat)
    #print('v')
    #print(v_hat)

    #Update p
    #p = [p[vs,:] - lr*m_hat[vs,:]/(np.sqrt(v_hat[vs,:]) + eps) for vs in range(len(v))]

    p = p - lr*m_hat/(np.sqrt(v_hat) + eps)


    #Make sure that p is never negative (Negetive conductances do not make sence in our scenario) 8-12-2025
    p = abs(p)

    #print('p')
    #print(p)

    #p = [p[vs] - lr*m_hat[vs] for vs in range(len(m))]

    return m,v,p,t
