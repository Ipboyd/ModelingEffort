import numpy as np

def adam_update(m,v,p,t,beta1,beta2,lr,eps,out_grad):
   
    t += 1

    #Caclulate first and second moment
    m = [beta1*m[ms] + (1-beta1) * out_grad[ms] for ms in range(len(m))]
    v = [beta2*v[vs] + (1-beta2) * (out_grad[vs]**2) for vs in range(len(v))]

    m_hat = [m[ms]/(1 - beta1 ** t) for ms in range(len(m))]
    v_hat = [v[vs]/(1 - beta2 ** t) for vs in range(len(v))]

    #Update p
    p = [p[vs] - lr*m_hat[vs]/(np.sqrt(v_hat[vs]) + eps) for vs in range(len(v))]

    return m,v,p,t
