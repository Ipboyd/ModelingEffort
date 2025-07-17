def grads_update(grads, p):
    
    
    p  = np.array([...])        # shape (n_params,)
    m  = np.zeros_like(p)       # 1st-moment
    v  = np.zeros_like(p)       # 2nd-moment
    t  = 0

    # --- training step -------------------------------------------------
    t += 1
    grad = np.array([...])      # same length as p

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    p -= lr * m_hat / (np.sqrt(v_hat) + eps)
