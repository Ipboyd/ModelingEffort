#Orginal Training loop for torchdifeq
f"""
    t = torch.linspace(0, 5, 100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


    # Training loop
    for step in range(500):
        optimizer.zero_grad()
    
        pred_z = odeint(model, ninit_cond, t)  # pred_z shape: (T, state_dim)

        # Suppose R2On_V is stored as part of the full state vector at index `i_R2On_V`
        R2On_V_trace = pred_z[:, {State_variable_Identifier.Find_Var(file_path)}]  # shape: (T, N_R2On)

        # Compute average firing rate over time
        avg_fr = compute_avg_firing_rate(R2On_V_trace)

        print(avg_fr)

        # Define a target firing rate (e.g., 10 Hz)
        target_fr = torch.tensor(0.01)  # if time units are seconds and you're simulating 1000 steps

        loss = (avg_fr - target_fr).pow(2)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 499:
            print(f"Step {{step}} | Loss: {{loss.item():.6f}} | Avg FR: {{avg_fr.item():.6f}}")

if __name__ == "__main__":
    main()


