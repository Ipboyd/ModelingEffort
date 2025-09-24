
import numpy as np
import heapq
import random

def GA_update(p,scores,selection_type,crossover_type,mutation_type,mutation_rate,mutation_strength,elite_pop_frac):

    #Hyper-parameters (Might be worth it to up k over epochs potentially)
    num_match = 2 #(k)
    

    #print(np.shape(p))

    #Exclude Elites
    num_elites = round(elite_pop_frac*len(scores))
    idx_top = heapq.nsmallest(num_elites, range(len(scores)), key=scores.__getitem__)

    mask = np.ones(p.shape[1],dtype=bool)
    mask[idx_top] = False

    non_elite_scores = scores[mask]
    non_elite_param_set = p[:,mask]

    #print(np.shape(non_elite_param_set))

    ############################
    #--       Selection      --#
    ############################

    if selection_type == 'tournament':

        selection_excess = len(non_elite_scores)%num_match 
        list_nums = non_elite_scores[0:len(non_elite_scores)-selection_excess] #Trim the list if stuff won't fit clean into the tournamnet style

        maxes = np.reshape(list_nums,[int(len(list_nums)/num_match),num_match])
        idx = np.argmin(maxes,axis=1) 

        winners = (idx + np.arange(0,len(list_nums)/num_match)*num_match)  #indexes of the selected
        winners = winners.astype(int)

        winners_mask = np.zeros(non_elite_scores.shape[0],dtype=bool)
        winners_mask[winners] = True
   
    ############################
    #--       Crossover      --#
    ############################

    if crossover_type == 'random_even':

        #print(non_elite_param_set)
        #print(np.shape(non_elite_param_set))
        winning_param_sets = non_elite_param_set[:,winners_mask]
        crossover_excess = np.shape(winning_param_sets)[1]%num_match 

        crossover_extras = random.sample(np.arange(0,np.shape(winning_param_sets)[1]-1).tolist(),crossover_excess)
        cross_mask = np.zeros(winning_param_sets.shape[1],dtype=bool)
        cross_mask[crossover_extras] = True

        winning_param_sets = np.append(winning_param_sets,winning_param_sets[:,cross_mask],axis=1)

        parent_reshape = np.reshape(winning_param_sets,(np.shape(winning_param_sets)[0],int(np.shape(winning_param_sets)[1]/num_match),num_match))

        #print(np.shape(parent_reshape))

        rng = np.random.default_rng()
        groups = np.array_split(rng.permutation(np.arange(0,np.shape(winning_param_sets)[0])),num_match)

        children = np.zeros_like(parent_reshape)
        for k in range(len(groups)):
            children[:,:,k] = parent_reshape[:,:,k]
            #children[group[k],:,k] = parent_reshape[]
            
            for m in range(len(groups)):
                src = (m + k) % len(groups)
                children[groups[m],:,k] = parent_reshape[groups[m],:,src]

        child_reshape = np.reshape(children,(np.shape(children)[0],int(np.shape(children)[1]*num_match)))

        new_population = np.append(winning_param_sets,child_reshape,axis=1)

        new_population = new_population[:,0:np.shape(non_elite_param_set)[1]] #Trim Excess from crossover mismatches

        #print(np.shape(new_population))

    ############################
    #--       Mutation       --#
    ############################

    if mutation_type == 'random':
       
        #Create a random matrix of values
        rng = np.random.default_rng()

        rand_prob = rng.random((np.shape(new_population)[0], np.shape(new_population)[1]))  #Uniformly disributed Probabilites
        rand_strengths = (rng.random((np.shape(new_population)[0], np.shape(new_population)[1])) - 0.5) * 2  #Uniformly disributed Strengths [-1,1]
        prob_add = (rand_prob < mutation_rate) * mutation_strength * rand_strengths

        mutated_pop = new_population + new_population*prob_add #Chaing mutation_strength effectively into a proportion since it should be moudlated by the underlying parameters now.

    #Reintroduce the elites
    elites = p[:,np.array(idx_top)]

    output_pop = np.append(elites,mutated_pop,axis=1)

    #print(output_pop[:,0:10])

    #print(np.shape(output_pop))

    return output_pop