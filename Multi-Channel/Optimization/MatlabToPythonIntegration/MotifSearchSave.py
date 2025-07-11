
def recurse(cur_node, tracker,nodes,edges):

  tracker.append(cur_node)

  for k in edges:
    if cur_node == k.split('->', -1)[1]:
      recurse(k.split('->', -1)[0],tracker,nodes,edges)
    

  return tracker


nodes = ['R2On','R1On','On','Off','S2OnOff','S1OnOff','R1Off','R2Off']
edges = ['On->R1On','On->S1OnOff','Off->R1Off','Off->S1OnOff','S1OnOff->R1On','S1OnOff->R1Off','R1On->R2On','R1On->S2OnOff','R1Off->R2Off','R1Off->S2OnOff','S2OnOff->R2On','S2OnOff->R2Off']

#Preform depth first search to get a list of all paths
all_paths = recurse('R2On',[],nodes,edges)
print(all_paths)

path_holder = []
path = ''
gate = 0
gate2 = 0


for k in range(len(all_paths)-1):

  #print(all_paths[k+1])

  #Compare with the eddges
  for m in range(len(edges)):
    if edges[m].split('->', -1)[0] == all_paths[k+1] and edges[m].split('->', -1)[1] == all_paths[k]:
      path = edges[m] + '->' + path
      gate = 1
      #print('here')
      #print(path)
      

  #If you have gone through all of them and the gate does not get flipped then append
  #print(gate)
  if gate == 0:
    path_holder.append(path[:-2])
    for z in range(k):
      for m in range(len(edges)):
        if edges[m].split('->', -1)[0] == all_paths[k+1] and edges[m].split('->', -1)[1] == all_paths[k-(z+1)]:
          #print(path)
          #print(z)
          #1. Search backwards to find the node that it connects to. Done
          #2. Search through the path and replace in place the new connection
          
      
          #print(path)
          
          path_segmented = path.split('->',-1)
          #print(path_segmented)

          for count, n in enumerate(path_segmented):
            
            #print(count)
            #print(n)
            #print(edges[m].split('->', -1)[1])
            #print(path)

            if n == edges[m].split('->', -1)[1]:
              
              #print(count)

              path = path.split('->',count+1)[count+1]
              path = edges[m] + '->' + path
              if k == len(all_paths)-2 and gate2 == 0:
                gate2 = 1
                path_holder.append(path[:-2])
              break


          #path = path.split('->', 2+z)[2+z]
          #print(path)
          #path = edges[m] + '->' + path
          break
          
 

  gate = 0

  

    

print(path_holder)
print(len(path_holder))
