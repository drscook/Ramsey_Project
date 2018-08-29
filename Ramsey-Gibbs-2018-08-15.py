check_gpu_sync = False#True#False    # for code testing and debugging.  Set False for better performance
N = 36
ramsey = [4,6]

# N = 30
# ramsey = [3,10]


steps_tot = 100000000

max_block_shape = 2**10
max_grid_shape = (2**31)-1

# Controls how many past clrngs are kept in memory and checked for duplications.  Larger value means we remember more.  
# Pro - avoids revisiting the same clrng.  For Gibbs, it is quite wasteful to revisit the same clrng because we will then
# waste time rechecking all (or most) of its neighbors.
# Con - More time spent checking for duplication.
# Need a good balance.
max_record_megabytes = 10    

filename = f"R{ramsey}_N{N}.npz"

import random
# random.seed(42)  # set if you want reproducibility

from timeit import default_timer as timer
import numpy as np
import numba as nb
import numba.cuda as cuda
import scipy.special

start_time = timer()

def time_format(x):
    h, m = np.divmod(x, 3600)
    m, s = np.divmod(m, 60)
    return f"{int(h):02d} hr {int(m):02d} min {s:05.02f} sec"

def get_dtype(A):
    """
    Find the smallest feasible data type
    """
    A = np.asarray(A)
    dtypes = [[np.uint8, nb.uint8], [np.uint16, nb.uint16], [np.uint32, nb.uint32], [np.uint64, nb.uint64]
             ,[np.int8,  nb.int8],  [np.int16,  nb.int16],  [np.int32,  nb.int32],  [np.int64,  nb.int64]
             ,[np.float64, nb.float64]]
    for d in dtypes:
        B = A.astype(d[0])
        if np.allclose(A, B):
            return d[0], d[1], B
    raise Exception('Could not select dtype')

def comb(N,K):
    """
    Vectorized N choose K computation with optimal data typing
    """
    A = scipy.special.binom(N, K)
    np_dtype, nb_dtype, B = get_dtype(A)
    return B


#######################################################################################################################################
### MAIN GPU KERNEL
#######################################################################################################################################

@cuda.jit
def main_kernel(verts_per_clique, cliques_per_clr, choose, verts_to_edge, clrng, changes, problems_tot):
    """
    The main gpu kernel
    """
    idx_block = cuda.blockIdx.x
    idx_loc = cuda.threadIdx.x
    idx_glob = idx_loc + idx_block * cuda.blockDim.x
        
    ### Read from global to shared memory ###
    # Each block has its own fast shared memory.  Fast performance if we first read global to shared.
    
    # The outer loop is over j (column index) to be consistent with the combinatorial numbering system (yes, this looks weird)
    choose_shr = cuda.shared.array(shape=choose_shape, dtype=choose_dtype_nb)
    verts_to_edge_shr = cuda.shared.array(shape=(verts_tot, verts_tot), dtype=verts_to_edge_dtype_nb)
    j = idx_loc
    while j < verts_tot: # only loops if block_shape < verts_tot.  If so, each thread loads more than one column.
        for i in range(verts_tot):
            verts_to_edge_shr[i,j] = verts_to_edge[i,j]
            choose_shr[i,j] = choose[i,j]
        # choose_shr has 1 more row than verts_to_edge
        i += 1 
        choose_shr[i,j] = choose[i,j]
        j += block_shape  

    # choose_shr has 1 more column than verts_to_edge
    if idx_loc == 0:
        j = verts_tot
        for i in range(verts_tot+1):
            choose_shr[i,j] = choose[i,j]

            
    ### Update clrng_gpu to stay consistent with clrng (on the cpu) ###
    # Note - change edge changes[e,0] to clr changes[e,1]
    # Note - It is possible for 2 rows to refer to same edge.  In this case, the latter clr is used and the former color is ignored.
    # Ex: changes[0,0] = changes[1,0] = 42.  In this case, edge 42 is assigned clr changes[1,1], not changes[0,1]
    e = idx_loc
    if e == changes[1,0]:
        clrng[e] = changes[1,1]
    elif e == changes[0,0]:
        clrng[e] = changes[0,1]



    clrng_shr = cuda.shared.array(shape=(edges_tot,), dtype=clrng_dtype_nb)
#     edge_to_verts_shr = cuda.shared.array(shape=(edges_tot, 2), dtype=edge_to_verts_dtype_nb)
    # We don't actually need edges_to_verts, so it is commented out.  I left it for possible future use - must add to function signature.
    while e < edges_tot: # only loops if block_shape < edges_tot.  If so, each thread loads more than one column.
        clrng_shr[e] = clrng[e]
#         edge_to_verts_shr[e,0] = edge_to_verts[e,0]
#         edge_to_verts_shr[e,1] = edge_to_verts[e,1]
        e += block_shape

    cuda.syncthreads()


    ### Each thread computes the clique(s) it should monitor, computes the vertices in that clique, and decides if it is a problem (aka - all edges between pairs of vertice have the specified "bad" color).
    
    # Goal - Move to next clique AS SOON as we find ANY edge in the clique with a "good" color.  A clique is only a problem if ALL edges have the bad color.
    
    # Each thread has a small, fast local memory.  
    clique_verts = cuda.local.array(shape=(verts_per_clique_max,), dtype=edge_to_verts_dtype_nb)
    

    problems_count = 0   # number of problems this thread has detected
    clr = 0    # "bad" color ... all other colors are "good".  If we find an 1 edge in the clique with a good color, we should immediately move to next clique
    counter = idx_glob    # determines which clique this thread should evaluate next
    while clr < clrs_tot:   # process completes when we run out of colors
        K = verts_per_clique[clr]
        stop = cliques_per_clr[clr]
        while counter < stop:    # when counter exceeds stop for this color, move to next color
            problem = True
            C = counter    # combinatorial numbering system says to repeatedly decrement C until it reaches 0 (or we find a "good edge")
            for j in range(K, 0, -1):    # j = subset size, in decreasing order from K, K-1, K-2, ..., 1 (but not 0)

                # Find largest value of a such that (a choose j) <= current value of C
                a = 1                
                while choose_shr[a,j] <= C:   # loop until (a choose j) > C
                    a += 1
                a -= 1   # decrement a
                clique_verts[j-1] = a   # a is the vertex for entry j-1
                C -= choose_shr[a,j]   # decrement C by (a choose j)
                
                # Before we compute another vertex, we check the colors on edges where both vertices have already been computed.
                # This is efficient.  If we find ANY edge with a good color, we can IMMEDIATELY move to next clique without computing
                # the remaining vertices.
                # We have now computed entries [j-1:K) of clique_verts.
                # We previously checked edges where both vertices are in entries [j:K).
                # They must all be bad, else we would have already hit a break.
                # Therefore, now we check edges where one vertex is a (the entry in j-1 we just computed) and the other is in [j:K)
                for b in clique_verts[j:K]:
                    e = verts_to_edge_shr[a, b]
                    if clrng_shr[e] != clr:
                        problem = False    # Hurray - a good color!  We can move to the next clique.
                        break    # breaks the b loop
                if not problem:
                    break    # breaks the j loop
            problems_count += problem    # increment if we just found a problem clique
            counter += threads_tot    # incremented by threads_tot after each clique so the thread can find its next clique to evaluate (if any)
        clr += 1    # when counter exceeds stop for this color, move to next color
        counter -= stop    # resets clique counter for the new color
        
    cuda.atomic.add(problems_tot, 0, problems_count)    # when this thread is done, add its local problem count to the global problem count
    
    
    
    
#######################################################################################################################################
### Compute combinatorics ###
#######################################################################################################################################
verts_tot = N
choose_shape = (verts_tot+1, verts_tot+1)
choose = np.fromfunction(comb, shape=choose_shape)

verts_per_clique = np.asarray(ramsey)
clrs_tot = len(verts_per_clique)
max_clr = np.argmax(verts_per_clique)
verts_per_clique_max = verts_per_clique[max_clr]

edges_tot = choose[verts_tot,2]
edges_per_clique = choose[verts_per_clique, 2]
edges_per_clique_max = edges_per_clique[max_clr]

cliques_per_clr = choose[verts_tot, verts_per_clique]
cliques_per_clr_max = np.max(cliques_per_clr)
cliques_tot = np.sum(cliques_per_clr)

clrngs_tot = float(clrs_tot) ** edges_tot



#######################################################################################################################################
### Create edge <-> verts translators ###
#######################################################################################################################################
verts_to_edge = np.full(shape=(verts_tot, verts_tot), fill_value=edges_tot)
edge_to_verts = np.full(shape=(edges_tot, 2), fill_value=verts_tot)

# Creates verts_to_edge and edge_to_verts enumerated using the combinatorial numbering system (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), ...
# Fill upper triangle of verts_to_edge first
edge_idx = 0
for j in range(verts_tot):
    for i in range(j):
        verts_to_edge[i,j] = edge_idx
        edge_to_verts[edge_idx,0] = i
        edge_to_verts[edge_idx,1] = j
        edge_idx += 1

# Now fill lower triangle so it is symmetric
verts_to_edge = np.fmin(verts_to_edge, verts_to_edge.T)


#######################################################################################################################################
### Function to enumerate clrngs (for record keeping) ###
#######################################################################################################################################
# Group edges into chunks and combine their colors into a single uint64
# Compute max chunk size that will fit into uint64
# Solve for largest chunk where clrs_tot^chunk <= 2^64
chunk_size = int(np.floor(64 * np.log(2) / np.log(clrs_tot)))
chunks_tot = int(np.ceil(edges_tot/chunk_size))    # rows = chunks_tot = ceil(edges_tot / chunk_size)
rows = chunks_tot
cols = edges_tot
x = clrs_tot ** np.arange(chunk_size, dtype=np.uint64)    # [clrs_tot^0, clrs_tot^1, clrs_tot^2, ...]

# Goal - an array of shape (rows,cols) where each row has a copy of x starting at position row_idx * chunk_size (otherwise 0)
# However, the last row may have its copy of x cut short to fit
y = np.append(x, np.zeros(cols))    # note len(x) <= cols.  Append cols 0's to its end.
z = np.tile(y, rows)    # now len(y) >= cols.  Lay rows copies of y end to end

# now len(z) >= rows * cols.  Keep only the first rows*cols entries of z and reshape in a array of shape (rows,cols)
clrng_enum_arr = z[:int(rows*cols)].copy().reshape(rows,cols).astype(np.uint64)

# Now, a clrng is enumerated by simply multiplying it by clrng_enum_arr.  Returns a vector of length chunks_tot.
def clrng_enum(C):
    return clrng_enum_arr.dot(C)

#######################################################################################################################################
### Compute optimal cuda block and grid shapes and record length ###
#######################################################################################################################################
block_shape = int(min(max_block_shape, cliques_tot))    # Use max_block_shape iff there are enough cliques to fill them

q = np.ceil(cliques_tot / block_shape)    # number of blocks needed
grid_shape = int(min(max_grid_shape, q))    # Make sure we aren't asking for more blocks than allowed.
threads_tot = int(block_shape * grid_shape)
cliques_per_thread = int(np.ceil(cliques_tot / threads_tot))    # If we need more blocks than allowed, threads will be asked to monitor multiple cliques
    
max_uint64 = 2**64-1
rec_length = int(np.floor(max_record_megabytes * 1000000 / chunks_tot / 8))    # each uint64 is 8 bytes
clrng_rec = np.full(shape=(rec_length, chunks_tot), fill_value=max_uint64, dtype=np.uint64)
problems_rec = np.full(shape=(rec_length,), fill_value=max_uint64, dtype=np.uint64)
problems_new = np.array([0], dtype=np.uint64)
    
#######################################################################################################################################
### Initialize ###
#######################################################################################################################################

### Set initial clrng of the edges ###
clrng = random.choices(range(clrs_tot), k=edges_tot)
if max(clrng) < clrs_tot-1:    # for dtype detection, we need the highest color to appear somewhere in clrng
    clrng[0] = clr_tots-1
    
### Compute optimal dtypes and push to gpu
# Must be careful about data types and pushing arrays to gpu.
L = {'choose':choose, 'verts_per_clique':verts_per_clique, 'cliques_per_clr':cliques_per_clr,
     'verts_to_edge':verts_to_edge, 'edge_to_verts':edge_to_verts, 'clrng':clrng}

for name, arr in L.items():
    np_dtype, nb_dtype, arr = get_dtype(arr)
    arr_gpu = cuda.to_device(arr)
    # Lines belows are hack-y (exec is frowned on).
    # But numba.cuda demands dtypes to have dedicated variables.  They can't be in a list or dict.
    # Let me know if you can do this more nicely
    exec(f"{name}_dtype_np, {name}_dtype_nb = np_dtype, nb_dtype")
    exec(f"{name} = arr")
    exec(f"{name}_gpu = arr_gpu")

# change edge changes[e,0] to clr changes[e,1]
changes = np.zeros((2,2), dtype=verts_to_edge_dtype_np)
changes[:,1] = clrng[changes[:,0]]
delta_clr = 1
ptr = 0    # next record slot to write to

### Compute initial problems
def count_problems(rec=False):
    problems_new[0] = 0
    main_kernel[grid_shape, block_shape](verts_per_clique_gpu, cliques_per_clr_gpu, choose_gpu, verts_to_edge_gpu, clrng_gpu, changes, problems_new)
    if check_gpu_sync:    # for code testing and debugging.  Disable for better performance
        assert np.allclose(clrng, clrng_gpu.copy_to_host()), f"{clrng-clrng_gpu.copy_to_host()}\n{changes}"
#         print("In sync")
    if rec:
        q = ptr + 1
        clrng_rec[q%rec_length] = clrng_enum(clrng)
        problems_rec[q%rec_length] = problems_new[0]
        return q
    return ptr

ptr = count_problems(rec=True)

problems_range = np.float64(problems_new[0] * 1)    # for MCMC annealing later
problems_best = problems_new[0]
clrng_best = clrng.copy()

print(f"""Ramsey = {ramsey}, N = {N}, cliques total = {cliques_tot}, block shape = {block_shape}, grid shape = {grid_shape}, cliques per thread = {cliques_per_thread}, rec_length = {rec_length}
clrngs_tot = {clrngs_tot}
""")


#######################################################################################################################################
### Run MCMC - Gibbs sampler ###
#######################################################################################################################################


edges_rec = np.arange(edges_tot, dtype=verts_to_edge_dtype_np )
clrs_rec = edges_rec * 0
probs_rec = np.full(len(edges_rec), fill_value=max_uint64)
for step in range(1,steps_tot+1):
    clrs_rec[:] = 0
    probs_rec[:] = max_uint64
    for (i, e) in enumerate(edges_rec):
        # Pick e, change clr, count problems, change clr back.  Repeat
        # Issue - must keep CPU and GPU in sync as efficiently as possible
        # Do not want to push entire coloring back and forth, since only 1-2 entries change
        # We use the "changes" array to do so
        # Plan - select e and new_clr on CPU.  Change clrng on CPU.  Push changes to GPU.  Change clrng_gpu & count problems.  Undo change on CPU.
        # But, on the next step, we need undo the prior change on clrng_gpu.  Row 0 of changes is used to undo the prior change.  Row 1 of changes does the new change for this step.
        # This process is a bit delicate, but is quite efficient.
                 
                 
        
#         Select its new clr
        if clrs_tot > 2:
            delta_clr = random.randrange(clrs_tot-1) + 1
    #     else:
    #         delta_clr = 1

        
        old_clr = clrng[e]
        new_clr = (old_clr + delta_clr) % clrs_tot
        changes[1,0] = e
        changes[1,1] = new_clr
        clrng[e] = new_clr
    #     print(f"Changing edge {change_edges[1]} by {delta_clr} from {old_clr} to {new_clrs[1]}")

        enum = clrng_enum(clrng)    # enumerate new clrng
        match = np.all(np.equal(enum, clrng_rec[:ptr]), axis=1)    # checks for match in clrng_rec
        if np.any(match):    # If so, do not re-count problems and forbid moving there at end of this step
            problems_new[0] = max_uint64
            clrng[e] = old_clr    # undo on CPU
#             print(f"clrng {enum} was already recorded in slot {np.argmax(match)}")
        else:
            ptr = count_problems(rec=False)

            edges_rec[i] = e
            clrs_rec[i] = new_clr
            probs_rec[i] = problems_new[0]
            if problems_best > problems_new[0]:
                print(f"New {problems_new[0]} is better than best {problems_best}")
                problems_best = problems_new[0]
                clrng_best = clrng.copy()
                A = np.append(edge_to_verts, clrng_best[:,np.newaxis], axis=1)
                np.savez(filename, clrng=A, problems=problems_new)

                if problems_new[0] == 0:
                    print(f"Found clrng with no issues\n{clrng}!!!")
                    print(f"After {step} steps, best clrng has {problems_best} problems. Time elapsed = {time_format(timer() - start_time)}")
                    quit()
            clrng[e] = old_clr    # undo on CPU
            # Mark for undo on GPU on next step
            changes[1,1] = old_clr
            changes[0] = changes[1]    # Move change from prior step to entry 0 so that change is done on clrng_gpu in next kernel call

    ## The lines below are one way to do "annealing".  As problems_best decreases, we are less willing to accept changes that are
    ## sub-optimal (contain more problems than the best option).  Idea - we should "explore" the space early in the run by accepting
    ## more sub-optimal moves.  As we "approach" the solution (problems_best decreases), we should get more and more "picky"
    ## by more strongly preferring optimal moves (that have the least problems).

    anneal = problems_best / problems_range    # gets smaller as problems_best decreases; always <= 1
    weights = anneal ** probs_rec    # clrngs with more problems raise anneal to higher power, giving a much smaller weight.  Thus, it is much less likely to be selected below
    weights[probs_rec==max_uint64] = 0.0    # forbids selection of previously visited clrngs
#     print(anneal)
#     print(weights)
#     weights /= weights.sum()
#     print(weights)
#     print(probs_rec)
    nbr = random.choices(range(len(probs_rec)), weights)    # Selects new clrng with bias based on number of problems (weights)
    
    # Re-apply the change from the selected clrng
    e = edges_rec[nbr]
    new_clr = clrs_rec[nbr]
    changes[1,0] = e
    changes[1,1] = new_clr
    clrng[e] = new_clr
    ptr = count_problems(rec=True)    # Re-count so that the last change of the for loop above gets undone on clrng_gpu
    changes[0] = changes[1]

    
    if (step % 1) == 0:
            elapsed = timer() - start_time
#             print(f"Gibbs selected a new clrng {nbr} with {problems_new[0]} problems.  Best this step was {np.min(probs_rec)}.")
            print(f"After {step} steps, best clrng has {problems_best} problems. Time elapsed = {time_format(elapsed)} = {(elapsed / step):.2f} sec / step")
            
#     for i, (pr,wt) in enumerate(zip(probs_rec,weights)):
#         print(f"{i} - {pr} : {100*wt:.4f}")
