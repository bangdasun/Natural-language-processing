def shift(stack, buff,  dgraph):
    # TODO
    # raise NotImplementedError
	
    # move the top value from buff and push into stack
    i = buff.pop()
    stack.append(i)
    
    
def left_arc(stack, buff, dgraph):
    # TODO
    # raise NotImplementedError
    
    # stack[i, j => stack[j , get arc i <- j 
    j = stack.pop()
    i = stack.pop()
    stack.append(j)
    dgraph.append((i, j))


def right_arc(stack, buff, dgraph):
    # TODO
    # raise NotImplementedError 
	
    # stack[i, j => stack[i , get arc i -> j
    j = stack.pop()
    i = stack.pop()
    stack.append(i)
    dgraph.append((j, i))


def oracle_std(stack, buff, dgraph, gold_arcs):
    # TODO
    # raise NotImplementedError()  
                
    if len(stack) <= 1:
        return 'shift'
    else:
        i, j = stack[-2:]
        if (i, j) in gold_arcs:
            return 'left_arc'
        elif (j, i) in gold_arcs:
            # all gold arcs in gold_arcs involving j as governor are already in dgraph: return 'right'
            head_w_j = []
            [head_w_j.append(a) for a in gold_arcs if a[1] == j]                
            for hwj in head_w_j:
                if hwj not in dgraph:
                    return 'shift'
            else:
                return 'right_arc'
        else:
            return 'shift'
            

def make_transitions(buff, oracle, gold_arcs=None):
    stack = []
    dgraph = []
    configurations = []
    while (len(buff) > 0 or len(stack) > 1):        
        choice = oracle(stack, buff, dgraph, gold_arcs)
        # Makes a copy. Else configuration has a reference to buff and stack.
        config_buff = list(buff)
        config_stack = list(stack)
        configurations.append([config_stack,config_buff,choice])
        if choice == 'shift':	shift(stack, buff, dgraph)
        elif choice == 'left_arc': left_arc(stack, buff, dgraph)
        elif choice == 'right_arc': right_arc(stack, buff, dgraph)
        else: return None
    return dgraph, configurations
