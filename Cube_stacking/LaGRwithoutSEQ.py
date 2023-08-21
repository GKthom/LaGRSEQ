import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
import math
import time
import llm
import openai
import pickle

ideal_seq_char=['e','f','g','a','b','c','d','h']#target pattern
ideal_seq=llm.strtonum(ideal_seq_char)
N_episodes=100#no. of episodes
N_trials=10#trials
bonus=1#reward bonus
N_horizon=50#episode horizon
seqlen=len(ideal_seq)
A=seqlen+1#1 for each object and 1 for removing
S=0
for i in range(seqlen+1):
	S=S+math.factorial(i)
S2=seqlen#secondary state space
A2=2#secondary agent action space
alpha=0.1#learning rate
alpha2=0.1#secondary agent learning rate
gamma=0.95#primary agent discount factor
gamma2=0.95#secondary agent discount factor
temp=0.0

def take_action(state,action):
	reward1,match1=evalreward(state)#evaluate current solution
	lenstate=np.array([state]).size
	if action==0:#remove from stack
		if lenstate>0:
			state=np.delete(state,lenstate-1)
	else:#add to stack
		if lenstate<seqlen:
			state=np.append(state,action-1)
	newstate=state.copy()
	reward2,match2=evalreward(newstate)#evaluate modified solution
	reward=reward2-reward1#reward function
	if match1==1 or match2==1:
		reward+=bonus#add bonus
	return newstate,reward

def evalreward(state):
	match=0
	lenstate=np.array([state]).size
	ideal_part_seq=ideal_seq[0:lenstate]
	diff=np.array(state)-np.array(ideal_part_seq)
	reward=lenstate*(lenstate-np.count_nonzero(diff))#evaluation function
	if np.count_nonzero(diff)==0 and lenstate==np.array(ideal_seq).size:
		match=1
	return reward,match

def findstatefromseq(state):
	#converts sequence to state
	cnt=0
	lenstate=np.array([state]).size
	if lenstate>0:
		ideal_part_seq=ideal_seq[0:lenstate]
	else:
		ideal_part_seq=[]
	for p in multiset_permutations(ideal_part_seq):
		if len(p)==1:
			if p==state:
				break
			else:
				cnt+=1
		elif len(p)>1:
			if p==list(state):
				break
			else:
				cnt+=1
	return cnt-1

def findgreedyact(Q,state):
	#act greedily (primary agent)
	s=findstatefromseq(state)
	maxQ=np.max(Q[s,:])
	maxact=np.argmax(Q[s,:])
	return maxQ,maxact

def findgreedyact2(Q2,state2):
	#secondary agent greedy actions
	maxQ2=np.max(Q2[state2,:])
	maxact2=np.argmax(Q2[state2,:])
	return maxQ2,maxact2

def findincache(lst, ip):#search for solution index in cache
    return [i for i, x in enumerate(lst) if x==ip]


def LaGR_SEQ(flat_data):
	Q=np.zeros((S,A))#primary agent Q function
	Q2=np.zeros((S2,A2))#secondary agent Q function
	epsilon=1.0#initial epsilon
	min_eps=0.05#minimum epsilon
	retlog=[]
	corrlog=[]
	llmcountlog=[]
	llmlog=[]
	for i in range(N_episodes):
		epsilon=max(epsilon/(i+1),min_eps)
		print("episode no:"+str(i))
		state=[]
		ret=0
		llmcount=0
		llmcorrect_count=0
		llmsoln_flag=0
		correct_size=0
		last_correct_sol=[]
		for j in range(N_horizon):
			#####take action and get reward and next state#####
			if llmsoln_flag==0:
				if np.random.rand()>epsilon:#epsilon greedy action for primary agent
					maxQ,action=findgreedyact(Q,state)
				else:
					action=np.random.randint(A)
				state_new,reward=take_action(state,action)#execute primary action
				while np.count_nonzero(np.array(state)-action+1)<np.array(state).size:#check if sequence is valid
					action=np.random.randint(A)
					state_new,reward=take_action(state,action)
			else:
				state=last_correct_sol
				lenstate=np.array([state]).size
				#Use LLM solution to choose action
				if lenstate<np.array([llmsol]).size:
					action=llmsol[lenstate]+1
				else:
					action=llmsol[np.array([llmsol]).size-1]
				state_new,reward=take_action(state,action)#execute LLM action
				
			ret=ret+reward
			if llmsoln_flag==0:
				maxQnext,actionnext=findgreedyact(Q,state_new)#choose Q learning action
				#############Q-learning###################
				s=findstatefromseq(state)
				Q[s,action]=Q[s,action]+np.array(alpha)*[np.array(reward)+np.array(gamma)*maxQnext-Q[s,action]]#update primary Q function
			#########################################
			s2=len(last_correct_sol)-1
			#choose secondary agent action
			if np.random.rand()>epsilon:
				maxQ,a2=findgreedyact2(Q2,s2)
			else:
				a2=np.random.randint(A2)
			################Check if LLM can provide solution based on partial sequence###############
			if llmsoln_flag==0 and np.array([last_correct_sol]).size>correct_size:
				correct_size=np.array(last_correct_sol).size
				part_sol_lst = [int(x) for x in last_correct_sol]
				try:
					llm_hist_ind = findincache(flat_data, part_sol_lst)#try to find solution in cache
					if len(llm_hist_ind)==0:#if no solution in cache, then send LLM query to server
						print("Querying LLM")
						resp=llm.getLLMresponse(llm.convarrtollminput(llm.numtostr(part_sol_lst)),temp)
						answer=llm.conv_ans_to_arr(resp)
						flat_data.extend([part_sol_lst,resp])
					else:
						print('Retrieving from LLM history')#otherwise, retrieve solution from cache
						ind=llm_hist_ind[np.random.randint(len(llm_hist_ind))]
						resp=flat_data[ind+1]
						answer=llm.conv_ans_to_arr(llm.convarrtollminput(resp))						
					llmcount+=1
				except:
					print("Some error happened here.")#in case of any problems with server or invalid LLM solution format
				
				llmsol=answer
				if answer==ideal_seq:
					llmsoln_flag=1#set LLM solution flag as 1
					print('Solution found!')
					llmcorrect_count+=1
					r2=1#set secondary reward as 1
				else:
					print('Wrong Solution')
					r2=-1#set secondary reward as -1
				Q2[s2,a2]=Q2[s2,a2]+alpha2*(r2-Q2[s2,a2])	
			else:
				if a2==0:#if asking is skipped (as suggested by secondary agent)
					r2=0#set seondary reward as 0
					Q2[s2,a2]=Q2[s2,a2]+alpha2*(r2-Q2[s2,a2])#update secondary agent action values.
			
			############Check if correct action was taken- reset state to last correct state if incorrect##########
			lenstatenew=np.array([state_new]).size
			if lenstatenew>0:
				if state_new[lenstatenew-1]!=ideal_seq[lenstatenew-1]:
					state_new=state
			state=state_new.copy()
			#######################################################
			last_correct_sol=state_new
		if np.array([retlog]).size==0:
			retlog=ret
			corrlog=llmcorrect_count
			llmcountlog=llmcount
		else:
			retlog=np.vstack((retlog,ret))#record avg reward
			corrlog=np.vstack((corrlog,llmcorrect_count))#count correct responses
			llmcountlog=np.vstack((llmcountlog,llmcount))#count LLM responses

	return Q,ret,state,retlog,corrlog,llmcountlog,flat_data,Q2


if __name__=="__main__":
	retloglog=[]
	corrloglog=[]
	llmcountloglog=[]

	with open("cache0.0"+str(temp), "rb") as fp:   # Unpickling cache
		flat_data = pickle.load(fp)
	llm_hist_ind = findincache(flat_data, [4])
	for i in range(N_trials):
		print("Trial no:"+str(i))
		Q,ret,state,retlog,corrlog,llmcountlog,flat_data,Q2=LaGR_SEQ(flat_data)
		if np.array([retloglog]).size==0:
			retloglog=retlog
			corrloglog=corrlog
			llmcountloglog=llmcountlog
		else:
			retloglog=np.hstack((retloglog,retlog))
			corrloglog=np.hstack((corrloglog,corrlog))
			llmcountloglog=np.hstack((llmcountloglog,llmcountlog))
	with open("cache"+str(temp), "wb") as fp:   #update cache -Pickling query response pairs
		pickle.dump(flat_data, fp)

	np.savez("lagr_without_seq"+str(temp)+".npy.npz",retloglog,corrloglog,llmcountloglog,Q)#store results