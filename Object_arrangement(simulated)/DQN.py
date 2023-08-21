import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import matplotlib.pyplot as plt
import LLM
import pickle

GAMMA = 0.95#discount factor
LEARNING_RATE = 0.001#DQN learning rate
MEMORY_SIZE = 10000#DQN buffer size
BATCH_SIZE = 16
A=2#size of action space: drop/dont drop object
EXPLORATION_MAX = 1.0#maximum epsilon
EXPLORATION_MIN = 0.1#minimum expsilon
EXPLORATION_DECAY = 0.998#exponential decay factor
Nruns=10#trials
Nsteps=50#episode horizon
temp=0#LLM temperature
MAX_STEPS=5000#total steps 5k for simulated run
im_dim=5# for 5x5 arrangement pattern
log_freq=10#logging window
#target arrangement matrix
ideal_im=np.array([[0,0,1,0,0],
 [0,1,0,1,0],
 [1,0,0,0,1],
 [0,1,0,1,0],
 [0,0,1,0,0]])
thresh=0.99#threshold $\delta$

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))#add to buffer

    def act(self, state):#action selection
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.array([state]),verbose=0)
        return np.argmax(q_values[0])

    def experience_replay(self,episode_no):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            state=np.array([state])
            if not terminal:
                state_next=np.array([state_next])
                q_update = (reward + GAMMA*np.amax(self.model.predict(state_next,verbose=0)))#DQN target
            q_values = self.model.predict(state,verbose=0)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)#update DQN
        self.exploration_rate=max(1/episode_no,EXPLORATION_MIN)#reduce epsilon with episodes

    def experience_replay2(self,episode_no):#replay for secondary agent
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                state_next=np.array([state_next])
                q_update = reward
            state=np.array([state])
            q_values = self.model.predict(state,verbose=0)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        
    def load(self, name):
        self.model.load_weights(name)#load DQN weights

    def save(self, name):
        self.model.save_weights(name)#save DQN weights

   
def step_function(state,action,img):
    img_orig=img.copy()#keep copy of original arrangement
    corr1=np.sum(img_orig==ideal_im)#evaluate similarity with target arrangement pattern
    if action==1:#drop object
        if img[int(state[0]),int(state[1])]==0:#check if object is already present
            img[int(state[0]),int(state[1])]=1#if not, drop
    corr=np.sum(img==ideal_im)#evaluate modified configuration's closeness with target pattern
    if corr==im_dim*im_dim:
        bonus=1
        terminal=True
        print("exact match!")
    else:
        bonus=0
        terminal=False
    reward=((corr-corr1)/(im_dim*im_dim))+bonus#reward function
    if img[int(state[0]),int(state[1])]==1 and img[int(state[0]),int(state[1])]!=ideal_im[int(state[0]),int(state[1])]:#prevent incorrect drops
        img[int(state[0]),int(state[1])]=0
        reward=0
    return img,reward,terminal


def LLMgreedy(state,img,llm_sol):
    #act greedily by following LLM actions.
    img_orig=img.copy()
    corr1=np.sum(img_orig==ideal_im)
    foundflag=0
    for i in range(im_dim):
        for j in range(im_dim):
            if llm_sol[i,j]==1 and img[i,j]==0:
                foundflag=1
                validstate=[i,j]
                break
        if foundflag==1:
            break
    if foundflag==1:
        img[int(validstate[0]),int(validstate[1])]=llm_sol[int(validstate[0]),int(validstate[1])]#execute greedy LLM action
    corr=np.sum(img==ideal_im)
    if (corr/(im_dim*im_dim))>thresh:
        bonus=1
        terminal=True
        print("exact llm match!")
    else:
        bonus=0
        terminal=False
    reward=((corr-corr1)/(im_dim*im_dim))+bonus#reward function
    return img,reward,terminal

    
def findincache(lst, ip):
    return [i for i, x in enumerate(lst) if (np.sum(x.astype(int)==ip)==(im_dim*im_dim) and i%2==0)]#search for matching index in cache

def arrange_obj(trial_no):
    initial_img=np.zeros((im_dim,im_dim))#initialize initial arrangement
    initial_state=[0,0,0]
    initial_state.extend(np.reshape(initial_img,im_dim*im_dim))
    initial_state=np.array(initial_state)#initialize initial state
    observation_space = np.shape(initial_state)[0]
    action_space = A
    dqn_solver = DQNSolver(observation_space, action_space)#primary agent
    dqn_solver2 = DQNSolver(im_dim*im_dim, action_space)#secondary agent
    run = 0
    logret=[]
    step_ret=[]
    i=0
    llmsolflag=0#set to 1 if LLM finds correct solution.
    llm_queries=0
    terminal=False
    with open("cache_robotgpt4", "rb") as fp:   # Unpickling recorded LLM responses
        flat_data = pickle.load(fp)
    while i<=MAX_STEPS:#continue till max steps reached
        run += 1
        state = initial_state
        img=np.zeros((im_dim,im_dim))
        sumrew=0
        for j in range(Nsteps):
            x=int(np.random.randint(im_dim))#choose a random location on the table
            y=int(np.random.randint(im_dim))
            state=[x,y,img[x,y]]
            state.extend(np.reshape(img,im_dim*im_dim))#construct state corresponding to this position
            state=np.array(state)
            if i>MAX_STEPS:
                terminal=True
                break
            action = dqn_solver.act(state)#choose primary DQN action
            action2 = dqn_solver2.act(np.reshape(img,(im_dim*im_dim)))#choose secondary DQN action
            action2=0#this line makes this code equivalent to DQN
            if llmsolflag==0:#if LLM solution is yet to be found
                img, reward, terminal = step_function(state,action,img)#execute chosen primary action
                xnext=int(np.random.randint(im_dim))#choose next position randomly from the grid
                ynext=int(np.random.randint(im_dim))
                state_next=[xnext,ynext,img[xnext,ynext]]
                state_next.extend(np.reshape(img,im_dim*im_dim))#construct corresponding next state
                sumrew=sumrew+reward#accumulate rewards
                print("starting query loop")
                if action2==1:#if secondary agent suggests to ask LLM
                    try:
                        llm_hist_ind = findincache(flat_data, img)#check to see if state exists in cache
                        if len(llm_hist_ind)==0:#if state was not previously queried (if it doesnt exist in cache)
                            print("Querying LLM")
                            answer=LLM.conv_ans_to_arr(LLM.getLLMresponse(img.astype(int),temp))#ask LLM by querying server
                            flat_data.extend([img.astype(int),answer])                               
                        else:
                            print('Retrieving from LLM history')
                            ind=llm_hist_ind[np.random.randint(len(llm_hist_ind))]#otherwise, retrieve from cache
                            resp=flat_data[ind+1]
                            answer=resp.astype('int')                                                        
                        LLMquality=np.sum(answer==ideal_im)/(im_dim*im_dim)#evaluate LLM solution
                        if LLMquality>thresh:#if solution is good enough
                            print("Soln found!")
                            llmsolflag=1#set llm solution flag to 1
                            llm_sol=answer
                            reward2=1
                        else:
                            reward2==0#non-binary reward:1/(1+2.71**(20*-((LLMquality/100)-0.9)))#LLMquality/100#0                            
                        llm_queries+=1
                    except:
                        print("Some error occurred")#in case some server error occurs or LLM response is in incorrect format
                        reward2=0
                else:
                    print("asking skipped")#if secondary agent does not recommend asking LLM
                    reward2=0                        
                print("training secondary RL")
                dqn_solver2.remember(np.reshape(img,(im_dim*im_dim)), action2, reward2, state_next, terminal)#build secondary agent buffer
                dqn_solver2.experience_replay2(run)#update secondary DQN
            else:
                #find greedy soln of LLM
                img,reward,terminal=LLMgreedy(state,img,llm_sol)#execute LLM solution greedily
                sumrew=sumrew+reward#log corresponding reward 
                    
            i+=1

            if i==0:
                step_ret.append(0)
            elif i<=log_freq:
                step_ret.append(np.mean(logret))
            else:
                step_ret.append(np.mean(logret[(len(logret)-log_freq):len(logret)]))
                
            dqn_solver.remember(state, action, reward, state_next, terminal)#primary DQN buffer
            dqn_solver.experience_replay(run)#update primary DQN
                
            if terminal:
                break
                                         
            with open("cache_robotgpt4", "wb") as fp:#store responses to cache
                pickle.dump(flat_data,fp)
        logret.append(sumrew)
        #print episode training details
        print('run:' + str(trial_no) +',steps:' + str(i)+',episode:' + str(run) + ', exploration: ' + str(dqn_solver.exploration_rate) + ',sumrew:' + str(sumrew))
        
    return logret, step_ret,img,llm_queries


if __name__ == "__main__":
    loglogret=[]
    logstepret=[]
    loglllmqueries=[]
    for i in range(Nruns):
        logret,stepret,img,llm_queries=arrange_obj(i)
        if i==0:
            logstepret=stepret
            loglllmqueries=llm_queries
        else:
            logstepret=np.vstack((logstepret,stepret))#average rewards
            loglllmqueries=np.vstack((loglllmqueries,llm_queries))#average no. of LLM queries

    log_filename="DQN_extended_robot_expts"+str(Nruns)+"_runs"+".npy.npz"
    np.savez(log_filename,logstepret,loglllmqueries)#save results
