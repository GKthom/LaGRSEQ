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
LEARNING_RATE = 0.001#dqn learning rate
MEMORY_SIZE = 10000#buffer size
BATCH_SIZE = 64
A=2#size of action space: change or dont change
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
log_freq=10#logging window -set to 10 steps
EXPLORATION_DECAY = 0.998#exponential decay factor by which epsilon decreases in each episode
Nruns=10#total trials
Nsteps=5#5 `scans' of the image, so it is actually 500 steps.
temp=0#LLM temperature
MAX_STEPS=10000#total steps in a trial
im_dim=10
######desired solution########
ideal_im=np.array([[1,0,0,0,0,0,0,0,0,1],
[0,1,0,0,0,0,0,0,1,0],
[0,0,1,0,0,0,0,1,0,0],
[0,0,0,1,0,0,1,0,0,0],
[0,0,0,0,1,1,0,0,0,0],
[0,0,0,0,1,1,0,0,0,0],
[0,0,0,1,0,0,1,0,0,0],
[0,0,1,0,0,0,0,1,0,0],
[0,1,0,0,0,0,0,0,1,0],
[1,0,0,0,0,0,0,0,0,1]])

thresh=0.95#threshold $\delta$ from the paper

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
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
                q_update = (reward + GAMMA*np.amax(self.model.predict(state_next,verbose=0)))
            q_values = self.model.predict(state,verbose=0)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate=max(1/episode_no,EXPLORATION_MIN)#reduce epsilon

    def experience_replay2(self,episode_no):
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
        self.model.load_weights(name)#load neural net weights

    def save(self, name):
        self.model.save_weights(name)#save neural net weights
   
def step_function(state,action,img):
    img_orig=img.copy()#retain copy of original image
    corr1=np.sum(img_orig==ideal_im)#number of pixels that match with target image
    if action==1:
        if img[int(state[0]),int(state[1])]==0:
            img[int(state[0]),int(state[1])]=1#flip pixel value
        elif img[int(state[0]),int(state[1])]==1:
            img[int(state[0]),int(state[1])]=0#flip pixel value
    corr=np.sum(img==ideal_im)#number of pixels in modified image that match with target image
    if corr==im_dim*im_dim:
        bonus=1
        terminal=True
        print("exact match!")
    else:
        bonus=0
        terminal=False
    reward=((corr-corr1)/(im_dim*im_dim))+bonus#reward function
    if img_orig[int(state[0]),int(state[1])]==0 and img[int(state[0]),int(state[1])]==1 and corr1>corr:
        img=img_orig.copy()#disallow pixel flips if action flipped pixel value from 0 to 1
        reward=0
    return img,reward,terminal


def LLMgreedy(state,img,llm_sol):
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
    reward=((corr-corr1)/(im_dim*im_dim))+bonus
    return img,reward,terminal
    
def findincache(lst, ip):
    return [i for i, x in enumerate(lst) if (np.sum(x.astype(int)==ip)==(im_dim*im_dim) and i%2==0)]#find matching index in cache

def image_completion(trial_no):
    initial_img=np.zeros((im_dim,im_dim))
    initial_state=[0,0,0]
    initial_state.extend(np.reshape(initial_img,im_dim*im_dim))#append pixel position and intensity with vectorized image
    initial_state=np.array(initial_state)
    observation_space = np.shape(initial_state)[0]
    action_space = A
    dqn_solver = DQNSolver(observation_space, action_space)#primary agent
    dqn_solver2 = DQNSolver(im_dim*im_dim, action_space)#secondary agent
    run = 0
    logret=[]
    step_ret=[]
    i=0
    llmsolflag=0#solution flag if LLM has found solution, initialize to 0
    llm_queries=0#initialize number of LLM queries
    terminal=False
    with open("imagecache_init0", "rb") as fp:   # Unpickling the cache
        flat_data = pickle.load(fp)
    while i<=MAX_STEPS:# continue untill max steps reached
        run += 1
        state = initial_state
        img=np.zeros((im_dim,im_dim))#initialize with blank image
        sumrew=0
        for j in range(Nsteps):#5 scans of the image
            for k in range(im_dim):#scan pixels
                for l in range(im_dim):
                    state=[k,l,img[k,l]]#construct state corresponding to each pixel
                    state.extend(np.reshape(img,im_dim*im_dim))#append with vectorized image
                    state=np.array(state)#state
                    if i>MAX_STEPS:
                        terminal=True
                        break
                    action = dqn_solver.act(state)#choose primary action
                    #train secondary RL
                    action2 = dqn_solver2.act(np.reshape(img,(im_dim*im_dim)))#choose secondary action -query/dont query
                    if llmsolflag==0:#if solution has not yet been found by LLM
                        img, reward, terminal = step_function(state,action,img)#execute primary action
                        if l<=(im_dim-2) and k<=(im_dim-2):#select next state as the state corresponding to the next scanned pixel
                            state_next=[k,l+1,img[k,l+1]]
                            state_next.extend(np.reshape(img,im_dim*im_dim))
                            state_next=np.array(state_next)
                        elif l==(im_dim-1) and (k+1)<=im_dim-1:
                            state_next=[k+1,0,img[k+1,0]]
                            state_next.extend(np.reshape(img,im_dim*im_dim))
                            state_next=np.array(state_next)
                        sumrew=sumrew+reward#update cumulative reward

                        if action2==1:#LLM is queried
                            try:
                                llm_hist_ind = findincache(flat_data, img)#try and find the state in cache
                                if len(llm_hist_ind)==0:#if not found in cache, query LLM response from server
                                    print("Querying LLM")
                                    answer=LLM.conv_ans_to_arr(LLM.getLLMresponse(img.astype(int),temp))
                                    flat_data.extend([img.astype(int),answer])                               
                                else:
                                    print('Retrieving from LLM history')#if found in cache then take the saved answer
                                    ind=llm_hist_ind[np.random.randint(len(llm_hist_ind))]
                                    resp=flat_data[ind+1]
                                    answer=resp.astype('int')
                    
                                LLMquality=np.sum(answer==ideal_im)/(im_dim*im_dim)#evaluate LLM solution
                                if LLMquality>thresh:#is it good enough?
                                    print("Soln found!")#yes
                                    llmsolflag=1
                                    llm_sol=answer
                                    reward2=1
                                else:
                                    reward2==0#Not good enough                            
                                llm_queries+=1
                            except:
                                print("Some error occurred")# if server error or incorrect format from LLM for example
                                reward2=0
                        else:
                            reward2=0#reward if LLM is not queried
                        
                        dqn_solver2.remember(np.reshape(img,(im_dim*im_dim)), action2, reward2, state_next, terminal)#build secondary buffer
                        dqn_solver2.experience_replay2(run)#update secondary agent DQN
                    else:
                        img,reward,terminal=LLMgreedy(state,img,llm_sol)#if solution is found, the execute LLM actions greedily
                        sumrew=sumrew+reward#update cumulative reward in this case as well
                    
                    i+=1
                    ###logging###
                    if i==0:
                        step_ret.append(0)
                    elif i<=log_freq:
                        step_ret.append(np.mean(logret))
                    else:
                        step_ret.append(np.mean(logret[(len(logret)-log_freq):len(logret)]))
                    ############
                    ########update DQN###################
                    dqn_solver.remember(state, action, reward, state_next, terminal)#update replay buffer
                    dqn_solver.experience_replay(run)#update action values
                    ################################
                    if terminal:
                        break
                if terminal:
                    break
                                         
            with open("imagecache_init0", "wb") as fp:#update cache
                pickle.dump(flat_data,fp)
            logret.append(sumrew)
        #print episode details#
        print('run:' + str(trial_no) +',steps:' + str(i)+',episode:' + str(run) + ', exploration: ' + str(dqn_solver.exploration_rate) + ',sumrew:' + str(sumrew))

    return logret, step_ret,img,llm_queries


if __name__ == "__main__":
    loglogret=[]
    logstepret=[]
    loglllmqueries=[]
    for i in range(Nruns):
        logret,stepret,img,llm_queries=image_completion(i)
        if i==0:
            logstepret=stepret
            loglllmqueries=llm_queries
        else:
            logstepret=np.vstack((logstepret,stepret))#returns
            loglllmqueries=np.vstack((loglllmqueries,llm_queries))#queries

    log_filename="image_completion_LaGRSEQ"+str(Nruns)+"_runs"+".npy.npz"
    np.savez(log_filename,logstepret,loglllmqueries)#save results