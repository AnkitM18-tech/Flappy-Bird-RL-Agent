#Import dependencies
import numpy as np
import random
import flappy_bird_gym
from collections import deque
from keras.layers import Dense, Input
from keras.models import load_model,save_model,Sequential
from keras.optimizers import RMSprop

#Neural Network for Agent
def NeuralNetwork(input_shape, output_shape):
    model = Sequential()
    # model.add(Input(input_shape))
    model.add(Dense(512,input_shape=input_shape,activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256,activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64,activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_shape,activation='linear', kernel_initializer='he_uniform'))
    model.compile(loss='mse',optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01),metrics=['accuracy'])
    model.summary()
    return model

#Building Class for FLappy Bird Agent
class DQN_agent:
    def __init__(self):
        #Env Variables
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n   
        self.memory = deque(maxlen=2000)

        #Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.batch_number = 64 # 16 32 128 256


        self.train_start = 1000 
        self.jump_prob = 0.01  

        self.model = NeuralNetwork(input_shape=(self.state_space,), output_shape=(self.action_space))
    #Acting function
    def act(self,state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() < self.jump_prob else 0

    #Learn Function
    def learn(self):
        #Making sure we have enough data
        if len(self.memory) < self.train_start:
            return
        
        #Create a small batch
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_number))
        #Variables to store batch info
        state = np.zeros((self.batch_number, self.state_space))
        next_state = np.zeros((self.batch_number, self.state_space))

        action, reward, done = [],[],[]
        #Storing data
        for i in range(self.batch_number):
            state[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_state[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        
        #Predict Y values
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_number):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_number, verbose=0)

    #train function
    def train(self):
        #Episode iteration for training
        for i in range(self.episodes):
            #Env variables for training
            state = self.env.reset()
            state = np.reshape(state, [1,self.state_space])
            done = False
            score=0
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min

            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                #reshape next state
                next_state = np.reshape(next_state, [1,self.state_space])
                score+=1
                
                if done:
                    reward-=100

                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print("Episode: {}\nScore: {}\nEpsilon: {:.2}".format(i, score, self.epsilon))
                    #Save Model
                    if score>=1000:
                        self.model.save_model('flappyBird.h5')
                        return

                self.learn()

    #Visualize our Model
    def perform(self):
        self.model = load_model('flappyBird.h5')
        while 1:
            state = self.env.reset() 
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            while not done:
                self.env.render()  
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score+=1

                print("Current Score is: {}".format(score))

                if done:
                    print("DEAD")
                    break

if __name__ == '__main__':
    agent = DQN_agent()
    # agent.train()  #after training commenting it out
    agent.perform()