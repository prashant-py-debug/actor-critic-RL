import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self,fc1_dims = 512 , fc2_dims=512 , name = "critic" , chkpt_dir = "temp/ddpg_2000"):
        super(CriticNetwork,self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir,self.model_name+"_ddpg.h5")

        self.fc1 = Dense(self.fc1_dims,activation="relu")
        self.fc2 = Dense(self.fc2_dims,activation="relu")
        self.q = Dense(1 )

    def call(self,state,action):
        action_value = self.fc1(tf.concat([state,action], axis = 1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self,fc1_dims=512,fc2_dims=512 , n_actions = 2 , name="actor" , chkpt_dir="temp/ddpg_2000"):
        super(ActorNetwork,self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name 
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir , self.model_name+"_ddpg.h5")
        

        self.fc1 = Dense(self.fc1_dims,activation="relu")
        self.fc2 = Dense(self.fc2_dims,activation ="relu")
        self.mu = Dense(self.n_actions,activation ="tanh")

    def call(self,state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu




