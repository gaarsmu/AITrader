import numpy as np
import pandas as pd


class Simple_env:

    def __init__(self, path_to_data, window=10, tariff=3e-3, max_steps=1e3 ):
        self.window = window
        self.fee = tariff
        data = pd.read_csv(path_to_data).as_matrix()
        arr = []
        for i in range(10):
            arr.append(data[:,i*3:(i+1)*3].reshape(-1, 3, 1))
        self.data = np.concatenate(arr, axis=2)
        self.max_steps = max_steps

        self.current_step = 0
        self.current_port = np.zeros((1, 11))
        self.current_port[0,0] = 1
        self.start = np.random.randint(self.window, self.data.shape[0] - self.max_steps)
        self.current_pos = self.start
        self.done = False


    def reset(self, random_start=True):
        self.current_step = 0
        self.current_port = np.zeros((1, 11))
        self.current_port[0,0] = 1
        self.start = np.random.randint(self.window, self.data.shape[0] - self.max_steps)
        self.current_pos = self.start
        self.done = False
        return self.data[self.start-self.window:self.start,:,:]


    def step(self, portfolio):
        if self.done:
            print('You are done, begin new try')
        else:
            if 0.99 < np.sum(portfolio) < 1.01:
                cost = np.sum(np.abs(portfolio[0, 1:] - self.current_port[0, 1:]))
                old_prices = self.data[self.current_pos, 0, :]
                new_prices = self.data[self.current_pos+1,0,:]
                y = np.concatenate([[1], np.divide(new_prices, old_prices)])
                new_value = np.sum(y*portfolio) - cost
                reward = np.log(new_value)

                self.current_port = (y*portfolio)/np.sum(y*portfolio)

                self.current_pos +=1
                self.current_step += 1
                if self.current_step == self.max_steps:
                    self.done = True
                return self.data[self.current_pos-self.window:self.current_pos,:,:], reward, self.done

            else:
                print('Portfolio weights must sum to unity')




