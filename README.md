# drl-candlesticks-trader
The code for paper "An Automated Cryptocurrency Trading Approach Using Ensemble Deep Reinforcement Learning: Learn to Understand Candlesticks"

## Download

Candlesticks image data download: https://drive.google.com/file/d/1qX8K8CRnS3lmhA0Trd1Yk-i7_Cdchwbx/view?usp=sharing  
Add the unzipped folder "candlesticks" to folder "Image"  

## Run Example

Train all PPO agents in GPU with multi-resolution raw numerical data  
Type in the terminal python3 code/main.py -t train -g ppo -r multi  

Validate the first DQN agent in GPU with candlestick image  
Type in the terminal python3 code/main.py -t valid -g dqn -f 0 

## License

[MIT](https://choosealicense.com/licenses/mit/)
