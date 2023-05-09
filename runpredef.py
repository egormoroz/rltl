from cityflow import Engine
from tqdm import trange

# Load config and run the simulation with a predefiend traffic light
# for a few steps to make sure everythign works correctly.

eng = Engine('data/predefined/config.json', thread_num=1)

for step in trange(1000):
    eng.next_step()

print('done')

