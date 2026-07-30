import random

class PalisadesFireSim:
    def __init__(self):
        # 32 sectors (8x4). 0 = Safe, 1 = Fire, 2 = Burned, 3 = Defended
        self.grid = [0] * 32
        self.grid[10] = 1 # Start fire at Sector 10

    def step(self, action_idx, resource_type):
        # Apply the RL action
        if resource_type == 'helicopter':
            self.grid[action_idx] = 3 
            
        # Basic fire spread logic (example)
        # In a real sim, you'd use cellular automata or wind vectors here
        for i in range(len(self.grid)):
            if self.grid[i] == 1 and random.random() > 0.8:
                # Spread to a random neighbor
                neighbor = (i + 1) % 32
                if self.grid[neighbor] == 0:
                    self.grid[neighbor] = 1
                    
        return self.grid