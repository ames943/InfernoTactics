import torch
import torch.nn as nn

# Example dummy RL Network
class FireRLModel(nn.Module):
    def __init__(self):
        super(FireRLModel, self).__init__()
        self.fc = nn.Linear(32, 3) # 32 sectors -> 3 actions (wait, helicopter, trench)

    def forward(self, state):
        return self.fc(state)

def get_action(state_tensor):
    # Load your trained model (in reality, you'd load state_dict once at startup)
    model = FireRLModel()
    model.eval()
    
    with torch.no_grad():
        q_values = model(state_tensor)
        action_idx = torch.argmax(q_values).item()
        
    return action_idx