import torch
import torch.nn as nn
import torch.optim as optim
from pygame.math import Vector2

def emulate(state , action_val , direction_inintial , x , y):

    state_new = state.clone()
    
    direction_final = direction_inintial

    if action_val == 0:
        pass

    if action_val == 1:
        if direction_final == Vector2(0,-1):
            direction_final = Vector2(1,0)
        elif direction_final == Vector2(-1,0):
            direction_final = Vector2(0,-1)
        elif direction_final == Vector2(1,0):
            direction_final = Vector2(0,1)
        else:
            direction_final = Vector2(-1,0)

    if action_val == 2:
        if direction_final == Vector2(0,-1):
            direction_final = Vector2(-1,0)
        elif direction_final == Vector2(-1,0):
            direction_final = Vector2(0,1)
        elif direction_final == Vector2(1,0):
            direction_final = Vector2(0,-1)
        else:
            direction_final = Vector2(1,0)
        
    index_head_x, index_head_y = None, None
    for x in range(10):
        for y in range(10):
            if state[0, 2, x, y] == 1:
                index_head_x = x
                index_head_y = y
                break
     
    if index_head_x + direction_final.x > 9 or index_head_x + direction_final.x < 0 or index_head_y + direction_final.y > 9 or index_head_y + direction_final.y < 0:
        return 0
    
    if state[0,1,index_head_y + int(direction_final.y), index_head_x + int(direction_final.x)] == 1:
        return 0
    
    state_new[0,2,index_head_y , index_head_x] = 0
    state_new[0,2,index_head_y+ int(direction_final.y) , index_head_x + int(direction_final.x)] = 1
    state_new[0,1,index_head_y+ int(direction_final.y) , index_head_x + int(direction_final.x)] = 1
    state_new[0,2,y ,x] = 0
    
    if state[0,0,index_head_y + int(direction_final.y), index_head_x + int(direction_final.x)] == 1:
        return 0.5 , state_new
    
    else:
        return 0 , state_new
    
    

def QTrainer(model , REPLAY_MEMORY ):
    loss = nn.MSELoss()
    model.train()
    #optimizer = optim.SGD(model.parameters(),lr=0.01)
    optimizer = optim.Adam(model.parameters(),lr=0.005)

    length = len(REPLAY_MEMORY)
    if length > 240:
        REPLAY_MEMORY = REPLAY_MEMORY[-240:]
        length = 240

    for epoch in range(30):

        LEN = int(length/6)

        tl = 0

        for i in range(LEN):
            model.zero_grad()
            q = model(REPLAY_MEMORY[-1*(i+1)*6])
            q_star = torch.zeros(1,3)
            a_0 = emulate(REPLAY_MEMORY[-1*(i+1)*6] , 0 ,REPLAY_MEMORY[-1*(i+1)*6 + 1] ,REPLAY_MEMORY[-1*(i+1)*6 + 2] , REPLAY_MEMORY[-1*(i+1)*6 + 3] ) # return ( reward , state )
            a_1 = emulate(REPLAY_MEMORY[-1*(i+1)*6] , 1 ,REPLAY_MEMORY[-1*(i+1)*6 + 1] ,REPLAY_MEMORY[-1*(i+1)*6 + 2] , REPLAY_MEMORY[-1*(i+1)*6 + 3] )
            a_2 = emulate(REPLAY_MEMORY[-1*(i+1)*6] , 2 ,REPLAY_MEMORY[-1*(i+1)*6 + 1] ,REPLAY_MEMORY[-1*(i+1)*6 + 2] , REPLAY_MEMORY[-1*(i+1)*6 + 3] )
            
            if a_0 == 0:
                q_star[0,0] = -1
            else:
                q_star[0,0] = a_0[0] + 0.9*torch.max((model(a_0[1])) , dim=1)[0].item()
            if a_1 == 0:
                q_star[0,1] = -1
            else:
                q_star[0,1] = a_1[0] + 0.9*torch.max((model(a_1[1])) , dim=1)[0].item()
            if a_2 == 0:
                q_star[0,2] = -1
            else:
                q_star[0,2] = a_2[0] + 0.9*torch.max((model(a_2[1])) , dim=1)[0].item()
                
            #print("q =", q, "q* = ",q_star)
            l = loss(q , q_star*0.1 + q*0.9)
            #print("loss",l)
            tl += l.item()
            l.backward()
            optimizer.step()
        print(epoch,":",tl)