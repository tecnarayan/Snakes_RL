import pygame
import sys
import random
from pygame.math import Vector2
import pygame.font
import torch
from model_QNet import model
from QTrainer import QTrainer

cell_size = 20  # dimention of 1 block in a grid
cell_number = 10  # number of blocks along the length of board
REPLAY_MEMORY = list()
REWARD = 0
ITERATION = 0
EXPLORATION = 0.99
GAME_NUMBER = 1

RECORD = 0
SCORE = 0

class SNAKE():
    def __init__(self):
        self.body = [Vector2(2,5),Vector2(1,5),Vector2(0,5)]
        self.direction = Vector2(1,0)

    def draw_snake(self):
        for block in self.body[1:]:
            snake_rect = pygame.Rect(int(block.x * cell_size ) , int(block.y * cell_size) , cell_size , cell_size)
            pygame.draw.rect(screen , (0 , 0 , 255), snake_rect) 
        for block in self.body[0:1]:
            snake_rect = pygame.Rect(int(block.x * cell_size ) , int(block.y * cell_size) , cell_size , cell_size)
            pygame.draw.rect(screen , (0 , 0 , 0), snake_rect)  

    def move_snake(self):
        body_copy = self.body[:-1]
        body_copy.insert(0,body_copy[0] + self.direction)
        self.body = body_copy[:]       

class FRUIT():
    def __init__(self):
        self.x = random.randint( 0 , cell_number -1)
        self.y = random.randint(0 , cell_number - 1)
        self.pos = Vector2(self.x , self.y)

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size ) , int(self.pos.y * cell_size) , cell_size , cell_size)
        pygame.draw.rect(screen , (255 , 0 , 0), fruit_rect)

    def repostion_fruit(self):
        self.x = random.randint( 0 , cell_number -1)
        self.y = random.randint(0 , cell_number - 1)
        self.pos = Vector2(self.x , self.y)


class MAIN():
    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()

    def update(self):
        self.snake.move_snake()
        font_size = 36
        font = pygame.font.Font(None, font_size)
        global SCORE
        score_text = font.render(f"Score: {SCORE}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

    def draw_elements(self):
        self.fruit.draw_fruit()
        self.snake.draw_snake()

    def check_fruit(self):
        if self.snake.body[0] + self.snake.direction == self.fruit.pos:
            self.snake.body.append(self.snake.body[-1])
            global SCORE 
            SCORE += 1
            global REPLAY_MEMORY
            REPLAY_MEMORY.append(0.5)
            global ITERATION
            ITERATION = 0
            while True:
                self.fruit.repostion_fruit()
                if self.fruit.pos in self.snake.body:
                    pass
                else:
                    break

    def check_wall(self):
        position =  self.snake.body[0] + self.snake.direction 
        if position.x < 0 or position.x >9 or position.y < 0 or position.y >9:
            global SCORE
            global RECORD
            global REPLAY_MEMORY
            REPLAY_MEMORY.append(-1)
            if SCORE > RECORD:
                RECORD = SCORE
            print(f" SCORE IS : {SCORE} || RECORD IS : {RECORD}")
            return 0
        else:
            return 1
    
    def check_body(self):
        position =  self.snake.body[0] + self.snake.direction
        if position in self.snake.body:
            global SCORE
            global RECORD
            if SCORE > RECORD:
                RECORD = SCORE
            print(f" SCORE IS : {SCORE} || RECORD IS : {RECORD}")
            global REPLAY_MEMORY
            REPLAY_MEMORY.append(-1)
            return 0
        else:
            return 1   

    def state_update(self,state):
        state[ 0, 0 , int(self.fruit.pos.y) , int(self.fruit.pos.x)] = 1
        for i in self.snake.body:
            state[ 0, 1 , int(i.y) , int(i.x)] = 1
        state[ 0 , 2 , int(self.snake.body[0].y) , int(self.snake.body[0].x)] = 1

    def reset(self):
        self.snake.body = [Vector2(2,5),Vector2(1,5),Vector2(0,5)]
        self.snake.direction = Vector2(1,0)
        global REPLAY_MEMORY
        REPLAY_MEMORY = []
        global SCORE
        SCORE = 0




pygame.init()  #  initialize the game

screen = pygame.display.set_mode((cell_size*cell_number ,cell_size*cell_number ))  #  to display the MAIN BACK-GROUND screen with dimention 400*400 [ base_lenght * height ]

clock = pygame.time.Clock()  #  helps to regulate loop speed [ game speed ] or else game will run fast if CPU is fast

## THE GAME LOOP :
##  --> Get player INPUT
##  --> POSITION elements
##  --> Draw GRAPHICS

main_game = MAIN()


# SCREEN_UPDATE = pygame.USEREVENT # to triger motion of snake every few millisecond and not all the time
# pygame.time.set_timer(SCREEN_UPDATE,1)  # 150 millisecond and it will be triggered

def play():

    while True:
        state = torch.zeros(1,3,10,10)
        action = torch.zeros(1,3)
        main_game.state_update(state)
        REPLAY_MEMORY.append(state)
        REPLAY_MEMORY.append(main_game.snake.direction)
        REPLAY_MEMORY.append(main_game.snake.body[-1].x)
        REPLAY_MEMORY.append(main_game.snake.body[-1].y)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
 
        model.eval()
        #action = model(state)
        #print("action ",action)
        num = random.random()
        if num >= EXPLORATION:
            action = model(state)
            action_val = torch.argmax(action , dim=1).item()
        else:
            action_val = random.randint(0 , 2)

        REPLAY_MEMORY.append(action_val)


        if action_val == 0:
            pass

        if action_val == 1:
            if main_game.snake.direction == Vector2(0,-1):
                main_game.snake.direction = Vector2(1,0)
            elif main_game.snake.direction == Vector2(-1,0):
                main_game.snake.direction = Vector2(0,-1)
            elif main_game.snake.direction == Vector2(1,0):
                main_game.snake.direction = Vector2(0,1)
            else:
                main_game.snake.direction = Vector2(-1,0)

        if action_val == 2:
            if main_game.snake.direction == Vector2(0,-1):
                main_game.snake.direction = Vector2(-1,0)
            elif main_game.snake.direction == Vector2(-1,0):
                main_game.snake.direction = Vector2(0,1)
            elif main_game.snake.direction == Vector2(1,0):
                main_game.snake.direction = Vector2(0,-1)
            else:
                main_game.snake.direction = Vector2(1,0)


        screen.fill((0 , 255 , 0))
        main_game.check_fruit()
        suspense = main_game.check_wall()
        if suspense == 0:
            return
        suspense = main_game.check_body()
        if suspense == 0:
            return
        if REPLAY_MEMORY[-1] != 0.5 and REPLAY_MEMORY[-1] != -1:
            REPLAY_MEMORY.append(0)
        main_game.update()
        main_game.draw_elements()
        pygame.display.update()
        global ITERATION
        ITERATION += 1
        if ITERATION >= 100:
            ITERATION = 0
            return
        clock.tick(100)  #  100 = framerate = number of times that for loop will get executed per-second

while True:
    print("GAME_NUMBER",GAME_NUMBER)
    GAME_NUMBER+=1
    print("EXPLORATION",EXPLORATION)
    play()
    QTrainer(model , REPLAY_MEMORY)
    EXPLORATION *= 0.99
    main_game.reset()