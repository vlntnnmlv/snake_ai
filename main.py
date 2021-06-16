from app import *

# snake_g = Game()
# snake_g.run_train()

snake_g = App("./data/snake.jpeg", "./data/food.png", "snake_ai", width = 410, height = 410)
pyglet.app.run()
