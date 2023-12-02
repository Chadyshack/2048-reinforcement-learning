import pygame
import sys
from game import Board  # Import your existing game class

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 400, 450  # Increase height to 450 to add space for the score
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2048 Game")

# Colors
background_color = (250, 248, 239)
tile_colors = {0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200), 
               8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95), 
               64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97), 
               512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)}

# Font
font = pygame.font.SysFont("arial", 40)

# Create a game instance
game = Board()

def draw_board():
    screen.fill(background_color)
    for i, row in enumerate(game.board):
        for j, value in enumerate(row):
            tile_color = tile_colors[value]
            pygame.draw.rect(screen, tile_color, (j*100, i*100, 100, 100))
            if value != 0:
                text_surface = font.render(str(value), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(j*100 + 50, i*100 + 50))
                screen.blit(text_surface, text_rect)

    # Draw the score below the game board
    score_text = font.render(f"Score: {game.score}", True, (0, 0, 0))
    score_rect = score_text.get_rect(center=(width // 2, height - 25))  # Position the score at the bottom
    screen.blit(score_text, score_rect)


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_w, pygame.K_UP):
                game.move_tiles('u')
            elif event.key in (pygame.K_s, pygame.K_DOWN):
                game.move_tiles('d')
            elif event.key in (pygame.K_a, pygame.K_LEFT):
                game.move_tiles('l')
            elif event.key in (pygame.K_d, pygame.K_RIGHT):
                game.move_tiles('r')

    # Draw the game board
    draw_board()

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()

