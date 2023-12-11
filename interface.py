import pygame
from game import Board

# Initialize Pygame
pygame.init()

# Set up the display and caption
width, height = 400, 450
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("2048 Game")

# Set styling for colors and font
background_color = (250, 248, 239)
tile_colors = {0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200), 
               8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95), 
               64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97), 
               512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)}
font = pygame.font.SysFont("arial", 40)

# Create a game instance
game = Board()

# Function to draw game board
def draw_board():
    # Fill screen with background color
    screen.fill(background_color)
    # Loop over each value in the game board
    for i, row in enumerate(game.board):
        for j, value in enumerate(row):
            # Get the color for this value (white if missing) and draw rectangle on screen
            tile_color = tile_colors.get(value, (255, 255, 255))
            pygame.draw.rect(screen, tile_color, (j * 100, i * 100, 100, 100))
            # If the value was not zero, draw the numerical value on this tile's center as well
            if value != 0:
                # If we are at the index of the last added tile, draw it red for distinction
                if (i, j) == game.last_added_tile:
                    text_surface = font.render(str(value), True, (255, 0, 0))
                else:
                    text_surface = font.render(str(value), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(j * 100 + 50, i * 100 + 50))
                screen.blit(text_surface, text_rect)

    # Draw the score below the game board
    score_text = font.render(f"Score: {game.score}", True, (0, 0, 0))
    score_rect = score_text.get_rect(center=(width // 2, height - 25))
    screen.blit(score_text, score_rect)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Get input from arrow keys or WASD on users keyboard for moves
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
