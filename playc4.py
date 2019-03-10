from __future__ import print_function

import state

game = state.state()
game_over = False

while game_over == False:
	print(game.get_raw_board())
	move = input('Make your move:')
	game = game.move(move-1)
	game_over = game.check_game()
print('player %d won!' % game.get_winner())