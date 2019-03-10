import ai as uct
import state, numpy

THINK_TIME = 3

ai = uct.ai(THINK_TIME, 'weights_1600.h5')
num_games = int(input('how many games do you want to play?'))

for episode in range(num_games):
	my_turn = int(input('which player do you want to play? (1 or -1)'))
	turn = 1
	game_over = False
	game = state.state()
	while not game_over:
		if turn == my_turn:
			print(game.get_raw_board().swapaxes(0,1))
			move = int(input('make your move.'))
		else:
			move = ai.get_move(game)
		game = game.move(move)
		eval = ai.model.predict_on_batch(numpy.array([game.get_processed_board()])) * game.turn
		print('Current ANN-only evaluation = %.5f' % eval)
		print('Current full evaluation = %.5f' % ai.get_eval(game))
		game_over = game.check_game()
		turn *= -1
	ai.flush_data(game.get_winner())
	if game.get_winner() == my_turn:
		print("you win!")
	elif game.get_winner() == -my_turn:
		print("you lose!")
	else:
		print("tie!")

